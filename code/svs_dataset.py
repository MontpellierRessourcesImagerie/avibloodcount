import numpy as np
import onnxruntime as ort
import os
import re
import struct

from PIL import Image
from torch.utils.data import Dataset

# Whole slide images easily exceed PIL's default decompression bomb limit of
# ~179 megapixels (the TIFFs processed here are ~317 megapixels). The images are
# loaded on purpose, so the limit is disabled instead of refusing to open them.
Image.MAX_IMAGE_PIXELS = None


class UnsupportedTiffError(Exception):
    """The TIFF cannot be read region-wise and needs a full decode."""


# TIFF field types / sizes used while parsing the image directory.
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
# 5 = RATIONAL and 10 = SRATIONAL are pairs of 32-bit integers.
_TYPE_FMT = {1: "B", 3: "H", 4: "I", 5: "II", 6: "b", 8: "h", 9: "i", 10: "ii",
             11: "f", 12: "d"}

# TIFF tags used to determine the physical size of a pixel.
_TAG_IMAGE_DESCRIPTION = 270
_TAG_X_RESOLUTION = 282
_TAG_Y_RESOLUTION = 283
_TAG_RESOLUTION_UNIT = 296
_OME_PHYSICAL_SIZE_X = re.compile(r'PhysicalSizeX="([0-9.eE+-]+)"')


def mpp_from_tags(tags):
    """Return micrometers per pixel of a TIFF directory, or None if unknown.

    OME-TIFF metadata (tag 270, ImageDescription) is preferred; the classic
    XResolution/ResolutionUnit tags are used as fallback.
    """
    description = tags.get(_TAG_IMAGE_DESCRIPTION)
    if isinstance(description, str):
        match = _OME_PHYSICAL_SIZE_X.search(description)
        if match and float(match.group(1)) > 0:
            return float(match.group(1))

    resolution = tags.get(_TAG_X_RESOLUTION)
    unit = tags.get(_TAG_RESOLUTION_UNIT, [1])
    if resolution and unit and int(unit[0]) in (2, 3):
        # Resolution is stored as pixels per inch (2) or per centimeter (3).
        pixels_per_unit = resolution[0] / resolution[1] if resolution[1] else 0
        if pixels_per_unit > 0:
            micrometers_per_unit = 25400.0 if int(unit[0]) == 2 else 10000.0
            return micrometers_per_unit / pixels_per_unit
    return None


def tiff_mpp(path):
    """Read the physical pixel size of any TIFF file, or None if unknown."""
    try:
        with open(path, "rb") as fp:
            _, tags = TiffRegionReader._read_tags(fp)
        return mpp_from_tags(tags)
    except (OSError, UnsupportedTiffError, struct.error):
        return None


class TiffRegionReader:
    """Region reader for uncompressed, 8-bit TIFF images.

    Whole slide TIFFs are frequently stored uncompressed and amount to ~1 GB of
    pixel data per image. PIL decodes such an image completely into memory as
    soon as the first region is cropped, which would happen once per data loader
    worker. Since the data is uncompressed, only the bytes belonging to the
    requested region are read from disk instead, keeping the memory footprint
    per worker in the megabyte range.
    """

    def __init__(self, path):
        self.path = path
        self._fp = None
        self._pid = None

        with open(path, "rb") as fp:
            self._endian, tags = self._read_tags(fp)

        try:
            self.width = self._scalar(tags, 256)
            self.height = self._scalar(tags, 257)
            self.samples = self._scalar(tags, 277, 1)
            compression = self._scalar(tags, 259, 1)
            planar = self._scalar(tags, 284, 1)
            bits = tags.get(258, [8])
            photometric = self._scalar(tags, 262, 1)
        except (KeyError, IndexError, TypeError) as exc:
            raise UnsupportedTiffError(f"incomplete TIFF directory: {exc}")

        if compression != 1:
            raise UnsupportedTiffError(f"compression {compression} is not supported")
        if planar != 1:
            raise UnsupportedTiffError("planar (non-chunky) pixel layout is not supported")
        if any(int(bit) != 8 for bit in bits) or self.samples not in (1, 3, 4):
            raise UnsupportedTiffError("only 8-bit grayscale/RGB/RGBA images are supported")
        if photometric not in (0, 1, 2):
            raise UnsupportedTiffError(f"photometric interpretation {photometric} is not supported")
        if self.width <= 0 or self.height <= 0:
            raise UnsupportedTiffError("invalid image dimensions")

        if 322 in tags and 323 in tags:
            self._tiled = True
            self._tile_w = self._scalar(tags, 322)
            self._tile_h = self._scalar(tags, 323)
            offsets, counts = tags.get(324), tags.get(325)
            if self._tile_w <= 0 or self._tile_h <= 0:
                raise UnsupportedTiffError("invalid tile size")
        else:
            self._tiled = False
            rows_per_strip = self._scalar(tags, 278, self.height)
            # The TIFF default for RowsPerStrip is 2**32-1, meaning one single strip.
            self._rows_per_strip = rows_per_strip if 0 < rows_per_strip <= self.height else self.height
            offsets, counts = tags.get(273), tags.get(279)

        if not offsets:
            raise UnsupportedTiffError("no strip/tile offsets found")

        self._offsets = np.asarray(offsets, dtype=np.int64)
        self._counts = None if counts is None else np.asarray(counts, dtype=np.int64)
        self._row_bytes = self.width * self.samples

        # Physical pixel size in micrometers, if the file declares one.
        self.mpp = mpp_from_tags(tags)

        # Bail out early if the layout does not match the uniform, uncompressed
        # assumption, so the caller can fall back to decoding the whole image.
        if self._tiled:
            self._check_tile_layout()
        else:
            expected = self._expected_strip_sizes()
            if expected is None or (self._counts is not None and not np.array_equal(expected, self._counts)):
                raise UnsupportedTiffError("irregular strip layout")

    def _check_tile_layout(self):
        """Verify that all tiles form a regular, uncompressed grid."""
        tiles_x = -(-self.width // self._tile_w)
        tiles_y = -(-self.height // self._tile_h)
        if len(self._offsets) != tiles_x * tiles_y:
            raise UnsupportedTiffError("unexpected number of tiles")
        if self._counts is None:
            return
        expected = np.full(len(self._offsets), self._tile_w * self._tile_h * self.samples,
                           dtype=np.int64)
        clipped = np.array([
            min(self._tile_w, self.width - (i % tiles_x) * self._tile_w)
            * min(self._tile_h, self.height - (i // tiles_x) * self._tile_h)
            * self.samples
            for i in range(len(self._offsets))
        ], dtype=np.int64)
        mismatch = (self._counts != expected) & (self._counts != clipped)
        if mismatch.any():
            raise UnsupportedTiffError("irregular tile layout")

    @staticmethod
    def _scalar(tags, tag, default=None):
        values = tags.get(tag)
        if values is None:
            if default is None:
                raise KeyError(tag)
            return int(default)
        return int(values[0])

    @staticmethod
    def _read_tags(fp):
        """Return the byte order and all numeric tags of the first IFD."""
        header = fp.read(8)
        if len(header) < 8:
            raise UnsupportedTiffError("truncated TIFF header")
        if header[:2] == b"II":
            endian = "<"
        elif header[:2] == b"MM":
            endian = ">"
        else:
            raise UnsupportedTiffError("not a TIFF file")

        magic, ifd_offset = struct.unpack(endian + "HI", header[2:8])
        if magic == 43:
            raise UnsupportedTiffError("BigTIFF is not supported")
        if magic != 42:
            raise UnsupportedTiffError("not a TIFF file")

        fp.seek(ifd_offset)
        (count,) = struct.unpack(endian + "H", fp.read(2))
        tags = {}
        for i in range(count):
            fp.seek(ifd_offset + 2 + 12 * i)
            entry = fp.read(12)
            if len(entry) < 12:
                raise UnsupportedTiffError("truncated TIFF directory")
            tag, value_type, length = struct.unpack(endian + "HHI", entry[:8])
            size = _TYPE_SIZE.get(value_type)
            fmt = _TYPE_FMT.get(value_type)
            if tag == _TAG_IMAGE_DESCRIPTION and value_type == 2:
                # ASCII: needed for the OME-TIFF physical pixel size.
                if length <= 4:
                    data = entry[8:8 + length]
                else:
                    (value_offset,) = struct.unpack(endian + "I", entry[8:12])
                    fp.seek(value_offset)
                    data = fp.read(length)
                tags[tag] = data.rstrip(b"\x00").decode("utf-8", "replace")
                continue
            if size is None or fmt is None:
                continue  # ASCII or unsupported type: not needed here
            total = size * length
            if total <= 4:
                data = entry[8:8 + total]
            else:
                (value_offset,) = struct.unpack(endian + "I", entry[8:12])
                fp.seek(value_offset)
                data = fp.read(total)
            if len(data) < total:
                raise UnsupportedTiffError("truncated TIFF tag value")
            tags[tag] = list(struct.unpack(endian + fmt * length, data))
        return endian, tags

    def _expected_strip_sizes(self):
        """Byte size of every strip, or None if the count of strips is wrong."""
        expected_count = -(-self.height // self._rows_per_strip)
        if len(self._offsets) != expected_count:
            return None
        starts = np.arange(expected_count, dtype=np.int64) * self._rows_per_strip
        rows = np.minimum(self._rows_per_strip, self.height - starts)
        return rows * self._row_bytes

    def _file(self):
        # Data loader workers inherit this object via fork(); reopen the file so
        # that every process owns its own file handle and read position.
        if self._fp is None or self._pid != os.getpid():
            self._fp = open(self.path, "rb")
            self._pid = os.getpid()
        return self._fp

    def _read_bytes(self, offset, count):
        fp = self._file()
        fp.seek(int(offset))
        data = fp.read(int(count))
        if len(data) != count:
            raise UnsupportedTiffError("unexpected end of file")
        return data

    @staticmethod
    def _to_rgb(arr):
        """Convert an (h, w, samples) buffer to an RGB image."""
        if arr.shape[2] == 3:
            return arr
        if arr.shape[2] == 1:
            return np.repeat(arr, 3, axis=2)
        return arr[:, :, :3]

    def read_region(self, left, top, width, height):
        """Return an (height, width, 3) uint8 RGB array for the requested region.

        Areas outside of the image are filled with black.
        """
        image = np.zeros((height, width, 3), dtype=np.uint8)
        x0, y0 = max(left, 0), max(top, 0)
        x1, y1 = min(left + width, self.width), min(top + height, self.height)
        if x1 <= x0 or y1 <= y0:
            return image

        if self._tiled:
            region = self._read_tiled(x0, y0, x1, y1)
        else:
            region = self._read_strips(x0, y0, x1, y1)
        image[y0 - top:y1 - top, x0 - left:x1 - left] = self._to_rgb(region)
        return image

    def _read_strips(self, x0, y0, x1, y1):
        rps = self._rows_per_strip
        samples = self.samples
        out = np.empty((y1 - y0, x1 - x0, samples), dtype=np.uint8)

        if rps == 1 and x1 - x0 < self.width:
            # One strip per row: seek directly into the row and read only the
            # bytes of the requested columns instead of whole rows.
            offset = self._offsets
            count = (x1 - x0) * samples
            shift = x0 * samples
            for y in range(y0, y1):
                fp = self._file()
                fp.seek(int(offset[y]) + shift)
                data = fp.read(count)
                if len(data) != count:
                    raise UnsupportedTiffError("unexpected end of file")
                out[y - y0] = np.frombuffer(data, np.uint8).reshape(-1, samples)
            return out

        first, last = y0 // rps, (y1 - 1) // rps
        for strip in range(first, last + 1):
            strip_start = strip * rps
            rows = min(rps, self.height - strip_start)
            buf = self._read_bytes(self._offsets[strip], rows * self._row_bytes)
            arr = np.frombuffer(buf, np.uint8).reshape(rows, self.width, samples)
            # Overlap between strip and requested region.
            abs0, abs1 = max(y0, strip_start), min(y1, strip_start + rows)
            out[abs0 - y0:abs1 - y0] = arr[abs0 - strip_start:abs1 - strip_start, x0:x1]
        return out

    def _read_tiled(self, x0, y0, x1, y1):
        tw, th = self._tile_w, self._tile_h
        tiles_x = -(-self.width // tw)
        first_x, last_x = x0 // tw, (x1 - 1) // tw
        first_y, last_y = y0 // th, (y1 - 1) // th
        out = np.empty((y1 - y0, x1 - x0, self.samples), dtype=np.uint8)
        for ty in range(first_y, last_y + 1):
            for tx in range(first_x, last_x + 1):
                index = ty * tiles_x + tx
                if index >= len(self._offsets):
                    raise UnsupportedTiffError("tile index out of range")
                tile_w = min(tw, self.width - tx * tw)
                tile_h = min(th, self.height - ty * th)
                count = int(self._counts[index]) if self._counts is not None else tile_w * tile_h * self.samples
                buf = self._read_bytes(self._offsets[index], count)
                arr = np.frombuffer(buf, np.uint8)
                if count == tw * th * self.samples:
                    arr = arr.reshape(th, tw, self.samples)
                elif count == tile_w * tile_h * self.samples:
                    arr = arr.reshape(tile_h, tile_w, self.samples)
                else:
                    raise UnsupportedTiffError("irregular tile size")
                # Overlap between tile and requested region.
                ox0, oy0 = max(x0, tx * tw), max(y0, ty * th)
                ox1, oy1 = min(x1, tx * tw + tile_w), min(y1, ty * th + tile_h)
                if ox1 <= ox0 or oy1 <= oy0:
                    continue
                out[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0] = arr[oy0 - ty * th:oy1 - ty * th,
                                                                 ox0 - tx * tw:ox1 - tx * tw]
        return out

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None
            self._pid = None


class SvsDataset(Dataset):

    TILE_WIDTH = 512
    TILE_HEIGHT = 384

    # Basic Instantiation
    def __init__(self, file_path, model_path, thresh, target_mpp=None):
        self.file_path = file_path
        self.thresh = thresh
        ext = os.path.splitext(file_path)[1].lower()
        mpp = None

        if ext == ".svs":
            from openslide import OpenSlide
            self.img_type = "svs"
            self.osr = OpenSlide(file_path)
            self.tiff_reader = None
            self.pil_image = None
            self._pil_pid = None
            self.img_width = self.osr.level_dimensions[0][0]
            self.img_height = self.osr.level_dimensions[0][1]
            try:
                mpp = float(self.osr.properties.get("openslide.mpp-x"))
            except (TypeError, ValueError):
                mpp = None
        elif ext in (".tif", ".tiff"):
            self.img_type = "tiff"
            self.osr = None
            self.pil_image = None
            self._pil_pid = None
            try:
                # Fast path: read only the requested region from disk.
                self.tiff_reader = TiffRegionReader(file_path)
            except UnsupportedTiffError:
                # Fall back to PIL, which decodes the complete image when cropped.
                self.tiff_reader = None
                self.pil_image = Image.open(file_path)
                self._pil_pid = os.getpid()
            self.img_width = self.tiff_reader.width if self.tiff_reader else self.pil_image.width
            self.img_height = self.tiff_reader.height if self.tiff_reader else self.pil_image.height
            mpp = self.tiff_reader.mpp if self.tiff_reader else tiff_mpp(file_path)
        else:
            raise ValueError(f"Unsupported file format: '{ext}'. Supported formats are .svs, .tif, .tiff")

        # The models expect tiles at the physical resolution they were trained
        # at (target_mpp). Images stored at a finer resolution (e.g. 100x
        # scans) are read with a proportionally larger region and resampled, so
        # that every tile covers the same physical area as a training tile.
        self.mpp = mpp
        self.scale = 1.0
        if target_mpp and mpp:
            # Number of native pixels that correspond to one pixel at the
            # target resolution: a finer image (smaller mpp) needs a larger
            # native region per tile.
            self.scale = target_mpp / mpp

        # Tile grid, expressed in the resampled (training scale) coordinate
        # system. The tile counts are rounded *up* so that the whole image is
        # covered: with floor division the leftover strip right of / below the
        # last full tile was never visited, which dropped up to ~28% of these
        # TIFFs (one tile spans 4371 native px after the 8.5x resample). The
        # last column and row are clamped to the image border in __getitem__.
        self.effective_width = int(self.img_width / self.scale)
        self.effective_height = int(self.img_height / self.scale)
        self.n_horizontal_tiles = -(-self.effective_width // self.TILE_WIDTH)
        self.n_vertical_tiles = -(-self.effective_height // self.TILE_HEIGHT)
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Length of the Dataset
    def __len__(self):
        return self.n_vertical_tiles * self.n_horizontal_tiles

    def _pil(self):
        """Return the PIL image, reopened after fork().

        DataLoader workers inherit this object via fork(), i.e. they share the
        parent's file descriptor *and its read position*. Concurrent
        seek()/read() pairs of several workers then interleave and libtiff
        decodes garbage from wrong offsets. The image is therefore reopened
        whenever it is used from a different process than the one that opened
        it, so every worker owns a private file handle.
        """
        if self._pil_pid != os.getpid():
            if self.pil_image is not None:
                self.pil_image.close()
            self.pil_image = Image.open(self.file_path)
            self._pil_pid = os.getpid()
        return self.pil_image

    @classmethod
    def _tile_origin(cls, index, tile_size, extent):
        """Start of tile ``index`` along one axis.

        The last tile is shifted back so that it ends exactly at the image
        border instead of running past it, i.e. it overlaps its neighbour
        instead of leaving an unanalysed strip.
        """
        return min(index * tile_size, max(extent - tile_size, 0))

    def tile_origins(self):
        """Start positions of all tiles along x and y (resampled pixels).

        infer.py uses them to recognise the boxes of a cell that a tile
        border cut in half: each half ends exactly on an edge of the tile it
        came from, and those edges are derived from these origins.
        """
        x = [self._tile_origin(i, self.TILE_WIDTH, self.effective_width)
             for i in range(self.n_horizontal_tiles)]
        y = [self._tile_origin(i, self.TILE_HEIGHT, self.effective_height)
             for i in range(self.n_vertical_tiles)]
        return x, y

    # Fetch an item from the Dataset
    def __getitem__(self, idx):
        vert_pos, horiz_pos = divmod(idx, self.n_horizontal_tiles)
        # Tile position in the resampled (training scale) coordinate system.
        # The last column and row end at the image border, so they overlap
        # their neighbour; infer.py merges the detections of both.
        left = self._tile_origin(horiz_pos, self.TILE_WIDTH, self.effective_width)
        top = self._tile_origin(vert_pos, self.TILE_HEIGHT, self.effective_height)

        if self.scale == 1.0:
            native_left, native_top = left, top
            native_width, native_height = self.TILE_WIDTH, self.TILE_HEIGHT
        else:
            native_left = int(round(left * self.scale))
            native_top = int(round(top * self.scale))
            native_width = int(round(self.TILE_WIDTH * self.scale))
            native_height = int(round(self.TILE_HEIGHT * self.scale))

        if self.img_type == "svs":
            r = (self.osr.read_region((native_left, native_top), 0,
                                      (native_width, native_height))).convert('RGB')
            tile_arr = np.array(r)
            r.close()
        elif self.tiff_reader is not None:
            tile_arr = self.tiff_reader.read_region(native_left, native_top,
                                                    native_width, native_height)
        else:
            region = self._pil().crop((native_left, native_top,
                                       native_left + native_width,
                                       native_top + native_height))
            tile_arr = np.array(region.convert('RGB'))

        if self.scale != 1.0:
            # Resample the larger native region to the tile size expected by
            # the models (anti-aliased to not alias fine cell structures).
            tile_arr = np.asarray(Image.fromarray(tile_arr).resize(
                (self.TILE_WIDTH, self.TILE_HEIGHT), Image.LANCZOS))

        # run model
        tile = tile_arr[np.newaxis, ...]

        results_ort = self.sess.run(["Identity:0"], {"input_1:0": tile.astype(np.float32)})
        model_decision = False
        if results_ort[0][0][0] >= self.thresh:
            model_decision = True
        if idx == self.__len__():
            self.close()
        if model_decision:
            return {
                "tile": tile,
                "top": top,
                "left": left,
            }
        else:
            return None

    def close(self):
        if self.osr is not None:
            self.osr.close()
        if self.tiff_reader is not None:
            self.tiff_reader.close()
        if self.pil_image is not None:
            self.pil_image.close()
            self.pil_image = None
            self._pil_pid = None
