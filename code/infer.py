import argparse
import json
import math
import os
import time
import torch
import warnings

from collections import namedtuple

from detector import DetModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from svs_dataset import SvsDataset

# Classes for the YOLO annotation files written next to the input images for
# visual QC with napari-simpleannotate. Index = class_id - 1 as used by the
# detector. The colors match the default palette of the plugin.
YOLO_CLASSES = [
    "Erythrocyte (with nucleus)",
    "Eosinophil",
    "Lymphocyte",
    "Heterophil",
    "Erythrocyte (cell) with parasite",
    "Monocyte",
    "Basophil",
]
YOLO_COLORS = ["red", "green", "blue", "cyan", "magenta", "yellow", "black"]

# Maximum distance (in resampled pixels) between two detections that are
# treated as the same cell by proximity. Two views of one cell land on top
# of each other (measured p90: 0.4 px), while the centres of two different
# cells are never closer than ~30 px (nuclei are about 10 um across at
# 0.2525 um/px).
MERGE_RADIUS = 8.0

# Cell size bound: no pair of detections that belongs to one cell is ever
# farther apart than this, so it doubles as the bucket size of the candidate
# search (one bucket + its 8 neighbours cover twice this distance).
CELL_LIMIT = 44.0

# Two tiles that meet at a border cut every cell straddling it in half: each
# tile returns only the part of the cell inside it, so the two boxes face each
# other across the border with their edges right on it (measured: within 1
# resampled px), while their centres are up to half a cell apart -- farther
# than MERGE_RADIUS, so proximity alone double counts them. The halves are
# recognised by their edges sitting exactly on an edge of the tile the box
# could have come from (a box that merely ends near a border is a whole cell
# of its own). Measured on the TIFF test set: pairs merged this way are one
# nucleus split by the border; pairs of two whole cells keep at least one box
# away from the border or on the wrong tile's edge.
BORDER_EDGE_TOL = 1.5   # how far a box edge may miss the tile border
BORDER_PERP = 3.5       # minimum overlap of the halves on the other axis

# A view of a whole cell and the clipped half of the same cell overlap almost
# completely (one box inside the other), which proximity does not catch when
# the centres are farther apart than MERGE_RADIUS. Two different cells can
# never overlap this much: their centres are at least ~30 px apart.
CONTAIN_RATIO = 0.8

# A detection in resampled image coordinates: (cx, cy) is the centre of the
# box, centrality is its distance to the centre of the tile it came from.
Detection = namedtuple("Detection", "cx cy x1 y1 x2 y2 class_id score centrality")


def write_yolo_metadata(directory):
    """Write class.yaml and colors.yaml for napari-simpleannotate."""
    with open(os.path.join(directory, "class.yaml"), "w") as f:
        f.write("names:\n")
        for class_id, name in enumerate(YOLO_CLASSES):
            f.write(f"  {class_id}: {name}\n")
    with open(os.path.join(directory, "colors.yaml"), "w") as f:
        for class_id, color in enumerate(YOLO_COLORS):
            f.write(f"{class_id}: {color}\n")


def to_resampled(tile_detections, tile):
    """Move detections of one tile into the resampled image coordinates.

    Also records how far the cell is from the centre of its tile, which is
    used to pick the best of several views of the same cell.
    """
    left, top = tile["left"], tile["top"]
    half_width, half_height = SvsDataset.TILE_WIDTH / 2, SvsDataset.TILE_HEIGHT / 2
    moved = []
    for x1, y1, x2, y2, class_id, score in tile_detections:
        centre_x, centre_y = (x1 + x2) / 2, (y1 + y2) / 2
        moved.append(Detection(centre_x + left, centre_y + top,
                               x1 + left, y1 + top, x2 + left, y2 + top,
                               class_id, score,
                               math.hypot(centre_x - half_width, centre_y - half_height)))
    return moved


def tile_edges(detection, x_origins, y_origins):
    """Edges of the tiles that could have produced this detection.

    Returns (lefts, rights, tops, bottoms): the start/end positions of every
    tile whose span contains the box centre. A box clipped by its tile ends
    exactly on one of these edges.
    """
    lefts = [o for o in x_origins if o <= detection.cx < o + SvsDataset.TILE_WIDTH]
    rights = [o + SvsDataset.TILE_WIDTH for o in lefts]
    tops = [o for o in y_origins if o <= detection.cy < o + SvsDataset.TILE_HEIGHT]
    bottoms = [o + SvsDataset.TILE_HEIGHT for o in tops]
    return lefts, rights, tops, bottoms


def clipped_halves(first, first_edges, second, second_edges):
    """True if the two boxes are the halves of one cell cut by a tile border.

    Both boxes must end exactly on an edge of a tile they could have come
    from and face each other across the same border (their facing edges at
    most CELL_LIMIT apart, i.e. within one cell), and they must overlap
    meaningfully on the axis along the border. A pair of two different cells
    fails: at least one of the two boxes is a whole cell whose edges are not
    tile edges, or the boxes are too far apart on the border axis.
    """
    if first.class_id != second.class_id:
        return False
    for (a, a_edges, b, b_edges) in ((first, first_edges, second, second_edges),
                                     (second, second_edges, first, first_edges)):
        # a ends on the right edge of its tile, b starts on the left edge
        # of its tile, both at the same border.
        for p_a in a_edges[1]:
            if abs(a.x2 - p_a) > BORDER_EDGE_TOL:
                continue
            for p_b in b_edges[0]:
                if abs(b.x1 - p_b) > BORDER_EDGE_TOL:
                    continue
                if abs(b.x1 - a.x2) > CELL_LIMIT:
                    continue
                if min(a.y2, b.y2) - max(a.y1, b.y1) >= BORDER_PERP:
                    return True
        # same along y: a ends on the bottom edge, b starts on the top edge.
        for p_a in a_edges[3]:
            if abs(a.y2 - p_a) > BORDER_EDGE_TOL:
                continue
            for p_b in b_edges[2]:
                if abs(b.y1 - p_b) > BORDER_EDGE_TOL:
                    continue
                if abs(b.y1 - a.y2) > CELL_LIMIT:
                    continue
                if min(a.x2, b.x2) - max(a.x1, b.x1) >= BORDER_PERP:
                    return True
    return False


def contains_cell(outer, inner):
    """True if one box lies almost entirely inside the other (same class).

    This is a whole-cell view next to a clipped half of the same cell, or the
    model drawing two boxes for one nucleus; two different cells never
    overlap this much because their centres are >= ~30 px apart.
    """
    if outer.class_id != inner.class_id:
        return False
    overlap_x = min(outer.x2, inner.x2) - max(outer.x1, inner.x1)
    overlap_y = min(outer.y2, inner.y2) - max(outer.y1, inner.y1)
    if overlap_x <= 0 or overlap_y <= 0:
        return False
    area_outer = (outer.x2 - outer.x1) * (outer.y2 - outer.y1)
    area_inner = (inner.x2 - inner.x1) * (inner.y2 - inner.y1)
    return overlap_x * overlap_y >= CONTAIN_RATIO * min(area_outer, area_inner)


def merge_detections(detections, borders):
    """Return one detection per cell.

    Three signatures count as the same cell, in order of how sure we are:

    1. proximity: the centres are within MERGE_RADIUS. The tile grid is
       clamped to the image border, so the last column and row overlap their
       neighbour and a cell inside the overlap is returned by both tiles.
    2. the two halves of one cell cut by a tile border (see clipped_halves),
       whose centres can be up to half a cell apart.
    3. one box almost inside the other (see contains_cell): a whole-cell view
       next to a clipped half, which can also be farther apart than
       MERGE_RADIUS.

    Merging (rather than assigning each cell to a fixed tile) also keeps a
    cell that only one of the two tiles managed to detect. The view in which
    the cell lies closest to the centre of its tile wins, because that is
    where a detection sees the whole cell with the most context.
    """
    x_origins, y_origins = borders
    edges = [tile_edges(d, x_origins, y_origins) for d in detections]
    buckets = {}
    dead = {}

    def live(slot):
        while slot in dead:
            slot = dead[slot]
        return slot

    def bucket_of(detection):
        return (int(detection.cx // CELL_LIMIT), int(detection.cy // CELL_LIMIT))

    def same_cell(index, other):
        a, b = detections[index], detections[other]
        if (abs(a.cx - b.cx) <= MERGE_RADIUS and abs(a.cy - b.cy) <= MERGE_RADIUS):
            return True
        if clipped_halves(a, edges[index], b, edges[other]):
            return True
        return contains_cell(a, b) or contains_cell(b, a)

    for index, detection in enumerate(detections):
        key = bucket_of(detection)
        matches = set()
        for step_x in (-1, 0, 1):
            for step_y in (-1, 0, 1):
                for slot in buckets.get((key[0] + step_x, key[1] + step_y), ()):
                    # Test the pair against the view's own box even if that
                    # view has already been merged away: a fragment can pair
                    # with the next half of the same cell through a whole
                    # view that sits in between.
                    if same_cell(index, slot):
                        matches.add(slot)
        buckets.setdefault(key, []).append(index)
        if not matches:
            continue
        # The same cell seen by several tiles: keep the best view of it.
        # Merged-away views are represented by the view that replaced them.
        roots = {live(slot) for slot in matches}
        winner = index
        for root in roots:
            if (detections[root].centrality, -detections[root].score) < \
                    (detections[winner].centrality, -detections[winner].score):
                winner = root
        if winner != index:
            dead[index] = winner
        for root in roots:
            if root != winner:
                dead[root] = winner
    # Every losing view points at the view that replaced it; the views that
    # were never replaced are the surviving one per cell.
    return [detection for index, detection in enumerate(detections)
            if index not in dead]


def yolo_lines(detections, scale, image_width, image_height):
    """Map detections to normalized YOLO lines of the full image.

    The model sees tiles resampled to the training resolution, while the
    annotation coordinates have to be relative to the original image that
    napari opens, so the detections (given in the resampled coordinate
    system) are converted back to native image pixels first. Coordinates are
    clipped to the image like the plugin does when it saves annotations.
    """
    lines = []
    for detection in detections:
        x_min = min(max(detection.x1 * scale / image_width, 0.0), 1.0)
        x_max = min(max(detection.x2 * scale / image_width, 0.0), 1.0)
        y_min = min(max(detection.y1 * scale / image_height, 0.0), 1.0)
        y_max = min(max(detection.y2 * scale / image_height, 0.0), 1.0)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        lines.append(f"{detection.class_id - 1} {x_center:.6f} {y_center:.6f} "
                     f"{x_max - x_min:.6f} {y_max - y_min:.6f}")
    return lines


def parse_args():
    parser = argparse.ArgumentParser(
        description='Counting blood cells in avian whole slide blood smear images.',
    )
    parser.add_argument(
        '--path',
        dest='path',
        help='A directory or a single SVS or TIFF file.',
        default='/data/input/',
        type=str,
    )
    parser.add_argument(
        '--output-dir', '-o',
        dest='output_dir',
        help='Results in JSON format will be saved in this directory.',
        default='/data/output/',
        type=str,
    )
    parser.add_argument(
        '--config', '-c',
        dest='cfg',
        help='Config file for instance segmentation model (YAML format).',
        default="../configs/MS_R_101_BiFPN_3x_sem.yaml",
        type=str
    )
    parser.add_argument(
        '--cls-model',
        dest='cls_model',
        help='Countability classification model in ONNX format.',
        default="/data/models/efficientNet_B0.onnx",
        type=str
    )
    parser.add_argument(
        '--det-model',
        dest='det_model',
        help='Instance segmentation model.',
        default="/data/models/condInst_R101.pth",
        type=str
    )
    parser.add_argument(
        '--countability-thresh', '-t',
        dest='cls_thresh',
        help='Score threshold for the instance segmentation model.',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--detection-thresh', '-d',
        dest='det_thresh',
        help='Score threshold for the instance segmentation model.',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--target-mpp',
        dest='target_mpp',
        help='Physical pixel size in micrometers the models were trained on (40x '
             'scans of the training slides). Images with a different resolution '
             'are resampled accordingly, using the physical size stored in the '
             'file metadata. Use 0 to disable resampling.',
        default=0.2525,
        type=float
    )
    parser.add_argument(
        '--no-yolo',
        dest='write_yolo',
        action='store_false',
        help='Do not write YOLO annotation files (class.yaml, colors.yaml and '
             '<image>.txt next to each input image) used for visual QC with '
             'napari-simpleannotate.',
    )
    parser.add_argument(
        '--gpu',
        dest='gpu',
        help='GPU ID to be used or None for CPU-only inference.',
        type=int
    )
    return parser.parse_args()


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def main_loop(svs_files):
    os.makedirs(args.output_dir, exist_ok=True)

    multi_file = len(svs_files) > 1
    target_mpp = args.target_mpp if args.target_mpp > 0 else None

    model = DetModel(cfg=args.cfg, gpu=args.gpu, model=args.det_model, thresh=args.det_thresh)
    if args.write_yolo:
        write_yolo_metadata(args.path)
    svs_loop = tqdm(svs_files, disable=not multi_file)
    svs_loop.set_description("Overall")
    for filename in svs_loop:
        # loc_start = time.time()
        dataset = SvsDataset(os.path.join(args.path, filename), args.cls_model, args.cls_thresh,
                             target_mpp=target_mpp)
        if dataset.scale != 1.0:
            print(f"{filename}: {dataset.mpp:g} um/px -> resampled x{dataset.scale:.3f} to "
                  f"{target_mpp:g} um/px ({len(dataset)} tiles)", flush=True)
        elif target_mpp and dataset.mpp is None:
            print(f"Warning: {filename} does not declare a physical pixel size, "
                  f"no resampling applied.", flush=True)
        detections = []
        loader = DataLoader(dataset,
                            batch_size=8,
                            shuffle=False,
                            num_workers=os.cpu_count() - 1,
                            pin_memory=True,
                            collate_fn=collate_fn)
        tile_loop = tqdm(loader, leave=False)
        tile_loop.set_description("Current file")
        for i, item in enumerate(tile_loop):
            if item:
                images = [img['tile'][..., ::-1].squeeze(axis=0) for img in item]
                for tile_detections, tile in zip(model.predict(images), item):
                    detections.extend(to_resampled(tile_detections, tile))
        # A tile border cuts every straddling cell in half and the clamped
        # last column/row overlaps its neighbour, so both would return the
        # same cell twice: merge the detections before counting.
        detections = merge_detections(detections, dataset.tile_origins())
        for detection in detections:
            model.counts[model.class_ids[str(detection.class_id)]] += 1
        if args.write_yolo:
            annotations = yolo_lines(detections, dataset.scale,
                                     dataset.img_width, dataset.img_height)
        counts = model.counts
        model.counts.update({'Overall tiles': len(dataset)})
        del counts[model.class_ids['5']]
        # loc_end = time.time()
        # print(time.strftime("%Hh%Mm%Ss", time.gmtime(loc_end - loc_start)))
        with open(os.path.join(args.output_dir, filename + ".txt"), 'w') as f:
            json.dump(counts, f)
        if args.write_yolo:
            # napari-simpleannotate expects <image_name>.txt next to the image.
            annotation_file = os.path.splitext(filename)[0] + ".txt"
            annotation_path = os.path.join(args.path, annotation_file)
            if annotations:
                with open(annotation_path, "w") as f:
                    f.write("\n".join(annotations) + "\n")
            elif os.path.exists(annotation_path):
                # No detections in this run: drop a stale file of an older run.
                os.remove(annotation_path)
            print(f"{filename}: {len(annotations)} annotated cells -> {annotation_file}", flush=True)
        model.reset_counter()


def get_input_files(path):
    if os.path.isfile(path):
        if path.endswith((".svs", ".SVS", ".tif", ".tiff", ".TIF", ".TIFF")):
            # Normalize to directory + file name: main_loop joins both and
            # writes the result file next to the image otherwise.
            args.path = os.path.dirname(os.path.abspath(path))
            svs_files = [os.path.basename(path)]
        else:
            raise argparse.ArgumentTypeError(f"Expecting input files to be in SVS, TIFF file format! Found "
                                             f"'{os.path.splitext(path)[-1]}'.")
    elif os.path.isdir(path):
        svs_files = [item for item in os.listdir(path) if item.endswith((".svs", ".SVS", ".tif", ".tiff", ".TIF", ".TIFF"))]
        if not svs_files:
            print(f"Warning: Directory {path} does not contain any SVS or TIFF file!")
            return
    else:
        raise FileNotFoundError
    return svs_files


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning,
                            message="torch.meshgrid: in an upcoming release, "
                                    "it will be required to pass the indexing argument.")
    args = parse_args()

    # set GPU
    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # avoid problems with shm
    torch.multiprocessing.set_sharing_strategy('file_system')

    # start analysis
    input_files = get_input_files(args.path)
    if input_files:
        start = time.time()
        main_loop(input_files)
        end = time.time()
        print(time.strftime("%Hh%Mm%Ss", time.gmtime(end - start)))
    print("Done.")
