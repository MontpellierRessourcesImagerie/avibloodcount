import argparse
import json
import os
import time
import torch
import warnings

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


def write_yolo_metadata(directory):
    """Write class.yaml and colors.yaml for napari-simpleannotate."""
    with open(os.path.join(directory, "class.yaml"), "w") as f:
        f.write("names:\n")
        for class_id, name in enumerate(YOLO_CLASSES):
            f.write(f"  {class_id}: {name}\n")
    with open(os.path.join(directory, "colors.yaml"), "w") as f:
        for class_id, color in enumerate(YOLO_COLORS):
            f.write(f"{class_id}: {color}\n")


def yolo_lines(detections, tile, scale, image_width, image_height):
    """Map detections of one tile to normalized YOLO lines of the full image.

    The model sees tiles resampled to the training resolution, while the
    annotation coordinates have to be relative to the original image that
    napari opens, so the tile position and the boxes are converted back to
    native image pixels first. Coordinates are clipped to the image like the
    plugin does when it saves annotations.
    """
    left = tile["left"] * scale
    top = tile["top"] * scale
    lines = []
    for x1, y1, x2, y2, class_id, _score in detections:
        x_min = min(max((left + x1 * scale) / image_width, 0.0), 1.0)
        x_max = min(max((left + x2 * scale) / image_width, 0.0), 1.0)
        y_min = min(max((top + y1 * scale) / image_height, 0.0), 1.0)
        y_max = min(max((top + y2 * scale) / image_height, 0.0), 1.0)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        lines.append(f"{class_id - 1} {x_center:.6f} {y_center:.6f} "
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
        annotations = []
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
                detections = model.predict(images)
                if args.write_yolo:
                    for tile_detections, tile in zip(detections, item):
                        annotations.extend(yolo_lines(tile_detections, tile, dataset.scale,
                                                      dataset.img_width, dataset.img_height))
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
