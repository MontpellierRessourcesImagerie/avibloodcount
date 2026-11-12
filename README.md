# AviBloodCount
This software enables fully automated counting of blood cells, including erythrocytes as well as leukocytes, in avian blood samples. AviBloodCount takes whole slide images as input and outputs the overall cell counts of all areas determined as countable.

We provide our models and annotated data at [data.uni-marburg.de](https://data.uni-marburg.de/handle/dataumr/250).

## Installation
We provide a docker file to create an environment containing all dependencies necessary to run our model on whole slide images stored in the SVS or TIFF file format.

### Install Docker
To install Docker, follow the instructions at [docker.com](https://www.docker.com/get-started/)

### Build Docker image
To build the docker image go to the `docker` directory and run the following command:

` docker build --build-arg="USER_ID=$(id -u)" -t umr-ds/avibloodcount .`

## Usage
### Preparation
1. Download our trained models, i.e., `efficientNet_B0.onnx` and `condInst_R101.pth` from [here](https://data.uni-marburg.de/handle/dataumr/250) to `./models`. 
2. Place the SVS or TIFF files to be processed in `./input`. We provide several example [SVS files](https://dshare.mathematik.uni-marburg.de/index.php/s/if5Y2KmSAX5rMLG).

### Run inference
Running the Docker image as follows automatically starts inference with default parameters on all SVS or TIFF files located at `./data`:

`docker run --rm -it -v /path/to/code:/code -v /path/to/data:/data --shm-size 8gb --name avibloodcount umr-ds/avibloodcount`

To run with GPU support, do the following:

`docker run --rm -it -v /path/to/code:/code -v /path/to/data:/data --shm-size 8gb --name avibloodcount --gpus 1 umr-ds/avibloodcount --gpu 0`

In this example, the code will run on GPU with index 0.

### Processing TIFF images
Both SVS and TIFF (`.tif`/`.tiff`) files are supported. To process images that do not live in `./input`, pass their directory via `--path`:

`docker run --rm -it -v /path/to/code:/code -v /path/to/data:/data --shm-size 8gb --name avibloodcount umr-ds/avibloodcount --path /data/images`

Uncompressed TIFFs are read region by region straight from disk, so multi-gigabyte whole slide images are processed without loading them into memory.

Images are analyzed at the physical resolution the models were trained on (0.2525 um per pixel, i.e. 40x scans). Images stored at a different resolution are resampled automatically based on the physical pixel size stored in the file metadata (OME-TIFF `PhysicalSizeX` or the TIFF resolution tags), so that each tile always covers the same physical area as a training tile. The resampling factor is printed per file. Images without resolution metadata are analyzed at their native resolution; pass `--target-mpp <value>` to analyze at another resolution or `--target-mpp 0` to disable resampling entirely.

### Visual QC with napari-simpleannotate
In addition to the JSON counts, inference writes YOLO bounding box annotations next to the input images so the detections can be reviewed with [napari-simpleannotate](https://github.com/MontpellierRessourcesImagerie/napari-simpleannotate) (`Open Directory` on the image folder):

- `<image_name>.txt` — one line per detected cell: `class_id x_center y_center width height`, normalized to 0-1 relative to the full image (the resampling is undone, so boxes align with the image as napari opens it). Written only for images that contain detections; files of previous runs without detections are removed.
- `class.yaml` — the class list (`names:` mapping).
- `colors.yaml` — the class to color mapping used by the plugin.

| class_id | class | color |
| --- | --- | --- |
| 0 | Erythrocyte (with nucleus) | red |
| 1 | Eosinophil | green |
| 2 | Lymphocyte | blue |
| 3 | Heterophil | cyan |
| 4 | Erythrocyte (cell) with parasite | magenta |
| 5 | Monocyte | yellow |
| 6 | Basophil | black |

Boxes cover the nuclei of the detected cells, as trained. Annotations are written for all detections above the detection threshold, including the parasite class, which is not part of the JSON counts. Note that existing `.txt`, `class.yaml` and `colors.yaml` files in the image folder are overwritten on every run, and that napari loads the complete image into memory. Pass `--no-yolo` to skip the annotation files.

You can modify several parameters by passing them trailing the `docker run command`. For example, you can set the thresholds used for the countability classification and the instance segmentation models via `--cls_thresh` and `--det_thresh`, respectively. All available parameters can be listed by issuing 

`docker run --rm -it -v /path/to/code:/code -v /path/to/data:/data --shm-size 8gb --name avibloodcount umr-ds/avibloodcount --help`


## Scientific Usage & Citation

If you are using this software in academia, we'd appreciate if you cited our [scientific research paper](https://doi.org/10.3390/birds5010004) as follows:

> Vogelbacher, M.; Strehmann, F.; Bellafkir, H.; Mühling, M.; Korfhage, N.; Schneider, D.; Rösner, S.; Schabo, D.G.; Farwig, N.; Freisleben, B. Identifying and Counting Avian Blood Cells in Whole Slide Images via Deep Learning. Birds 2024, 5, 48-66. https://doi.org/10.3390/birds5010004

```bibtex
@article{vogelbacher2024identifying,
  title = {{Identifying and Counting Avian Blood Cells in Whole Slide Images}},
  author = {Vogelbacher, Markus and Strehmann, Finja and Bellafkir, Hicham and M{\"u}hling, Markus and Korfhage, Nikolaus and Schneider, Daniel and R{\"o}sner, Sascha and Schabo, Dana G. and Farwig, Nina and Freisleben, Bernd},
  journal = {Birds},
  volume = {5},
  year = {2024},
  number = {1},
  pages = {48--66},
  url = {https://www.mdpi.com/2673-6004/5/1/4},
  issn = {2673-6004},
  keywords = {cell segmentation, bird blood analysis, microscopy images, blood smear images, object detection, ornithology},
  doi = {10.3390/birds5010004},
}
```
