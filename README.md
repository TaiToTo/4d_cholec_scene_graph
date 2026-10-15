# Training-free 3D time-series surgical scene graphs on CholecSeg8k

Code for *Toward Training-Free Construction of 3D Time-Series Surgical Scene
Graphs via Surface-Normal-Guided Segmentation* (Healthcare Technology
Letters, 2026). The pipeline predicts depth, intrinsics and camera poses
for a laparoscopic clip with Depth Anything 3, segments one frame's surface
normal map (or its RGB or depth image) into class-agnostic regions with
SAM, tracks the regions through the clip with SAM 3, and assembles them
into a scene graph. The evaluation scores the regions against
CholecSeg8k's annotation, and every number the paper prints can be
recomputed here.

## Check the paper's numbers

No dataset or GPU is needed; it takes about ten seconds.

```bash
pip install -e .
python -m cholec4d.numbers
```

The command recomputes the 63 numbers of the paper's tables and text from
the per-window values in `results/per_window/`, formats each as the paper
prints it, and compares it with `results/manuscript.csv`. 62 are equal.
The other is noted there: Table 2 prints 0.590 for the static control of
depth, whose value 0.5905 rounds to 0.591. The computed numbers, with their
intervals, are written to `results/numbers.csv`.

`results/per_window/` holds one value per window and condition. It was
written by `python -m cholec4d.measure` from the paper's tracks, which are
not distributed (see "Data").

## Run the pipeline

A Linux machine with an NVIDIA GPU, Python 3.12 (Depth Anything 3 needs
`numpy<2`), and about 30 GB of disk for the intermediate outputs.

```bash
pip install -e ".[pipeline]"
```

**Data.** Download CholecSeg8k
(<https://www.kaggle.com/datasets/newslab/cholecseg8k>) and Cholec80
(<https://camma.unistra.fr/datasets/>). Neither may be redistributed, so
this repository holds no frame or mask; `windows/` lists frame numbers only.

**Models.** The SAM ViT-H checkpoint `sam_vit_h_4b8939.pth` is linked from
the [segment-anything](https://github.com/facebookresearch/segment-anything)
repository. DA3-LARGE (`depth-anything/DA3-LARGE`) and SAM 3
(`facebook/sam3`) download from Hugging Face on first use; SAM 3 is gated,
so accept its license on its model page and log in first. Each model keeps
its own license (DA3-LARGE is for non-commercial use only).

```bash
export CHOLECSEG8K_ROOT=/path/to/CholecSeg8k        # holds video01/, video09/, ...
export CHOLEC80_VIDEOS=/path/to/cholec80/videos     # holds video01.mp4, ...
export SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth
export CHOLEC4D_WORK=/path/to/work                  # intermediate outputs

python -m cholec4d.extract --workers 8                       # frames and masks, cropped
python -m cholec4d.depth --gpu 0                             # depth, intrinsics, poses
python -m cholec4d.segment --gpu 0                           # SAM seed maps, 8 grid densities
python -m cholec4d.segment --gpu 0 --quality 0.7 --pps 4 48  # SAM at looser quality cuts,
python -m cholec4d.segment --gpu 0 --quality 0.6 --pps 4 48  # for the mask counts only
python -m cholec4d.track --gpu 0                             # SAM 3 tracking
python -m cholec4d.measure --workers 16 --out results/per_window_rerun
python -m cholec4d.numbers --tables results/per_window_rerun --out results/numbers_rerun.csv
python -m cholec4d.graph --windows W000                      # a window's scene graph, as JSON
```

Each stage skips what is already written, so a stopped run resumes where
it stopped. The GPU stages take `--shard k/n` to give every n-th window,
starting at the k-th, to one GPU, so n copies with k = 0 … n-1 split the
work. On one GPU, depth takes a few minutes, segmentation about an hour and
tracking (2,064 tracks of 30 frames) several hours.

`python -m cholec4d.windows` writes `windows/` again from the datasets; the
files come out identical.

## What the pipeline does

| stage | module | writes, under the work directory |
|---|---|---|
| windows | `cholec4d.windows` | `windows/windows.csv` (in this repository): one window per CholecSeg8k clip, 86 kept, 15 excluded with the reason |
| frames | `cholec4d.extract` | `windows/<clip>/frames/`, `masks/`, `window.json`: 30 frames 15 CholecSeg8k numbers apart, annotated frames from CholecSeg8k, the others from the video, all cropped to one rectangle |
| depth | `cholec4d.depth` | `windows/<clip>/depth.npz`: DA3-LARGE depth, confidence, intrinsics and world-to-camera poses at 504 px, the world frame on the seed frame's camera |
| segmentation | `cholec4d.segment` | `seeds/q<cut>/pps<N>/<clip>/<input>.npz`: SAM's region map of the seed frame, and each mask's area and scores |
| tracking | `cholec4d.track` | `tracks/pps<N>/<clip>/<input>.npy`: the regions tracked forward and backward from the seed frame |
| evaluation | `cholec4d.measure` | `results/per_window/*.csv` |
| statistics | `cholec4d.numbers` | `results/numbers.csv` |
| scene graph | `cholec4d.graph` | `graphs/pps<N>/<clip>/<input>.json`: per-frame nodes (3D centroid, second-moment ellipsoid, area), relations and events |

Every setting is in `cholec4d/settings.py`. Where the run departs from a
model's defaults, that file says so: SAM keeps masks whose predicted IoU
and stability are at least 0.8 (defaults 0.88 and 0.95) and removes
regions under 100 px (default 0); seed regions under 400 px are not
tracked; DA3's reference view is fixed to the seed frame.

The evaluation (`metrics.py`, `temporal.py`, `relations.py`) scores the
annotated frames of each window within their valid-depth pixels, with no
area cut. Ground-truth instances are the connected components of each
non-background class. Confidence intervals are 95 % percentile bootstrap
intervals over the 17 videos, 10,000 resamples (`stats.py`).

## Reproducibility

This repository is a reorganized port of the code that produced the
paper's numbers, checked against it:

- The windows, frames, masks and crops come out byte-identical to the
  paper's run.
- From the paper's tracks, `cholec4d.measure` gives every per-window value
  of `results/per_window/` bit for bit, and `cholec4d.numbers` every mean
  and interval: the bootstrap seeds itself from the values.
- Run side by side on one GPU, this code and the paper's give identical
  depth, SAM masks and tracks.
- Across GPU models they do not. The paper's run used an NVIDIA RTX 6000
  Ada; on an H100, DA3's depth of a window differs from it by 0.06 % at
  the median pixel (1.2 % at most) and its focal length by up to 0.4 %.
  That is enough to change some of SAM's regions on the depth image and the
  normal map, so a re-run on another GPU gives per-window values, and
  numbers, that differ somewhat from the paper's. RGB-seeded regions are
  nearly unaffected.
- Versions: Python 3.12.3, torch 2.10.0 (CUDA 12.8), transformers 5.8.1,
  numpy 1.26.4, pandas 3.0.1, OpenCV 4.11.0, Pillow 11.3.0; Depth Anything 3
  and segment-anything at the commits pinned in `pyproject.toml`.

## Tests

```bash
pip install -e ".[test]"
pytest
```

## License and citation

The code and the tables are released under the Apache License 2.0
(`LICENSE`). CholecSeg8k, Cholec80 and the models keep their own licenses.
To cite this work, see `CITATION.cff`.
