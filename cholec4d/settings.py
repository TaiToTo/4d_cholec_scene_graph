"""Every setting of the run the paper reports.

The stages and the evaluation read their settings from here, so this module
is the record of what was run. Values that differ from a model's own
defaults say so.
"""

# --- Windows ---------------------------------------------------------------
# One window per CholecSeg8k clip: SAMPLES frames, STRIDE CholecSeg8k numbers
# apart, the clip's middle annotated frame at index SEED_INDEX.
SAMPLES = 30
STRIDE = 15
SEED_INDEX = 15
MIN_ANNOTATED = 5
# CholecSeg8k numbers these videos at their own 25 frames a second and the
# other twelve at about 30, so one number is one video frame here and 25/30
# of one elsewhere.
VIDEOS_AT_25FPS = (1, 9, 12, 20, 35)

# --- Frames ----------------------------------------------------------------
# An annotated image is located in the Cholec80 video by the smallest mean
# absolute difference; the search reaches back number // 5 + 150 frames and
# ahead 60, and a match must differ by at most MATCH_MAX_MAE.
SEARCH_BACK_DIVISOR = 5
SEARCH_BACK = 150
SEARCH_AHEAD = 60
MATCH_MAX_MAE = 10.0
# Native frames per CholecSeg8k number the annotated frames may run at.
NUMBER_RATES = (1.0, 25 / 30)
RATE_TOLERANCE = 0.02
# The crop is the largest rectangle inside the pixels whose luminance, at the
# CROP_PERCENTILE-th percentile over the window's frames, reaches CROP_LUMINANCE.
CROP_LUMINANCE = 15.0
CROP_PERCENTILE = 90.0

# --- Depth -----------------------------------------------------------------
DEPTH_MODEL = "depth-anything/DA3-LARGE"
DEPTH_PROCESS_RES = 504
# DA3 places the window's coordinates on one reference view. Its default
# picks the view by an argmin computed in bfloat16, which another GPU can
# resolve differently; the run fixes it to the middle view, the seed frame.
DEPTH_REFERENCE_VIEW = "middle"

# --- Segmentation (SAM automatic mask generator, ViT-H) ---------------------
SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT_SHA256 = "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"
SAM_INPUTS = ("rgb", "depth", "normal")
POINTS_PER_SIDE = (4, 6, 8, 12, 16, 24, 32, 48)
DEFAULT_POINTS_PER_SIDE = 24
# segment_anything's defaults are 0.88, 0.95 and 0.
SAM_PRED_IOU_THRESH = 0.8
SAM_STABILITY_THRESH = 0.8
SAM_MIN_MASK_REGION_AREA = 100
# Looser quality cuts SAM was run at again, on the ends of the grid, to see
# whether the growth of its mask count depends on the cut.
EXTRA_QUALITY = (0.6, 0.7)
EXTRA_QUALITY_POINTS_PER_SIDE = (4, 48)
# Stricter cuts applied afterwards to the masks recorded at 0.8.
STRICTER_QUALITY = (0.8, 0.85, 0.9, 0.95)

# --- Tracking (SAM 3's mask-prompted video tracker) ------------------------
TRACK_MODEL = "facebook/sam3"
# Seed regions under this many pixels are not tracked.
SEED_MIN_AREA = 400

# --- Evaluation ------------------------------------------------------------
IOU_THRESHOLDS = (0.5, 0.75)
MATCH_IOU = 0.5
# A region is kept on a frame when at least this share of it lies on its
# reference class.
KEPT_FRACTION = 0.5
# A region mixes annotated instances when its largest instance holds less
# than this share of it.
PURE_FRACTION = 0.9
# The share a seed region must hold on its reference class to count as pure.
SEED_PURITY = (0.8, 0.9)
SECOND_CLASS_FRACTION = 0.25
# A structure counts as covered when at least this share of it lies in some region.
COVERED_FRACTION = 0.5

# --- Statistics ------------------------------------------------------------
N_BOOT = 10000
BOOT_SEED = 0
PAIRS = (("normal", "rgb"), ("normal", "depth"), ("rgb", "depth"))

# --- CholecSeg8k classes ---------------------------------------------------
BACKGROUND = 0
N_CLASSES = 13
CLASS_NAMES = {
    0: "background", 1: "abdominal wall", 2: "liver", 3: "gastrointestinal tract",
    4: "fat", 5: "grasper", 6: "connective tissue", 7: "blood", 8: "cystic duct",
    9: "L-hook electrocautery", 10: "gallbladder", 11: "hepatic vein", 12: "liver ligament",
}
INSTRUMENTS = (5, 9)
TISSUES = (1, 2, 3, 4, 6, 7, 8, 10, 11, 12)
