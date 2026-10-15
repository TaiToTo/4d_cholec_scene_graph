"""Instance F1 penalizes a split twice; oracle mIoU does not see it."""
import numpy as np

from cholec4d.metrics import frame_scores, gt_instances, merge_per_instance


def scene():
    gt = np.full((10, 10), 2, np.int32)      # liver on the left
    gt[:, 5:] = 10                            # gallbladder on the right
    labels = np.where(gt == 2, 0, 1).astype(np.int32)
    return gt, labels, np.ones_like(gt, bool)


def test_one_region_per_structure_scores_one():
    gt, labels, valid = scene()
    gts = [m for m, _ in gt_instances(gt, valid)]
    s = frame_scores(labels, gt, valid, gts)
    assert s["f1@0.5"] == 1.0 and s["miou"] == 1.0 and s["n_pred"] == 2 and s["n_gt"] == 2


def test_split_lowers_f1_but_not_oracle_miou():
    gt, labels, valid = scene()
    labels[:4, :5] = 2                        # 20 of the liver's 50 pixels become a third region
    gts = [m for m, _ in gt_instances(gt, valid)]
    s = frame_scores(labels, gt, valid, gts)
    assert s["f1@0.5"] == 2 * 2 / (2 * 2 + 1 + 0)   # the 30-pixel part still matches (IoU 0.6)
    assert s["f1@0.75"] == 2 * 1 / (2 * 1 + 2 + 1)
    assert s["miou"] == 1.0
    merged = merge_per_instance(labels, valid, gts)
    assert frame_scores(merged, gt, valid, gts)["f1@0.75"] == 1.0


def test_instances_are_connected_components():
    gt, _, valid = scene()
    gt[:, 2] = 9                              # an instrument cuts the liver in two
    classes = sorted(c for _, c in gt_instances(gt, valid))
    assert classes == [2, 2, 9, 10]
