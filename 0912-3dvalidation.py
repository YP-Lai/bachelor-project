import os
import json
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree
import pandas as pd
import matplotlib.pyplot as plt

def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    spacing = img.header.get_zooms()  # voxel spacing (x,y,z)
    return data, spacing

def binarize_and_filter(pred_prob, thresh=0.5, min_size=1000):
    pred_bin = (pred_prob >= thresh).astype(np.uint8)
    lbl, n = ndimage.label(pred_bin)
    sizes = ndimage.sum(pred_bin, lbl, range(1, n+1))
    keep = np.zeros(n+1, dtype=bool)
    for i, s in enumerate(sizes, start=1):
        if s >= min_size:
            keep[i] = True
    filtered = keep[lbl]
    return filtered.astype(np.uint8)

def dice_3d(y_true, y_pred, eps=1e-6):
    y_true_f = (y_true > 0).astype(np.int64)
    y_pred_f = (y_pred > 0).astype(np.int64)
    intersection = (y_true_f & y_pred_f).sum()
    return 2.0 * intersection / (y_true_f.sum() + y_pred_f.sum() + eps)

def surface_voxels(binary):
    eroded = ndimage.binary_erosion(binary)
    surf = binary & (~eroded)
    coords = np.array(np.where(surf)).T
    return coords

def hd95_from_voxels(bin1, bin2, spacing=(1.0,1.0,1.0)):
    s1 = surface_voxels(bin1)
    s2 = surface_voxels(bin2)
    if s1.size == 0 or s2.size == 0:
        return np.nan
    s1_mm = s1 * np.array(spacing[::-1])  # (z,y,x) → spacing (x,y,z) 
    s2_mm = s2 * np.array(spacing[::-1])
    tree_s2 = cKDTree(s2_mm)
    d1, _ = tree_s2.query(s1_mm, k=1)
    tree_s1 = cKDTree(s1_mm)
    d2, _ = tree_s1.query(s2_mm, k=1)
    all_d = np.concatenate([d1, d2])
    return np.percentile(all_d, 95)

# main
def evaluate_pairs(pairs_json, output_dir, min_size=500):
    os.makedirs(output_dir, exist_ok=True)

    with open(pairs_json, "r") as f:
        pairs = json.load(f)

    results = []

    for sample in pairs:
        pred_path = sample["pred"]
        gt_path = sample["label"]

        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            print(f"跳過, 檔案不存在: {pred_path} or {gt_path}")
            continue

        pred_arr, spacing_pred = load_nifti(pred_path)
        gt_arr, spacing_gt = load_nifti(gt_path)

        if pred_arr.shape != gt_arr.shape:
            print(f" Shape mismatch: {pred_path}")
            continue

        # prob → binary
        pred_bin = (pred_arr >= 0.5).astype(np.uint8)
        pred_filtered = binarize_and_filter(pred_bin, min_size=min_size)

        gt_bin = (gt_arr > 0).astype(np.uint8)

        dice = dice_3d(gt_bin, pred_filtered)
        hd95 = hd95_from_voxels(gt_bin, pred_filtered, spacing=spacing_gt)

        results.append({
            "pred": pred_path,
            "gt": gt_path,
            "dice": dice,
            "hd95": hd95
        })

        print(f" {os.path.basename(pred_path)} | Dice={dice:.4f}, HD95={hd95:.2f} mm")

 
    df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, "qc_results.xlsx")
    df.to_excel(out_path, index=False)
    print(f"\n output: {out_path}")
    # 計算平均值
    mean_dice = df["dice"].mean()
    mean_hd95 = df["hd95"].mean()

if __name__ == "__main__":
    pairs_json = r"F:\yplai\segmentation_pairs.json"
    output_dir = r"F:\yplai\qc_results"
    evaluate_pairs(pairs_json, output_dir)

  # read Excel 
out_path = os.path.join(output_dir, "qc_results.xlsx")
df = pd.read_excel(out_path)

plt.figure(figsize=(12,5))
plt.show()


