#!/usr/bin/env python3
"""Rodent T1 mapping workflow using a T2-weighted image and a series of MDEFT images.

This script performs the following:
    1. Loads a T2w image and a set of MDEFT inversion-recovery images.
    2. Estimates a voxel-wise T1 map from the multi-TI MDEFT data.
    3. Saves the T1 map image.
    4. Summarizes T1 values within a parcellation atlas and saves them to CSV.
    5. Produces an HTML report with T1 map diagnostics.
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
# import ants
from scipy import ndimage
from scipy.optimize import curve_fit


def parse_args():
    parser = argparse.ArgumentParser(description="Rodent T1 mapping from T2w + MDEFT data")
    parser.add_argument("--t2w", help="Path to the T2-weighted anatomical image (NIfTI)")
    parser.add_argument("--mdeft", nargs="*", help="One or more MDEFT images or a single 4D MDEFT file")
    parser.add_argument("--ti", nargs="*", type=float, help="Inversion times (ms) corresponding to --mdeft images")
    parser.add_argument("--erode", type=int, default=0, help="Optional number of erosion iterations for the T2w mask")
    parser.add_argument("--parcellation", help="Optional parcellation image (integer labels) used for region statistics")
    parser.add_argument("--output-dir", default="./t1_mapping_output", help="Directory for outputs")
    parser.add_argument("--mask-threshold", type=float, default=None, help="Optional threshold for T2w mask generation")
    parser.add_argument("--report-name", default="t1_mapping_report.html", help="HTML report filename")
    parser.add_argument("--reports-only", action="store_true", help="Skip the T1 fit and generate plots/report from an existing t1_map.nii.gz in the output directory")
    args = parser.parse_args()
    if not args.reports_only and (args.t2w is None or not args.mdeft):
        parser.error("--t2w and --mdeft are required unless --reports-only is set")
    return args


def load_nifti(path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header


def save_nifti(data, affine, hdr, path):
    out = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine, header=hdr)
    nib.save(out, str(path))


def as_3d(data):
    if data.ndim == 4 and data.shape[-1] == 1:
        return data[:, :, :, 0]
    if data.ndim == 3:
        return data
    raise ValueError(f"Expected 3D image but received shape {data.shape}")


def erode_mask(mask, iterations=1):
    if iterations > 0:
        mask = ndimage.binary_erosion(mask, iterations=iterations).astype(mask.dtype)
    return mask


def get_inversion_times_from_JSON(mdeft_images):
    """
    Extract inversion times (TI) from the JSON sidecar associated with each NIfTI image in the list.
    Returns a list of inversion times in milliseconds for each MDEFT image.
    """
    import json
    ti_list = []

    def _json_sidecar_path(nifti_path):
        p = Path(nifti_path)
        if str(p).endswith(".nii.gz"):
            return str(p)[:-7] + ".json"
        if p.suffix == ".nii":
            return str(p.with_suffix(".json"))
        return str(p.with_suffix(".json"))

    for mdeft_image in mdeft_images:
        json_path = _json_sidecar_path(mdeft_image)
        if "masked" in str(Path(mdeft_image)) and str(Path(mdeft_image)).endswith("_masked.nii.gz"):
            json_path = str(Path(mdeft_image)).replace("_masked.nii.gz", ".json")
        if not os.path.exists(json_path):
            raise ValueError(f"JSON sidecar path not found for the given MDEFT image: {mdeft_image}")
        with open(json_path, "r") as f:
            metadata = json.load(f)
        if "InversionTime" not in metadata:
            raise ValueError(f"Inversion times (TI) not found in JSON sidecar for image: {mdeft_image}")
        ti_values = metadata["InversionTime"]
        if isinstance(ti_values, (list, tuple, np.ndarray)):
            vals = [float(t) for t in ti_values]
            if len(mdeft_images) == 1:
                ti_list.extend(vals)
            elif len(vals) == 1:
                ti_list.append(vals[0])
            else:
                raise ValueError(
                    f"JSON sidecar for {mdeft_image} contains multiple inversion times, "
                    "but multiple --mdeft inputs were provided."
                )
        else:
            ti_list.append(float(ti_values))
    return ti_list


def prepare_mdeft_series(mdeft_inputs, ti_values):
    if len(mdeft_inputs) == 1:
        data, affine, header = load_nifti(mdeft_inputs[0])
        if data.ndim == 4:
            volumes = np.moveaxis(data, -1, 0)
            if ti_values is None:
                raise ValueError("--ti must be supplied when using a 4D MDEFT file")
            if len(ti_values) != volumes.shape[0]:
                raise ValueError(f"MDEFT 4D volume has {volumes.shape[0]} time points but {len(ti_values)} TI values were provided")
            return volumes, affine, header, ti_values
        if ti_values is None or len(ti_values) != 1:
            raise ValueError("A single 3D MDEFT image requires exactly one TI value")
        return np.asarray([data], dtype=np.float32), affine, header, [float(ti_values[0])]

    if ti_values is None or len(ti_values) != len(mdeft_inputs):
        raise ValueError("--ti must be supplied and match the number of MDEFT images")

    vols = []
    for p in mdeft_inputs:
        data, affine, header = load_nifti(p)
        vols.append(as_3d(data))
    volume_stack = np.asarray(vols, dtype=np.float32)
    return volume_stack, affine, header, [float(v) for v in ti_values]


def compute_brain_mask(t2w_data, threshold=None):
    t2w = np.asarray(t2w_data, dtype=np.float32)
    if threshold is None:
        vmin = np.nanpercentile(t2w[~np.isclose(t2w, 0)], 1)
        vmax = np.nanpercentile(t2w[~np.isclose(t2w, 0)], 99)
        threshold = max(vmin + 0.1 * (vmax - vmin), np.nanmedian(t2w) + 1.0 * np.nanstd(t2w))
    mask = np.asarray(t2w > threshold, dtype=np.uint8)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=2)
    labeled, nlabels = ndimage.label(mask)
    if nlabels > 0:
        sizes = np.bincount(labeled.ravel())[1:]
        if sizes.size > 0:
            largest = np.argmax(sizes) + 1
            mask = labeled == largest
    return mask.astype(bool)


def rotate_clockwise_2d(arr):
    # Rotate brain displays by 90 degrees counterclockwise to preserve expected left/right orientation.
    return np.rot90(np.asarray(arr), k=1)


def make_slice_figure(fig_path, fixed, moving, title):
    fixed_arr = np.asarray(fixed)
    moving_arr = np.asarray(moving)
    i = fixed_arr.shape[0] // 2
    j = fixed_arr.shape[1] // 2
    k = fixed_arr.shape[2] // 2
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    for ax in axs:
        ax.axis("off")

    axial_fixed = rotate_clockwise_2d(fixed_arr[i, :, :])
    axial_moving = rotate_clockwise_2d(moving_arr[i, :, :])
    axs[0].imshow(axial_fixed, cmap="gray", origin="lower")
    axs[0].imshow(axial_moving, cmap="gist_heat", alpha=0.5, origin="lower")
    axs[0].set_title(f"Axial {title}")

    sagittal_fixed = rotate_clockwise_2d(fixed_arr[:, j, :])
    sagittal_moving = rotate_clockwise_2d(moving_arr[:, j, :])
    axs[1].imshow(sagittal_fixed, cmap="gray", origin="lower")
    axs[1].imshow(sagittal_moving, cmap="gist_heat", alpha=0.5, origin="lower")
    axs[1].set_title(f"Sagittal {title}")

    coronal_fixed = rotate_clockwise_2d(fixed_arr[:, :, k])
    coronal_moving = rotate_clockwise_2d(moving_arr[:, :, k])
    axs[2].imshow(coronal_fixed, cmap="gray", origin="lower")
    axs[2].imshow(coronal_moving, cmap="gist_heat", alpha=0.5, origin="lower")
    axs[2].set_title(f"Coronal {title}")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def make_mask_figure(fig_path, t2w, mask):
    i = t2w.shape[0] // 2
    j = t2w.shape[1] // 2
    k = t2w.shape[2] // 2
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    for ax in axs:
        ax.axis("off")

    axs[0].imshow(rotate_clockwise_2d(t2w[i, :, :]), cmap="gray", origin="lower")
    axs[0].imshow(rotate_clockwise_2d(mask[i, :, :]), cmap="Blues", alpha=0.35, origin="lower")
    axs[0].set_title("Axial T2w brain mask")

    axs[1].imshow(rotate_clockwise_2d(t2w[:, j, :]), cmap="gray", origin="lower")
    axs[1].imshow(rotate_clockwise_2d(mask[:, j, :]), cmap="Blues", alpha=0.35, origin="lower")
    axs[1].set_title("Sagittal T2w brain mask")

    axs[2].imshow(rotate_clockwise_2d(t2w[:, :, k]), cmap="gray", origin="lower")
    axs[2].imshow(rotate_clockwise_2d(mask[:, :, k]), cmap="Blues", alpha=0.35, origin="lower")
    axs[2].set_title("Coronal T2w brain mask")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def make_atlas_overlay_figure(fig_path, background, atlas, title):
    i = background.shape[0] // 2
    j = background.shape[1] // 2
    k = background.shape[2] // 2
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    for ax in axs:
        ax.axis("off")

    for ax, label, bg_slice, atlas_slice in [
        (axs[0], "Axial", rotate_clockwise_2d(background[i, :, :]), rotate_clockwise_2d(atlas[i, :, :])),
        (axs[1], "Sagittal", rotate_clockwise_2d(background[:, j, :]), rotate_clockwise_2d(atlas[:, j, :])),
        (axs[2], "Coronal", rotate_clockwise_2d(background[:, :, k]), rotate_clockwise_2d(atlas[:, :, k])),
    ]:
        atlas_masked = np.ma.masked_where(atlas_slice <= 0, atlas_slice)
        ax.imshow(bg_slice, cmap="gray", origin="lower")
        ax.imshow(atlas_masked, cmap="nipy_spectral", alpha=0.45, origin="lower", interpolation="nearest")
        if np.any(atlas_slice > 0):
            ax.contour((atlas_slice > 0).astype(np.uint8), levels=[0.5], colors="white", linewidths=0.4)
        ax.set_title(f"{label} {title}")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def inversion_recovery_model(ti, A, B, T1):
    """Estimate T1 using a simple inversion-recovery model: y = |A. (1 - 2*(B.exp(-TI / T1)))|."""
    return np.abs(A * (1 - 2 * (B * np.exp(-ti / T1))))


def fit_t1_map(mdeft_series, ti_values, brain_mask):
    """
    Estimate T1 map from a series of MDEFT images and corresponding inversion times.
    Parameters:
        mdeft_series (numpy.ndarray): 4D array of MDEFT images with shape (n_ti, x, y, z)
        ti_values (array-like): Inversion times corresponding to the first dimension of mdeft_series
        brain_mask (numpy.ndarray): 3D boolean array indicating brain voxels
    Returns:
        t1_map (numpy.ndarray): 3D array of estimated T1 values within the brain mask
    """
    ti = np.asarray(ti_values, dtype=np.float32)
    if mdeft_series.shape[0] != ti.size:
        raise ValueError(f"MDEFT series length {mdeft_series.shape[0]} does not match the TI count {ti.size}")

    t1_map = np.full(brain_mask.shape, np.nan, dtype=np.float32)
    voxel_coords = np.argwhere(brain_mask)
    if voxel_coords.size == 0:
        return t1_map

    for x, y, z in voxel_coords:
        signal = np.asarray(mdeft_series[:, x, y, z], dtype=np.float32)
        valid = np.isfinite(signal)
        if np.count_nonzero(valid) < 3:
            continue

        ti_fit = ti[valid]
        signal_fit = signal[valid]
        if np.allclose(signal_fit, signal_fit[0]):
            continue

        amp_guess = float(np.nanmax(np.abs(signal_fit)))
        if amp_guess <= 0:
            continue

        try:
            popt, _ = curve_fit(
                inversion_recovery_model,
                ti_fit,
                signal_fit,
                p0=[amp_guess, 1.0, 500.0],
                bounds=([0.0, 0.0, 10.0], [1e9, 1e9, 5000.0]),
                maxfev=10000,
            )
            t1_value = float(np.clip(popt[2], 10.0, 5000.0))
        except (RuntimeError, ValueError):
            try:
                def mono_model(t, t1):
                    return np.abs(amp_guess * (1.0 - 2.0 * np.exp(-t / t1)))

                popt, _ = curve_fit(mono_model, ti_fit, signal_fit, p0=[500.0], bounds=([10.0], [5000.0]), maxfev=10000)
                t1_value = float(np.clip(popt[0], 10.0, 5000.0))
            except (RuntimeError, ValueError):
                continue

        t1_map[x, y, z] = t1_value

    return t1_map


def annotate_region_labels(stats_df, label_map):
    if stats_df is None or stats_df.empty:
        return stats_df
    df = stats_df.copy()
    if "region" not in df.columns:
        return df
    df["label"] = df["region"].map(lambda region: label_map.get(int(region), ""))
    columns = ["region", "label"] + [col for col in df.columns if col not in {"region", "label"}]
    return df[columns]


def save_results_csv(stats_df, csv_path, lut_path=None):
    stats_df = stats_df.copy()
    if stats_df.empty:
        stats_df = pd.DataFrame(columns=["region", "label", "count", "mean_t1", "median_t1", "std_t1", "min_t1", "max_t1"])
    if "region" not in stats_df.columns:
        if "label" in stats_df.columns:
            stats_df = stats_df.rename(columns={"label": "region"})
        else:
            raise ValueError("stats_df must include a 'region' column")

    label_map = load_sigma_lut(lut_path)
    if label_map and "label" not in stats_df.columns:
        stats_df = annotate_region_labels(stats_df, label_map)
    elif label_map and "label" in stats_df.columns:
        stats_df["label"] = stats_df["region"].map(lambda region: label_map.get(int(region), ""))
    elif "label" not in stats_df.columns:
        stats_df["label"] = ""

    columns = ["region", "label"] + [col for col in stats_df.columns if col not in {"region", "label"}]
    stats_df = stats_df[columns]
    stats_df.to_csv(csv_path, index=False)


def summarize_t1_regions(t1_map, parcellation_data):
    if parcellation_data is None:
        return pd.DataFrame(columns=["region", "count", "mean_t1", "median_t1", "std_t1", "min_t1", "max_t1"])
    if parcellation_data.shape != t1_map.shape:
        raise ValueError(f"Parcellation shape {parcellation_data.shape} does not match T1 map shape {t1_map.shape}")

    valid = np.isfinite(t1_map)
    labels = parcellation_data[valid]
    values = t1_map[valid]
    rows = []
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        if int(lab) == 0:
            continue
        idx = labels == lab
        region_vals = values[idx]
        if region_vals.size == 0:
            continue
        rows.append({
            "region": int(lab),
            "count": int(region_vals.size),
            "mean_t1": float(np.mean(region_vals)),
            "median_t1": float(np.median(region_vals)),
            "std_t1": float(np.std(region_vals)),
            "min_t1": float(np.min(region_vals)),
            "max_t1": float(np.max(region_vals)),
        })
    return pd.DataFrame(rows)


def load_sigma_lut(lut_path):
    label_map = {}
    if lut_path is None or not Path(lut_path).exists():
        return label_map

    # Read the LUT file (splitting by tab and not spaces)
    with open(lut_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            try:
                lab = int(row[0])
            except (ValueError, IndexError):
                continue
            if len(row) < 2:
                continue
            label_map[lab] = row[-1].replace("_", " ")

    return label_map



def find_sigma_lut(parcellation_path=None):
    candidates = [
        Path.cwd() / "SIGMA_LUT.txt",
        Path(__file__).resolve().parent / "SIGMA_LUT.txt",
    ]
    if parcellation_path:
        candidates.append(Path(parcellation_path).resolve().parent / "SIGMA_LUT.txt")
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def make_spider_plot(fig_path, stats_df, label_map):
    fig, ax = plt.subplots(figsize=(30, 24), subplot_kw={"polar": True})

    if stats_df is None or stats_df.empty:
        ax.text(0.5, 0.5, "No regional T1 data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        return

    plot_df = stats_df.sort_values("region")
    regions = [int(r) for r in plot_df["region"].tolist()]
    values = [float(v) for v in plot_df["mean_t1"].tolist()]
    labels = [label_map.get(r, f"Label {r}") for r in regions]

    n = len(values)
    if n == 0:
        ax.text(0.5, 0.5, "No regional T1 data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        return

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    values += values[:1]

    ax.plot(angles, values, color="tab:blue", linewidth=2)
    ax.fill(angles, values, color="tab:blue", alpha=0.25)

    tick_angles = [np.degrees(a) for a in angles[:-1]]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=14)
    ax.tick_params(axis="x", pad=28)

    # Orient each region label radially so text points away from the circle.
    # This keeps labels at the top pointing up and labels at the bottom
    # pointing down for better legibility.
    for lbl, ang in zip(ax.get_xticklabels(), tick_angles):
        ang = ang % 360
        lbl.set_rotation(ang)
        lbl.set_rotation_mode("anchor")

        # Keep labels outside the circle; center at near-vertical positions
        # to avoid clipping/overlap at the top and bottom.
        if np.isclose(ang, 90, atol=1.0) or np.isclose(ang, 270, atol=1.0):
            lbl.set_horizontalalignment("center")
        elif 90 < ang < 270:
            lbl.set_horizontalalignment("right")
        else:
            lbl.set_horizontalalignment("left")
    ax.set_title("Regional mean T1 by atlas label", pad=20)
    ax.grid(alpha=0.4)

    # fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def write_html_report(output_dir, report_name, inputs, t1_fig, histogram_fig, t1_map_path, mdeft_paths, parcellation_path, atlas_figs=None, spider_fig=None):
    report_path = output_dir / report_name
    atlas_figs = atlas_figs or []
    atlas_section = ""
    if atlas_figs:
        atlas_imgs = "\n".join([f'          <img src="{Path(p).name}" alt="Atlas overlay" />' for p in atlas_figs])
        atlas_section = f"""
            <div class="section">
                <h2>Atlas overlays</h2>
                <div class="images">
{atlas_imgs}
                </div>
            </div>
"""

    spider_img = f'          <img src="{spider_fig.name}" alt="Regional T1 spider plot" />' if spider_fig else ""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Rodent T1 Mapping Report</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; }}
        .section {{ margin-top: 24px; }}
        .images {{ display: flex; flex-wrap: wrap; gap: 12px; }}
        .images img {{ max-width: 45%; border: 1px solid #ddd; padding: 8px; background: #fff; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
      </style>
    </head>
    <body>
      <h1>Rodent T1 Mapping Report</h1>

      <div class="section">
        <h2>Inputs</h2>
        <ul>
          <li>T2w: {inputs['t2w']}</li>
          <li>MDEFT images: {len(mdeft_paths)} files</li>
          <li>Parcellation: {parcellation_path if parcellation_path else 'Not supplied'}</li>
          <li>Output directory: {output_dir}</li>
        </ul>
      </div>

      <div class="section">
        <h2>T1 mapping results</h2>
        <div class="images">
          <img src="{t1_fig.name}" alt="T1 map slice" />
          <img src="{histogram_fig.name}" alt="T1 histogram" />
{spider_img}
        </div>
      </div>

{atlas_section}

      <div class="section">
        <h2>Output images</h2>
        <ul>
          <li>T1 map: <a href="{t1_map_path.name}">{t1_map_path.name}</a></li>
        </ul>
      </div>
    </body>
    </html>
    """
    report_path.write_text(html, encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t1_map_path = out_dir / "t1_map.nii.gz"
    t2w_3d = None

    if args.reports_only:
        if not t1_map_path.exists():
            raise FileNotFoundError(f"--reports-only requested, but T1 map was not found: {t1_map_path}")
        t1_map_data, _, _ = load_nifti(t1_map_path)
        t1_map = as_3d(t1_map_data)
        if args.t2w:
            t2w_data, _, _ = load_nifti(args.t2w)
            t2w_3d = as_3d(t2w_data)
    else:
        t2w_data, t2w_affine, t2w_header = load_nifti(args.t2w)
        t2w_3d = as_3d(t2w_data)

        if args.erode > 0:
            full_mask = erode_mask(full_mask, iterations=args.erode)

        ti_values = [float(v) for v in args.ti] if args.ti else get_inversion_times_from_JSON(args.mdeft)
        mdeft_volumes, mdeft_affine, mdeft_header, ti_values = prepare_mdeft_series(args.mdeft, ti_values)
        if mdeft_volumes.shape[0] != len(ti_values):
            raise ValueError("Number of MDEFT volumes does not match the number of TI values")

        if mdeft_volumes.shape[1:] != t2w_3d.shape:
            raise ValueError("MDEFT and T2w shapes must match when co-registration is disabled")

        mdeft_stack = np.asarray(mdeft_volumes, dtype=np.float32)
        full_mask = np.ones(t2w_3d.shape, dtype=bool)
        t1_map = fit_t1_map(mdeft_stack, ti_values, full_mask)
        save_nifti(t1_map, t2w_affine, t2w_header, t1_map_path)

    t1_slice_fig = out_dir / "t1_slice.png"
    z = t1_map.shape[2] // 2
    if t2w_3d is not None and t2w_3d.shape == t1_map.shape:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(rotate_clockwise_2d(t2w_3d[:, :, z]), cmap="gray", origin="lower")
        axs[0].set_title("T2w anatomical reference")
        axs[0].axis("off")
        # add_orientation_labels(axs[0], "axial")
        cax = axs[1].imshow(rotate_clockwise_2d(t1_map[:, :, z]), cmap="magma", origin="lower", vmin=0, vmax=5000)
        axs[1].set_title("T1 map")
        axs[1].axis("off")
        #add_orientation_labels(axs[1], "axial")
        fig.colorbar(cax, ax=axs[1], fraction=0.046, pad=0.04)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        cax = ax.imshow(rotate_clockwise_2d(t1_map[:, :, z]), cmap="magma", origin="lower", vmin=0, vmax=5000)
        ax.set_title("T1 map")
        ax.axis("off")
        # add_orientation_labels(ax, "axial")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(t1_slice_fig, dpi=200)
    plt.close(fig)

    hist_fig = out_dir / "t1_histogram.png"
    vals = t1_map[np.isfinite(t1_map)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(vals, bins=50, color="steelblue", alpha=0.8)
    ax.set_xlabel("T1 (ms)")
    ax.set_ylabel("Voxel count")
    ax.set_title("T1 map histogram")
    fig.tight_layout()
    fig.savefig(hist_fig, dpi=200)
    plt.close(fig)

    spider_fig = None

    parcellation_data = None
    atlas_overlay_figs = []
    if args.parcellation:
        parcellation_data, _, _ = load_nifti(args.parcellation)
        parcellation_data = as_3d(parcellation_data)
        stats_df = summarize_t1_regions(t1_map, parcellation_data)
        csv_path = out_dir / "t1_region_summary.csv"
        save_results_csv(stats_df, csv_path)

        sigma_lut = find_sigma_lut(args.parcellation)
        label_map = load_sigma_lut(sigma_lut)
        spider_fig = out_dir / "t1_spider_plot.png"
        make_spider_plot(spider_fig, stats_df, label_map)

        atlas_on_t1_fig = out_dir / "atlas_overlay_t1.png"
        make_atlas_overlay_figure(atlas_on_t1_fig, t1_map, parcellation_data, "Atlas overlay on T1 map")
        atlas_overlay_figs.append(atlas_on_t1_fig)

        if t2w_3d is not None and t2w_3d.shape == parcellation_data.shape:
            atlas_on_t2_fig = out_dir / "atlas_overlay_t2w.png"
            make_atlas_overlay_figure(atlas_on_t2_fig, t2w_3d, parcellation_data, "Atlas overlay on T2w")
            atlas_overlay_figs.append(atlas_on_t2_fig)
    else:
        csv_path = out_dir / "t1_region_summary.csv"
        pd.DataFrame(columns=["region", "count", "mean_t1", "median_t1", "std_t1", "min_t1", "max_t1"]).to_csv(csv_path, index=False)

    report_path = out_dir / args.report_name
    write_html_report(
        output_dir=out_dir,
        report_name=args.report_name,
        inputs={"t2w": args.t2w if args.t2w else "Not supplied"},
        t1_fig=t1_slice_fig,
        histogram_fig=hist_fig,
        t1_map_path=t1_map_path,
        mdeft_paths=[Path(p) for p in (args.mdeft or [])],
        parcellation_path=args.parcellation,
        atlas_figs=atlas_overlay_figs,
        spider_fig=spider_fig,
    )

    print(f"Saved T1 map to: {t1_map_path}")
    print(f"Saved region statistics to: {out_dir / 't1_region_summary.csv'}")
    print(f"Saved HTML report to: {report_path}")


if __name__ == "__main__":
    main()
