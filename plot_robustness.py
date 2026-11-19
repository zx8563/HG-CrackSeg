"""Plot clean and corruption results from summary.csv."""

from __future__ import annotations
import argparse
import math
import re
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

CORRUPTIONS = (
    ("gaussian_noise", "GN", "Gaussian noise"),
    ("brightness_shift", "BS", "Brightness shift"),
    ("uneven_light", "UL", "Uneven light"),
    ("motion_blur", "MB", "Motion blur"),
    ("jpeg_compression", "JPEG", "JPEG compression"),
    ("marker_distractor", "MD", "Marker distractor"),
)
CORRUPTION_IDS = tuple((f"{key}/{level}" for (key, _, _) in CORRUPTIONS for level in (1, 2, 3)))
SYNTHETIC_IDS = ("clean",) + CORRUPTION_IDS
PLOT_TYPES = ("heatmap", "radar", "severity")
METRICS = {
    "mi_iou": "IoU",
    "mi_dice": "Dice",
    "mi_cldice": "clDice",
    "mi_precision": "Precision",
    "mi_recall": "Recall",
}
REQUIRED_COLUMNS = (
    "model_id",
    "model_name",
    "dataset_id",
    "dataset_kind",
    "corruption",
    "severity",
    "num_images",
    "status",
    *METRICS,
)


def normalized_name(value: str) -> str:
    return re.sub("[^a-z0-9]+", "", value.casefold())


def read_summary(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("summary.csv contains duplicate column names.")
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"summary.csv is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("summary.csv contains no results.")
    for column in ("model_id", "model_name", "dataset_id", "dataset_kind", "status", "corruption"):
        frame[column] = frame[column].astype(str).str.strip()
    for column in ("model_id", "model_name", "dataset_id"):
        if (frame[column] == "").any():
            raise ValueError(f"Blank {column} in summary.csv.")
    duplicated = frame.duplicated(["model_id", "dataset_id"], keep=False)
    if duplicated.any():
        pairs = frame.loc[duplicated, ["model_id", "dataset_id"]].to_dict("records")
        raise ValueError(f"Duplicate model/dataset rows: {pairs[:8]}")
    names_per_id = frame.groupby("model_id", sort=False)["model_name"].nunique()
    if (names_per_id != 1).any():
        raise ValueError(
            f"Inconsistent display names for: {names_per_id[names_per_id != 1].index.tolist()}"
        )
    return frame


def model_names(frame: pd.DataFrame) -> dict[str, str]:
    return dict(
        frame[["model_id", "model_name"]].drop_duplicates().itertuples(index=False, name=None)
    )


def resolve_models(frame: pd.DataFrame, requested: Sequence[str] | None) -> list[str]:
    names = model_names(frame)
    if not requested:
        return list(names)
    resolved = []
    for item in requested:
        key = normalized_name(item)
        matches = [
            mid
            for (mid, name) in names.items()
            if key in (normalized_name(mid), normalized_name(name))
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Unknown or ambiguous model {item!r}; available IDs: {', '.join(names)}"
            )
        if matches[0] in resolved:
            raise ValueError(f"Model selected more than once: {item}")
        resolved.append(matches[0])
    return resolved


def is_proposed(mid: str, name: str) -> bool:
    return any((normalized_name(value) in {"hgcrackseg", "proposed"} for value in (mid, name)))


def default_radar_models(models: Sequence[str], names: dict[str, str]) -> list[str]:
    chosen = [mid for mid in models if is_proposed(mid, names[mid])]
    chosen.extend((mid for mid in models if mid not in chosen))
    return chosen[:5]


def metric_matrix(
    frame: pd.DataFrame, models: Sequence[str], datasets: Sequence[str], metric: str
) -> np.ndarray:
    indexed = frame.set_index(["model_id", "dataset_id"])
    values = np.empty((len(models), len(datasets)), dtype=float)
    errors = []
    for i, mid in enumerate(models):
        for j, did in enumerate(datasets):
            label = f"{mid}/{did}"
            if (mid, did) not in indexed.index:
                errors.append(f"{label}: missing row")
                continue
            row = indexed.loc[mid, did]
            if row["status"].casefold() != "completed":
                errors.append(f"{label}: status={row['status']!r}, expected 'completed'")
                continue
            expected_kind = "clean" if did == "clean" else "corruption"
            if row["dataset_kind"] != expected_kind:
                errors.append(f"{label}: dataset_kind must be {expected_kind!r}")
            if expected_kind == "corruption":
                (expected_corruption, expected_severity) = did.rsplit("/", 1)
                try:
                    valid_severity = float(row["severity"]) == int(expected_severity)
                except (TypeError, ValueError):
                    valid_severity = False
                if row["corruption"] != expected_corruption or not valid_severity:
                    errors.append(f"{label}: corruption/severity disagrees with dataset_id")
            try:
                count = float(row["num_images"])
                if not math.isfinite(count) or count <= 0 or (not count.is_integer()):
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"{label}: num_images must be a positive integer")
            try:
                value = float(row[metric])
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError
                values[i, j] = value
            except (TypeError, ValueError):
                errors.append(f"{label}: {metric} must be a finite fraction in [0, 1]")
    if errors:
        preview = "\n  ".join(errors[:12])
        suffix = f"\n  ... and {len(errors) - 12} more." if len(errors) > 12 else ""
        raise ValueError(f"Incomplete or invalid data for {metric}:\n  {preview}{suffix}")
    return values


def radar_values(synthetic: np.ndarray, mode: str) -> np.ndarray:
    if synthetic.ndim != 2 or synthetic.shape[1] != 19 or (not np.isfinite(synthetic).all()):
        raise ValueError("Radar requires clean plus all 18 finite corruption values per model.")
    means = synthetic[:, 1:].reshape(len(synthetic), 6, 3).mean(axis=2)
    if mode == "performance":
        return means * 100.0
    if (synthetic[:, 0] <= 0).any():
        raise ValueError(
            "Clean metric is zero: clean-relative radar retention is undefined; use --radar-metric performance."
        )
    return means / synthetic[:, :1] * 100.0


def setup_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib: install it in the Python environment used to run this script."
        ) from exc
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def model_styles(models: Sequence[str], names: dict[str, str], plt) -> dict[str, dict]:
    palette = list(plt.get_cmap("tab20").colors)
    styles = {}
    baseline = 0
    for mid in models:
        proposed = is_proposed(mid, names[mid])
        styles[mid] = {
            "color": (
                "#B32635"
                if proposed
                else palette[
                    [0, 2, 4, 8, 10, 12, 14, 16, 18, 1, 3, 5, 9, 11, 13, 15, 17, 19][baseline % 18]
                ]
            ),
            "linewidth": 2.8 if proposed else 1.35,
            "marker": "D" if proposed else ("o", "s", "^", "v", "P", "X")[baseline % 6],
            "markersize": 5 if proposed else 3.5,
            "alpha": 1.0 if proposed else 0.82,
            "zorder": 10 if proposed else 3,
        }
        if not proposed:
            baseline += 1
    return styles


def save_figure(fig, stem: Path, plt) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        output = stem.with_suffix(f".{extension}")
        fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output}")
    plt.close(fig)


def plot_heatmap(values, models, names, metric, out_dir, plt):
    (fig, ax) = plt.subplots(figsize=(16, max(2.5, 0.43 * len(models) + 1.8)))
    heat = ax.imshow(values * 100, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    for row in range(len(models)):
        for col in range(19):
            ax.text(
                col,
                row,
                f"{values[row, col] * 100:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if values[row, col] >= 0.6 else "#152334",
            )
    ax.set_yticks(range(len(models)), [names[mid] for mid in models])
    ax.set_xticks(
        range(19),
        ["Clean"] + [f"{short}-{level}" for (_, short, _) in CORRUPTIONS for level in (1, 2, 3)],
        rotation=45,
        ha="right",
    )
    for boundary in (0.5, 3.5, 6.5, 9.5, 12.5, 15.5):
        ax.axvline(boundary, color="white", linewidth=1.8)
    ax.tick_params(axis="both", length=0)
    ax.set_title(f"{METRICS[metric]} across clean and corrupted test sets")
    fig.colorbar(heat, ax=ax, label=f"{METRICS[metric]} (%)", pad=0.02)
    fig.tight_layout()
    save_figure(fig, out_dir / f"heatmap_{metric}", plt)


def plot_radar(values, models, names, styles, metric, mode, out_dir, plt):
    (fig, ax) = plt.subplots(figsize=(8.2, 7.1), subplot_kw={"projection": "polar"})
    angles = np.linspace(0, 2 * np.pi, len(CORRUPTIONS), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for row, mid in zip(values, models):
        ax.plot(closed_angles, np.r_[row, row[0]], label=names[mid], **styles[mid])
    upper = max(100.0, math.ceil(float(values.max()) * 1.06 / 10) * 10)
    ax.set_ylim(0, upper)
    if mode == "retention":
        ax.plot(
            np.linspace(0, 2 * np.pi, 361),
            np.full(361, 100.0),
            color="#666666",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )
    ax.set_xticks(angles, [label.replace(" ", "\n", 1) for (_, _, label) in CORRUPTIONS])
    ax.tick_params(axis="x", pad=14)
    ax.set_rlabel_position(22.5)
    ax.grid(color="#BAC2CA", alpha=0.6)
    title = (
        f"{METRICS[metric]} retention (%)"
        if mode == "retention"
        else f"Mean corrupted {METRICS[metric]} (%)"
    )
    ax.set_title(title + "\nMean over three severities", pad=30)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    fig.subplots_adjust(bottom=0.22, top=0.82)
    save_figure(fig, out_dir / f"radar_{metric}_{mode}", plt)


def plot_severity(values, models, names, styles, metric, out_dir, plt):
    (fig, axes) = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    minimum = max(0.0, math.floor(float(values.min()) * 100 / 10) * 10 - 5)
    maximum = min(100.0, math.ceil(float(values.max()) * 100 / 10) * 10 + 5)
    for index, (ax, (_, _, label)) in enumerate(zip(axes.flat, CORRUPTIONS)):
        selected = np.column_stack((values[:, 0], values[:, 1 + 3 * index : 4 + 3 * index]))
        for row, mid in zip(selected, models):
            ax.plot(range(4), row * 100, label=names[mid], **styles[mid])
        ax.set_title(label)
        ax.set_xticks(range(4), ["0\nClean", "1", "2", "3"])
        ax.set_xlim(-0.1, 3.1)
        ax.set_ylim(minimum, maximum)
        ax.grid(axis="y", color="#D7DEE4", linewidth=0.7)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{METRICS[metric]} (%)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Severity")
    (handles, labels) = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(models)), frameon=False)
    fig.suptitle("Performance by corruption severity", fontsize=14)
    legend_rows = math.ceil(len(models) / 4)
    fig.tight_layout(rect=(0, 0.04 + 0.025 * legend_rows, 1, 0.95))
    save_figure(fig, out_dir / f"severity_{metric}", plt)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/robustness/summary.csv"),
        help="Unified long-form summary.csv",
    )
    parser.add_argument(
        "--out-dir", type=Path, help="Output directory; default: <input directory>/plots"
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        default=["all"],
        help="all or a selection of heatmap radar severity (spaces or commas)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model IDs or quoted names; at most five when radar is requested",
    )
    parser.add_argument(
        "--metric",
        choices=tuple(METRICS),
        default="mi_iou",
        help="Metric for heatmap, radar and severity",
    )
    parser.add_argument("--radar-metric", choices=("retention", "performance"), default="retention")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        requested_plots = [
            item.strip() for group in args.plots for item in group.split(",") if item.strip()
        ]
        unknown = set(requested_plots) - set(PLOT_TYPES) - {"all"}
        if unknown or not requested_plots:
            raise ValueError(
                f"Invalid --plots selection: {requested_plots}; use {', '.join(PLOT_TYPES)} or all."
            )
        plots = (
            list(PLOT_TYPES) if "all" in requested_plots else list(dict.fromkeys(requested_plots))
        )
        frame = read_summary(args.input)
        names = model_names(frame)
        models = resolve_models(frame, args.models)
        radar_models = models if args.models else default_radar_models(models, names)
        if "radar" in plots and len(radar_models) > 5:
            raise ValueError(
                "Radar supports at most five selected models; restrict --models or omit radar from --plots."
            )
        data = {}
        if "heatmap" in plots:
            data["heatmap"] = metric_matrix(frame, models, SYNTHETIC_IDS, args.metric)
        if "severity" in plots:
            data["severity"] = metric_matrix(frame, models, SYNTHETIC_IDS, args.metric)
        if "radar" in plots:
            data["radar"] = radar_values(
                metric_matrix(frame, radar_models, SYNTHETIC_IDS, args.metric), args.radar_metric
            )
        plt = setup_plotting()
        styles = model_styles(models, names, plt)
        out_dir = args.out_dir if args.out_dir else args.input.parent / "plots"
        for plot in plots:
            if plot == "heatmap":
                plot_heatmap(data[plot], models, names, args.metric, out_dir, plt)
            elif plot == "radar":
                print("Radar models: " + ", ".join(radar_models))
                plot_radar(
                    data[plot],
                    radar_models,
                    names,
                    styles,
                    args.metric,
                    args.radar_metric,
                    out_dir,
                    plt,
                )
            elif plot == "severity":
                plot_severity(data[plot], models, names, styles, args.metric, out_dir, plt)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
