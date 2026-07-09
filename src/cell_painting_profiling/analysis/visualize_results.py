from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

from cell_painting_profiling.data.multichannel_dataset import DEFAULT_CHANNEL_ORDER, load_channel_image
from cell_painting_profiling.embeddings.aggregate import get_embedding_columns


REPRESENTATION_FILES = [
    {
        "label": "Random fine-tuned",
        "path": "data/processed/cpg0002-jump-scope/BRO0117059_20X_baseline_12site_trained_compound_fingerprints.csv",
    },
    {
        "label": "Pretrained fine-tuned",
        "path": "data/processed/cpg0002-jump-scope/BRO0117059_20X_baseline_12site_pretrained_augmented_compound_fingerprints.csv",
    },
    {
        "label": "Frozen pretrained",
        "path": "data/processed/cpg0002-jump-scope/BRO0117059_20X_baseline_12site_frozen_pretrained_compound_fingerprints.csv",
    },
]


def percentile_scale(image: torch.Tensor, lower: float = 1.0, upper: float = 99.5) -> torch.Tensor:
    low = torch.quantile(image.flatten(), lower / 100.0)
    high = torch.quantile(image.flatten(), upper / 100.0)
    if float(high - low) <= 0:
        return torch.zeros_like(image)
    return ((image - low) / (high - low)).clamp(0, 1)


def load_site_channel_stack(site_manifest: pd.DataFrame, image_size: int = 224) -> dict[str, torch.Tensor]:
    channels = {}
    for _, row in site_manifest.iterrows():
        image = load_channel_image(row["local_path"])
        image = F.interpolate(
            image.unsqueeze(0).unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        channels[row["channel"]] = percentile_scale(image)
    missing = set(DEFAULT_CHANNEL_ORDER).difference(channels)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Image site is missing channels: {missing_text}")
    return channels


def make_rgb_composite(site_manifest: pd.DataFrame, image_size: int = 224) -> np.ndarray:
    channels = load_site_channel_stack(site_manifest, image_size=image_size)
    red = torch.maximum(channels["rna"], channels["agp"] * 0.6)
    green = torch.maximum(channels["mito"], channels["er"] * 0.6)
    blue = channels["dna"]
    rgb = torch.stack([red, green, blue], dim=-1).clamp(0, 1)
    return rgb.numpy()


def first_site_for_compound(manifest: pd.DataFrame, perturbation_id: str) -> pd.DataFrame:
    compound_rows = manifest.loc[manifest["perturbation_id"] == perturbation_id]
    if compound_rows.empty:
        raise ValueError(f"No manifest rows found for perturbation: {perturbation_id}")
    first_record = sorted(compound_rows["image_record_id"].unique())[0]
    return compound_rows.loc[compound_rows["image_record_id"] == first_record]


def select_retrieval_examples(neighbors: pd.DataFrame) -> list[dict]:
    top1 = neighbors.loc[neighbors["rank"] == 1].copy()
    success = top1.loc[top1["shared_mechanism"]].head(1)
    failure = top1.loc[~top1["shared_mechanism"]].head(1)
    examples = []
    for label, frame in [("Shared mechanism top-1", success), ("Different mechanism top-1", failure)]:
        if frame.empty:
            continue
        query_id = frame.iloc[0]["query_id"]
        query_neighbors = neighbors.loc[neighbors["query_id"] == query_id].sort_values("rank")
        examples.append({"label": label, "query_id": query_id, "neighbors": query_neighbors})
    return examples


def plot_umap_comparison(output_path: str | Path) -> None:
    from umap import UMAP

    frames = []
    for representation in REPRESENTATION_FILES:
        frame = pd.read_csv(representation["path"])
        embedding_columns = get_embedding_columns(frame)
        reducer = UMAP(n_neighbors=5, min_dist=0.3, metric="cosine", random_state=42)
        coordinates = reducer.fit_transform(frame[embedding_columns].to_numpy())
        plot_frame = frame[["compound_name", "mechanism_of_action"]].copy()
        plot_frame["umap_1"] = coordinates[:, 0]
        plot_frame["umap_2"] = coordinates[:, 1]
        plot_frame["representation"] = representation["label"]
        frames.append(plot_frame)
    plot_data = pd.concat(frames, ignore_index=True)

    mechanisms = sorted(plot_data["mechanism_of_action"].unique())
    palette = dict(zip(mechanisms, sns.color_palette("tab10", n_colors=len(mechanisms)), strict=False))
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharex=False, sharey=False)
    for ax, representation in zip(axes, [item["label"] for item in REPRESENTATION_FILES], strict=False):
        subset = plot_data.loc[plot_data["representation"] == representation]
        sns.scatterplot(
            data=subset,
            x="umap_1",
            y="umap_2",
            hue="mechanism_of_action",
            palette=palette,
            s=60,
            edgecolor="black",
            linewidth=0.3,
            ax=ax,
            legend=False,
        )
        ax.set_title(representation)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette[m],
            markeredgecolor="black",
            markersize=7,
            label=m,
        )
        for m in mechanisms
    ]
    fig.legend(handles=handles, title="Mechanism", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle("Compound-level embedding structure by representation strategy", y=1.02)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_retrieval_examples(
    manifest_path: str | Path,
    neighbors_path: str | Path,
    output_path: str | Path,
    output_table: str | Path,
    image_size: int = 224,
) -> None:
    manifest = pd.read_csv(manifest_path)
    neighbors = pd.read_csv(neighbors_path)
    examples = select_retrieval_examples(neighbors)
    if not examples:
        raise ValueError("No retrieval examples could be selected")

    rows = []
    fig, axes = plt.subplots(len(examples), 4, figsize=(12, 3.2 * len(examples)))
    if len(examples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, example in enumerate(examples):
        query_neighbors = example["neighbors"]
        query = query_neighbors.iloc[0]
        panel_items = [("Query", query["query_id"], query["query_compound_name"], query["query_mechanism_of_action"], None)]
        for _, neighbor in query_neighbors.head(3).iterrows():
            panel_items.append(
                (
                    f"Rank {int(neighbor['rank'])}",
                    neighbor["neighbor_id"],
                    neighbor["neighbor_compound_name"],
                    neighbor["neighbor_mechanism_of_action"],
                    bool(neighbor["shared_mechanism"]),
                )
            )
            rows.append(
                {
                    "example_type": example["label"],
                    "query_compound": query["query_compound_name"],
                    "query_mechanism": query["query_mechanism_of_action"],
                    "neighbor_rank": int(neighbor["rank"]),
                    "neighbor_compound": neighbor["neighbor_compound_name"],
                    "neighbor_mechanism": neighbor["neighbor_mechanism_of_action"],
                    "shared_mechanism": bool(neighbor["shared_mechanism"]),
                    "similarity": float(neighbor["similarity"]),
                }
            )
        for column_index, (title, perturbation_id, compound, mechanism, shared) in enumerate(panel_items):
            ax = axes[row_index, column_index]
            composite = make_rgb_composite(first_site_for_compound(manifest, perturbation_id), image_size=image_size)
            ax.imshow(composite)
            marker = "" if shared is None else ("shared" if shared else "different")
            ax.set_title(f"{title}\n{compound}\n{mechanism}\n{marker}", fontsize=8)
            ax.axis("off")
        axes[row_index, 0].set_ylabel(example["label"], fontsize=9)
    fig.suptitle("Representative retrieval examples from the pretrained fine-tuned model", y=1.01)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    output_table = Path(output_table)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_table, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate UMAP and retrieval example figures.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--neighbors", required=True)
    parser.add_argument("--umap-output", required=True)
    parser.add_argument("--retrieval-output", required=True)
    parser.add_argument("--retrieval-table", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_umap_comparison(args.umap_output)
    plot_retrieval_examples(
        manifest_path=args.manifest,
        neighbors_path=args.neighbors,
        output_path=args.retrieval_output,
        output_table=args.retrieval_table,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
