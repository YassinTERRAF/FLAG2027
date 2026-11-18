"""Map frozen release paths to the organized checkout and its private release."""

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
FROZEN_BACKBONE = Path("src/flag2027")
SOURCE_PACKAGES = {
    "research_v2": "baseline",
    "research_v5": "graph",
    "research_v6": "association",
}

CONFIGS = {
    "research_v2/config.yaml": "configs/baseline/default.yaml",
    "research_v2/ranking_config.yaml": "configs/baseline/ranking.yaml",
    "research_v5/config.yaml": "configs/graph/default.yaml",
    "research_v6/config.yaml": "configs/association/default.yaml",
    "research_v6/long_config.yaml": "configs/association/with_ranking.yaml",
    "research_v6/no_rank_config.yaml": "configs/association/selected.yaml",
}

# Keys here are the literal paths read by the frozen code and manifests.
RELEASE_INPUTS = {
    "artifacts/final": "release/checkpoints/mapper",
    "artifacts/demographic.npz": "release/features/training/paired_features.npz",
    "artifacts/demographic.json": "release/features/training/paired_features.json",
    "artifacts/search_demographic": "release/validation/mapper_selection",
    "models": "release/encoders",
    "submissions.zip": "release/submissions/graph-prior.zip",
    "submissions_egma.zip": "release/submissions/egma.zip",
    "research_v2/runs/ranking_ablation": "release/validation/baseline_folds",
    "research_v5/runs/validation_preserved_baseline": "release/validation/graph_selection",
    "research_v5/runs/training_content_hashes.json": "release/validation/training_content_hashes.json",
    "research_v5/runs/evaluation_features_preserved_baseline": "release/features/evaluation_pool",
    "research_v5/runs/NONCOMPLIANT_submission": "release/features/graph_prior",
    "research_v6/runs/final": "release/checkpoints/association",
    "research_v6/runs/submission": "release/submissions/record",
    "research_v6/runs/architecture": "release/validation/association/architecture",
    "research_v6/runs/architecture_long": "release/validation/association/architecture_long",
    "research_v6/runs/architecture_no_rank": "release/validation/association/architecture_no_rank",
}


def release_path(path):
    path = Path(path)
    if path.is_relative_to(FROZEN_BACKBONE):
        return PACKAGE / "backbone" / path.relative_to(FROZEN_BACKBONE)
    for original, current in {**CONFIGS, **RELEASE_INPUTS}.items():
        if path.is_relative_to(original):
            return ROOT / current / path.relative_to(original)
    if path.parts and path.parts[0] in SOURCE_PACKAGES:
        return PACKAGE / SOURCE_PACKAGES[path.parts[0]] / Path(*path.parts[1:])
    return ROOT / path


@contextmanager
def frozen_workspace(*, validation=False):
    """Make the same source and inputs available under their historical names."""
    with TemporaryDirectory(prefix="egma-reproduction-") as scratch:
        checkout = Path(scratch)
        ignore = shutil.ignore_patterns("__pycache__", "*.egg-info", ".DS_Store")
        # The frozen backbone derives its workspace root from its source path.
        shutil.copytree(PACKAGE / "backbone", checkout / FROZEN_BACKBONE, ignore=ignore)
        for old, package in SOURCE_PACKAGES.items():
            shutil.copytree(PACKAGE / package, checkout / old, ignore=ignore)
        for old, current in CONFIGS.items():
            (checkout / old).symlink_to(ROOT / current)

        for old, current in RELEASE_INPUTS.items():
            # Generated submission files belong to the temporary checkout.
            if old in {"research_v6/runs/submission", "submissions_egma.zip"}:
                continue
            target = checkout / old
            target.parent.mkdir(parents=True, exist_ok=True)
            source = ROOT / current
            target.symlink_to(source, target_is_directory=source.is_dir())
        (checkout / "data").symlink_to(ROOT / "data", target_is_directory=True)
        if validation:
            # Use the retained cross-fit mappings to rebuild fitting-only episodes.
            source = ROOT / "release/validation/association/architecture"
            for bank in source.glob("fold[0-2]/*_bank"):
                relative = bank.relative_to(source)
                target = checkout / "validation-output" / relative
                shutil.copytree(bank, target)
        yield checkout, {
            **os.environ,
            "PYTHONPATH": str(checkout / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
