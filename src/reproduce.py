"""Run the frozen scorer in a disposable checkout and keep its output separately."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from .backbone.common import sha256
from .layout import frozen_workspace, release_path
from .release import verify


def run_frozen(output, *, validation=False):
    destination = Path(output).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"Choose a new output directory: {destination}")
    if not validation:
        verify()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with frozen_workspace(validation=validation) as (checkout, env):
        if validation:
            command = ["research_v6.train", "--config", "research_v6/no_rank_config.yaml", "--out", "validation-output"]
            generated = checkout / "validation-output"
        else:
            command = ["research_v6.infer"]
            generated = checkout / "research_v6/runs/submission"
        subprocess.run([sys.executable, "-m", *command], cwd=checkout, env=env, check=True)
        with TemporaryDirectory(prefix=".egma-output-", dir=destination.parent) as staging:
            staged_output = Path(staging) / "submission"
            shutil.copytree(generated, staged_output)
            if os.path.lexists(destination):
                raise FileExistsError(f"Output directory appeared during reproduction: {destination}")
            staged_output.rename(destination)

    if validation:
        print(f"Validation results: {destination / 'summary.json'}")
    else:
        digest = sha256(destination / "submissions.zip")
        print(f"Reproduced archive: {destination / 'submissions.zip'}")
        print(f"SHA-256: {digest}")
        print(f"Matches archived ZIP byte for byte: {digest == sha256(release_path('submissions_egma.zip'))}")
