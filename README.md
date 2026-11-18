# FLAG 2027 -   EGMA 

Evidence-Gated Multiscale Association (EGMA) matches a face and a voice using learned pairwise similarity and gated score correction. The architecture figure shows the shared feature and matching pipeline, the graph-based training procedure, and a proposed independent-trial inference path.

This repository contains EGMA’s architecture, training scripts, configurations, and frozen graph-based scoring code. The team’s leaderboard record shows **3.41% mean EER** for CodaBench submission **1057368**, associated with the frozen scorer.

## The Proposed EGMA Architecture

![EGMA architecture overview showing shared feature extraction and matching, graph-based training, and proposed independent-trial inference](docs/EGMA_Architecture.png)


1. **Frozen feature extraction.** EGMA represents faces using organizer-provided VGGFace features, FaceNet features from a model trained on VGGFace2, and a ViT age representation. For voices, it combines organizer-provided ECAPA features, Microsoft WavLM Base Plus speaker embeddings, and an audEERING Wav2Vec2 age/gender representation. The public encoders remain frozen, and the release registries pin their versions and weights.

2. **Pairwise matching.** Four trained face–voice mappers project the features into a shared 128-dimensional space. The base score averages their cosine-based scores, which include a fixed gender-posterior agreement term for `no_gender` trials. The head’s six-value pair descriptor contains the base score, mean cosine similarity, mean gender-posterior agreement, mean posterior entropy for each modality, and the protocol flag.

3. **Multiscale graph evidence.** In each training episode, one token represents hard cluster co-occurrence and four tokens represent soft face–voice similarity kernels. Three kernels use FaceNet and WavLM at different bandwidths, and the fourth uses VGGFace and ECAPA. The head excludes the query’s exact media pair from its own support. Each token records observed support, expected support under independent face and voice affinities, their log ratio, the support degree for each modality, and a reliability measure.

4. **Gated score correction.** The pair descriptor supplies the attention query, and the five evidence tokens supply its keys and values. The head combines the attended evidence with the pair representation to predict a bounded residual. A gate scales that residual using the reliability of the available support. The graph-based score adds the base pair score, the hard-cluster log-ratio term, and the gated residual. The archived graph-based EGMA head adds **3,540 parameters**.

The graph-based EGMA head trains on episodes drawn from 56 English identities. Their training eligibility must be confirmed against the official dataset designation. Each fitting episode uses a cross-fit mapper trained on separate identities. The original 14 reserved identities are excluded from the head’s development and refit. The selected loss gives equal weight to verification in the `gender` and `no_gender` protocols. A consistency term encourages the residual to stay stable when training removes some supporting edges. Additional penalties discourage large residuals and large departures from the initial kernel bandwidths. The ranking loss weight is zero, and the archived head was refit for 23 epochs. No audio encoder was fine-tuned or adapted using Bengali speech.

## Evaluation inference

**Proposed independent-trial path (panel C).** The diagram shows one face–voice pair entering the shared feature and mapper pipeline. In this design, the correction would use only the current pair’s features and fixed parameters learned from permitted training data. The final similarity would be exported directly with its sign reversed, so a lower score indicates a stronger match. Panel C describes a design proposal.


## Reproduce the release

### Set up Python on macOS, Windows, or Linux

Install Miniconda for your operating system and processor. Open the shell for your platform, then change into the repository directory. [Conda installation guide](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)

| Platform | Shell |
| --- | --- |
| macOS on Apple Silicon | Terminal with native arm64 Miniconda |
| Windows (x86-64) | Anaconda Prompt from the Start menu |
| Linux (x86-64) | A terminal with Conda initialized |

Create and activate the environment on any of these platforms.

```bash
conda create -n flag-egma python=3.11 pip -y
conda activate flag-egma
```

On Windows and Linux, install the pinned CPU builds of PyTorch and TorchAudio before installing the rest of the dependencies. [PyTorch 2.6 installation instructions](https://pytorch.org/get-started/previous-versions/#v260)

```bash
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

On all three platforms, install the dependencies and EGMA package from the checkout.

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m flag_egma --help
```

### Verify the frozen release

Restore the authorized `data/` and `release/` bundles, the original `requirements-lock.txt`, and the release manifest at `docs/EGMA_RELEASE.json` before running the checksum check. These paths are relative to the repository root. Package installation uses `requirements.txt`.

```bash
python -m flag_egma verify
```

The supplied `validate` and `reproduce` configurations require PyTorch MPS. Run those commands on an Apple Silicon Mac. The Windows and Linux instructions cover package installation. On macOS, confirm that the active environment can access MPS before running validation or scoring. [PyTorch MPS documentation](https://docs.pytorch.org/docs/2.6/notes/mps.html)

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

### English identity-holdout validation

```bash
python -m flag_egma validate --output ../egma-validation
```

The command repeats the development procedure for the selected EGMA variant using the saved identity splits, cross-fit mappers, and graph settings. It fits the validation heads, prints `ARCHITECTURE SUMMARY`, and saves `summary.json` and `folds.json`. Compare `multiscale_attention.mean_english_eer_percent` in the generated summary with the archived graph-based validation result of **7.35%** in `release/validation/association/architecture_no_rank/summary.json`. The archived fold means are **9.21%**, **6.41%**, and **6.42%**.

### Evaluation submission ZIP

```bash
python -m flag_egma reproduce --output ../egma-evaluation
```

This command runs the frozen graph-based scorer with its archived head, four mappers, and verified evaluation feature pool. It writes `../egma-evaluation/submissions.zip`, four score files, and `manifest.json`, then compares the generated ZIP’s SHA-256 with the archived ZIP.
