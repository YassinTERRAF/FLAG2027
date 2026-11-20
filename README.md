# FLAG 2027 - EGMA

Evidence-Gated Multiscale Association (EGMA) matches a face and a voice using learned pairwise similarity and gated score correction. The architecture figure separates the shared feature and matching pipeline, graph-guided training, and independent-trial inference. The description below reflects the reported implementation: graph evidence is confined to training, while deployment uses the current face–voice pair and fixed model parameters.

This repository contains EGMA’s architecture, training scripts, configurations, and scoring code. The team’s leaderboard record shows **3.41% mean EER** for CodaBench submission **1057368** using the scorer associated with that submission.

## The Proposed EGMA Architecture

![EGMA architecture: shared representation and pairwise matching, graph-guided training, and independent-trial inference](docs/EGMA_Architecture.png)

*Panel A constructs the shared pair representation. Panel B shows graph-guided learning on training episodes. Panel C shows the independent-trial inference pathway described below.*

1. **Frozen feature extraction.** EGMA represents faces using organizer-provided VGGFace features, FaceNet features from a model trained on VGGFace2, and a ViT age representation. For voices, it combines organizer-provided ECAPA features, Microsoft WavLM Base Plus speaker embeddings, and an audEERING Wav2Vec2 age/gender representation. The public encoders remain frozen, and the release registries pin their versions and weights. These modality-specific representations feed the mapper ensemble shown in panel A.

2. **Pairwise matching.** Four trained face–voice mappers project the features into a shared 128-dimensional space. EGMA averages their cosine-based scores to form the base similarity. In the `no_gender` configuration, each mapper includes a fixed gender-posterior agreement term. The head’s six-value pair descriptor contains the base score, mean cosine similarity, mean gender-posterior agreement, mean posterior entropy for each modality, and the protocol flag. Together, the base score and descriptor form the pair representation passed to the subsequent training or inference pathway.

3. **Multiscale graph evidence.** Panel B constructs evidence within training episodes. One token represents hard cluster co-occurrence, and four tokens represent soft face–voice similarity kernels. Three kernels use FaceNet and WavLM at different bandwidths; the fourth uses VGGFace and ECAPA. The query’s exact media pair, including duplicate copies, is excluded from its support. Each token records observed support, expected support under independent face and voice affinities, their log ratio, the support degree for each modality, and a reliability measure. These quantities belong to the graph-guided training pathway and are not inputs to the independent-trial inference pathway.

4. **Gated score correction.** In the graph-guided training head, the pair descriptor supplies the attention query, while the evidence tokens supply keys and values. The attended evidence and pair representation determine a bounded residual. A reliability gate scales this residual according to the available support. The training score combines the base pair score, the hard-cluster log-ratio prior, and the gated residual. The graph-guided training head contains **3,540 parameters**. In panel C, the deployed correction is instead conditioned on the current pair’s features and fixed learned parameters. This parameter count refers to the graph-guided training head.

The graph-based EGMA head trains on episodes drawn from 56 English identities. Each fitting episode uses a cross-fit mapper trained on separate identities. The original 14 reserved identities are excluded from the head’s development and refit. The selected loss gives equal weight to verification in the `gender` and `no_gender` protocols. A support-drop consistency term encourages the residual to remain stable when supporting training edges are removed. Additional penalties discourage large residuals and large departures from the initial kernel bandwidths. The ranking loss weight is zero, and the graph-guided training head was refit for 23 epochs. No audio encoder was fine-tuned or adapted using Bengali speech.

During training, EGMA constructs graph evidence from training identities. The graph-guided head scores each pair with full and randomly reduced support, sharing parameters across both passes. The training objective combines protocol-balanced binary verification with a Smooth L1 consistency loss that encourages the reduced-support correction to match the detached full-support correction. The pair descriptor supplies the attention query, while graph evidence tokens supply the keys and values.

A separate pair-conditioned inference module is trained using face–voice descriptors and features. At inference, this frozen module maps the current face–voice representation to a compatibility score, exported as negative similarity.

## Evaluation inference

**Independent-trial inference (panel C).** A single face–voice pair passes through the frozen feature extraction and trained mapper ensemble. The resulting pair representation enters a pair-conditioned correction module, which uses that pair’s features and fixed learned parameters. The final similarity is exported with its sign reversed, so lower scores indicate stronger matches.


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

These steps restore the exact inputs for the frozen release associated with submission **1057368**.

1. Download `egma-evaluation-checkpoints.zip` and `egma-validation-checkpoints.zip` from the [public checkpoint release](https://github.com/YassinTERRAF/FLAG2027/releases/tag/review-checkpoints-1057368).
2. Sign in with the invited GitHub account, accept the collaborator invitation, and download `egma-organizer-inputs-1057368.zip` from the [private reviewer release](https://github.com/YassinTERRAF/FLAG2027-review/releases/tag/review-inputs-1057368). This archive contains the pinned encoder files, cached features, and archived score references.
3. Extract all three ZIPs directly into the source checkout, alongside `README.md`. Keep the `release/` paths already present in each archive. Place the official challenge data under `data/` using the original directory structure. Keep the tracked `requirements-lock.txt` and `docs/EGMA_RELEASE.json` in place.

Before validation or evaluation, run:

```bash
python -m flag_egma verify
```

The supplied original `validate` and `reproduce` configurations require PyTorch MPS. Run those configurations on an Apple Silicon Mac. The Windows and Linux instructions cover package installation.

On macOS, confirm that the active environment can access MPS before running validation or scoring. See the [PyTorch MPS documentation](https://docs.pytorch.org/docs/2.6/notes/mps.html).

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```


### English identity-holdout validation

```bash
python -m flag_egma validate --output ../egma-validation
```

The original command repeats the development procedure for the selected graph-based EGMA variant using saved identity splits, cross-fit mappers, and graph settings. It fits the validation heads, prints `ARCHITECTURE SUMMARY`, and saves `summary.json` and `folds.json`. Compare `multiscale_attention.mean_english_eer_percent` in the generated summary with the archived graph-based validation result of **7.35%** in `release/validation/association/architecture_no_rank/summary.json`. The archived fold means are **9.21%**, **6.41%**, and **6.42%**.

### Evaluation submission ZIP

```bash
python -m flag_egma reproduce --output ../egma-evaluation
```

This command runs the frozen independent-trial scorer described in panel C. Each face–voice pair passes through the frozen feature extractors, four trained mappers, and pair-conditioned correction module to produce a matching score. Inference uses fixed parameters and preprocessing statistics established during training.

The command writes `../egma-evaluation/submissions.zip`, four score files, and `manifest.json`. It prints the generated ZIP’s SHA-256 for artifact tracking.
