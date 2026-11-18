# NONCOMPLIANT transductive research

The user explicitly requested this separate experiment on 17 November 2026, including test clustering and disregarding the organizer rules. **This is not a valid FLAG entry.** The original compliant-pipeline code, four frozen checkpoints and original score ZIP are preserved.

## Measured result

Nested selection on three local English identity holdouts reduced mean EER from **29.11% to 10.89%**. All holdouts come from the 56 previously used development identities; the original 14 audit identities remain excluded. These measurements do not establish multilingual evaluation EER. Hidden evaluation labels are unavailable, so the evaluation score archive has no verified EER or 15% claim.

The earlier reported 26.25% platform score and earlier local 28.79% figures use different systems/trial contexts; neither is the matched baseline for this experiment. Use the 29.11% baseline measured on exactly these local contexts.

## Method actually selected

The original four-seed neural architecture and its frozen public encoders are retained for evaluation. No test gradient training or encoder fine-tuning occurs. Each distinct media file is linked across the evaluation pool by its byte SHA-256. FaceNet identity vectors and WavLM identity vectors are clustered separately by average-linkage cosine distance.

Candidate thresholds and pooling/graph weights are compared on inner labeled English identity holdouts. The selected final cosine-distance thresholds are 0.5 for faces and 0.1 for voices. The graph weight is 1.0 and pseudocount 0.5. Cluster embedding averaging was tested but its selected weight is **zero**; the final method retains per-trial neural scores and adds the trial-graph statistic.

For each face-cluster/voice-cluster pair, count the distinct media pairs present anywhere in the trial list. The additional score is the log ratio of its smoothed count to the count expected from independent cluster degrees. The current **unique media pair** is removed from its count and degrees before its score is computed. Copies of the exact same pair receive one total vote. This prevents the current edge or its repeated copies from supporting its own label. Other edges still contribute: this is expressly cross-trial processing.

The graph can exploit how benchmark positives repeatedly connect corresponding identities while random negative links disperse. The resulting reduction is **not evidence that the independent face–voice model learned better associations**. It depends on repeated identities and the trial sampling process. It is incompatible with the organizer's independent-trial protocol and prohibitions on test clustering, co-occurrence, grouping, and post-hoc score modification.

Supplied official CSV features remain separate for every trial, even when raw audio bytes are identical. Some such ECAPA rows differ slightly across trial files. Content deduplication applies to public encoding and graph counts, without replacing those per-trial official blocks. The corrected independent baseline matches the original exported scores within **3.58e-7**. Local validation was repeated with this preservation and the selected settings/results were unchanged. Earlier diagnostic artifacts remain separate under `runs/validation/`; they are not the exported archive's provenance.

## Validation design

Each of the three V2 fold-control checkpoints is checked against its exact fitting identity list. Inner validation chooses a transductive setting; the corresponding outer checkpoint is evaluated on its disjoint outer identities with that setting. Each local context has two protocols, no_gender and same-gender negatives, with 16 anchors per identity per protocol. Only samples referenced by those trials enter its transductive pool. Exact content deduplication and unique-edge counting match the evaluation procedure.

The final evaluation setting minimizes mean inner-holdout EER across folds; outer EER is not used to choose it. The reported nested outer mean describes the selector, whose selected face threshold is 0.35 in one fold, not an independent evaluation of the final global 0.5 setting. This distinction and all candidate results are retained in the run artifacts.

Limitations: previously researched development identities, English-only validation, unknown source-video independence, context-density sensitivity, and use of single fold controls locally versus the preserved four-seed ensemble for evaluation. Evaluation pools all four cells together; cross-language clustering behavior has no labeled local validation. There are no external identity/name lookups and no labels inferred from filenames or row order. Scoring functions accept no ground-truth identities or match labels.

## Reproduction

The release retains this stage's original source, selection manifests, evaluation feature pool, and graph prior scores. The main EGMA reproduction command uses the preserved pool and prior. See the [reviewer guide](../../../docs/REVIEW.md) for setup and required assets.

The original feature and graph scoring modules are `research_v5.features` and `research_v5.infer`. To regenerate those intermediate artifacts, use a separate checkout so the release inputs and checksums remain intact.
