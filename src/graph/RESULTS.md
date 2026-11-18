# NONCOMPLIANT transductive research results

**This method violates FLAG rules on evaluation clustering, cross-trial linking, co-occurrence and score modification. Do not present it as an eligible entry.**

| Method | Nested local mean English EER (%) | Fold sample SD (points) |
|---|---:|---:|
| Preserved architecture, independent trial scores | 29.11 | 3.56 |
| Clusters + leave-one-unique-pair-out trial graph | 10.89 | 3.24 |

| Fold | Baseline | Transductive | Faces / voices / trials |
|---|---:|---:|---|
| 0 | 31.41 | 13.82 | 748 / 389 / 1216 |
| 1 | 25.00 | 7.40 | 774 / 382 / 1216 |
| 2 | 30.90 | 11.46 | 758 / 392 / 1152 |

These are labeled English training-identity holdouts, not hidden challenge labels or a four-cell English/Bangla score. Inner validation selected each fold setting before its outer measurement. The original 14 audit identities are excluded; the other 56 development identities have been researched previously. Fold SD is not a confidence interval.

The local mean describes nested selection: two folds chose face threshold 0.5 and one chose 0.35. The final global setting was selected from mean inner EER, not outer results. Local checkpoints are single original-architecture fold models; evaluation retains the original four-seed ensemble. Trial sampling and cross-language behavior limit extrapolation.

The final setting uses average-linkage cosine thresholds **0.5 face / 0.1 voice**, graph weight **1.0**, pseudocount **0.5**, and pooling weight **0.0**. No pretrained encoder or original neural checkpoint was modified.

## Context-density stress check

Settings remained fixed. These extra measurements did not choose or modify the system.

| Anchors per identity per protocol | Baseline English EER (%) | Transductive English EER (%) |
|---|---:|---:|
| 8 | 27.49 | 12.63 |
| 32 | 28.61 | 11.28 |

These contexts use a separate random seed. Repeated-identity density and the benchmark negative-sampling mechanism materially influence this method.

## Evaluation archive

**Evaluation EER is unknown. The requested 15% multilingual evaluation target is not verified.**

Saved: `runs/NONCOMPLIANT_submission/submissions.zip`.

Format checked: exactly four files, **2326** expected trial keys, correct order, finite distance scores. No upload was performed.

Unlabeled evaluation pool formed 23 anonymous face clusters and 42 anonymous voice clusters; no reference identities or labels were obtained.

Archive SHA-256: `f8973012144375dbab12db4d0a37b379ef038f23313c907836a5d5d93a31250e`.

The adjacent `COMPLIANCE_NOTICE.txt` and `NONCOMPLIANT_manifest.json` explicitly identify the prohibited operations and unknown hidden EER. Format validity does not establish eligibility or performance.

## Reproducibility

Commands and algorithm details: [README.md](README.md). All source/data/checkpoint hashes, inner candidate scores, selected configurations, local labels/scores, stress contexts and unlabeled extraction provenance are retained locally under `runs/`.

Original competition system SHA-256: `cda6a932193b7b3118ccb710f37fe408cf1e75289d48e331f8f961b3a06f1ac6`.

Original submission ZIP SHA-256: `9a6234329e160e876ec7683c04094a55771b243dff333c4ea6f2bf81fd298b81`.
