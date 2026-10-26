# Third-party notices and provenance

## What is original here

`src/`, `tests/`, `examples/`, `docs/` and the repository tooling are original to this project and
are released under the [MIT License](LICENSE).

## What is not

The files in [`reference/`](reference) and [`notebooks/`](notebooks) are extracted from **public Kaggle
notebooks by other authors**. They are included for study and reproducibility; they are **not**
relicensed under MIT by this repository, and their original terms apply.

Credits, as stated in the notebook's own acknowledgements:

- Raptor CoAtNet/MIL inference recipe and baseline weights — **dreaddevelopment**
  (<https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense>)
- Retrained Raptor and S6 checkpoints, inference scripts — **nartaa**
  (<https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo>)
- Baseline blend notebook and SKM-TEA anatomy V-Net / data pipeline — **heliosli**
  (<https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters>)
- 224-px stage and its share of the Raptor half — **varshithakolupuri**
  (<https://www.kaggle.com/code/varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956>)
- Weak-label sources — stevenleehans, pilkwang, riadmohamed42
- CoAtNet/ConvNeXt backbones — [timm](https://github.com/huggingface/pytorch-image-models)

Data and research tools used by upstream were obtained from the Osteoarthritis Initiative (OAI),
NDA dataset identifier 10.15154/0hcg-f676 (<https://nda.nih.gov/study.html?id=3407>), under its
acknowledgement requirements. The competition data is governed by the
[competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).

> **Maintainer TODO:** confirm the licence of each upstream notebook/dataset before relying on
> redistribution, and remove anything you are not permitted to republish. Part of
> `reference/s6_reference.py` mentions "Apache-2.0 code extras"; keep such notices intact.

## Not medical software

This repository is for research and education. It is not a medical device and must not be used for
clinical decisions.
