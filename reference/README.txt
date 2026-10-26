Upstream reference code extracted from public Kaggle notebooks.
Not covered by this repository's MIT license; see ../THIRD_PARTY_NOTICES.md.

  notebook_pipeline.py   Full notebook as one script (stages 1-3, fusion in "Cell 10")
  raptor224_run.py       Raptor-224 (CoAtNet) inference, fixed slots, per-finding attention head
  s6_reference.py        S6 (ConvNeXt) controller: watchdog, per-GPU workers, receipts, rank blend

These need a Kaggle T4 environment, the competition data and the exact weight datasets
(SHA-256 checked). See ../docs/reproduction.md.
