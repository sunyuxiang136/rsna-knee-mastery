# Reproducing the Kaggle run

> The notebook is a **scored Kaggle inference notebook**. It needs a T4 GPU, internet **off**, the
> competition data, and the exact weight datasets it checks by SHA-256. Without them it will stop
> by design; it does not substitute other weights.

## Requirements

- A Kaggle account that has joined the *RSNA Knee Abnormality Detection* competition
- Notebook settings: GPU **T4**, Internet **off**
- Attached inputs referenced in the notebook (names as used upstream):
  - `rsna-knee-publication-swa-weights-20261007` (Raptor-224 weights)
  - `rsna-knee-s6fall-weights` (S6 weights + code manifest)
  - the weights/code bundle used for the R6 stage
  - the competition data

Dataset slugs and hashes are hard-coded in `reference/*.py`; check them there before running.

## Steps

1. Import `notebooks/rsna-knee-accuracy50-e8-replace-raptor384-20261020.ipynb` into Kaggle.
2. Attach the inputs above and enable the T4 GPU with Internet off.
3. Set `RUN_INFERENCE = True` in the Raptor cell (it is a guard that defaults to a no-op in some cells).
4. Run all cells. Each stage writes CSVs under `/kaggle/working`; Stage 3 writes `submission.csv`
   plus a receipt (`replacement_receipt.json`) recording component hashes.

## Verifying the fusion logic locally (no GPU)

```bash
pip install -e ".[dev]"
pytest                          # unit tests of the ranking/fusion/cropping logic
python examples/beginner_demo.py
```

The package in `src/rsna_knee` re-implements the *logic* of cropping, slot sampling, attention
pooling and Stage 3 fusion so it can be tested without weights or data. It is not a drop-in
replacement for the model code in `reference/`.

## Known limitations

- No training code, no labels, and no weights are included.
- Reported scores (public LB ≈ 0.95) come from comments in the upstream notebooks and have not been
  re-verified here; the final private-LB placement is not recorded in this repository.

---

## Related Documentation

- [Architecture Overview](architecture.md): Visual data flow and design choices.
- [Dataset & Label Engineering](dataset_and_labels.md): Clinical context and weak supervision details.
- [FAQ](faq.md): Common architectural and engineering questions.
- [API Reference](api_reference.md): Public function reference for `rsna_knee`.

