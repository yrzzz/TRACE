# Release validation

Validated on 2026-09-28 with Python 3.10 and the versions in `requirements.txt`.

Command:

```bash
TRACE_REFERENCE_ROOT=/path/to/original/skin_cell_experts \
  python -m unittest discover -s tests -v
```

Result: **12 tests passed**, including the optional research-source regression.
The tests use CPU, synthetic patches, small frozen token encoders, and a synthetic
TIFF/CSV Xenium specimen. No foundation-model download or full-data training was
performed for this release.

Verified:

- Exact CPU float32 equality of original/released expert embeddings, logits,
  gate weights, local/context attention maps, cooperative loss, and trainable
  parameter gradients with an identical state dict and inputs.
- Exact agreement between standalone forward probabilities and the probability
  mixture inside the training loss. A constructed case distinguishes probability
  mixing from logit mixing.
- Detached responsibilities, the direction KL(r || alpha), normalized attention,
  and finite gradients through both attention pools, expert heads, and router.
- One complete training epoch, best checkpoint selection, head-only checkpoint
  reload, standalone evaluation, and identical reloaded prediction probabilities.
- Matched local/context cell IDs for both correct and incorrect predictions.
- Real TIFF-backed Xenium crop reads, inverse-alignment coordinate conversion,
  compressed annotation loading, level aliases, removal of unannotated cells,
  and consistent target-versus-others labels.
- Explicit split membership and ordered dataset-label fingerprint checks.
- Config parsing and `--help` for training, evaluation, and preparation commands.

Without `TRACE_REFERENCE_ROOT`, the public suite skips only the original-source
comparison. Source file and label checksums are recorded in `source_manifest.json`.

These checks do not establish full-scale GPU performance or recover missing
manuscript train/validation splits. See `implementation.md` for the documented
differences between the manuscript and the available experiments.
