# Implementation and manuscript correspondence

This release packages the available main-method experiment code. It does not
silently alter an existing experiment to match a later manuscript description.
The reference manuscript is identified by its SHA-256 in `source_manifest.json`.

| Item | Manuscript | Available code / saved Xenium runs |
| --- | --- | --- |
| Masked cell pooling, Eq. (1) | Denominator is mask sum + epsilon | Denominator uses `clamp_min(1e-6)`; same for nonempty binary masks |
| Retrieval matching, Eq. (2) | Learnable Wq and Wk in a shared matching dimension | Learnable affine query projection to the native token dimension; LayerNorm tokens act as keys, without an additional Wk |
| Attention, Eq. (3) | Cosine attention; original tokens as values | Same cosine attention and unprojected values; tau2=2, tau3=1 in saved Xenium configs |
| Context query, Sec. 2.2 | E2 by default, E1 optional | E1 in both saved Xenium runs and the released main method |
| Shared head dimension | Shared expert projection | LayerNorm + Linear to 512; heads default 512 -> 256 -> classes |
| Router, Eqs. (4)-(5) | Embeddings plus entropy/margin/max probability | Same; input dimension 3*512+9=1545; MLP 1545 -> 256 -> 256 -> 3; softmax weights |
| Fusion, Sec. 2.3 | Probability mixture | Same for training and label-free forward/evaluation |
| Responsibilities, Eq. (6) | Detached posterior responsibilities, sum shown as 1 | Same formula with +epsilon denominator; sum is approximately 1 numerically |
| Loss, Eqs. (7)-(9) | Mixture CE, KL(r || alpha), responsibility-weighted expert CE, load balance | Same terms plus the existing JS conflict weights on gate/expert losses in the saved runs |
| Local crop | 224 x 224 pixels | 256 x 256 crop resized to 224 x 224 |
| Context crop | 1024 x 1024 pixels | 1024 x 1024 crop resized to 256 x 256 for DINO |
| Optimization | AdamW, 3e-4, at most 100 epochs, early stopping on validation loss | AdamW, 1e-3, 30 epochs, best checkpoint by validation macro ROC-AUC; no early stopping |
| Validation | In-house slide holdout and contiguous region of public WSI | Saved Xenium configs use a stratified random-cell 80/20 split; paper region membership is unavailable |

The source's alternative `ctx_attn_query=e2` path used UNI mean pooling as the
query, not the attention-pooled E2 output described in the paper. That alternative
is not exposed in this main-method release. Both query sources are fixed to E1.
No key projection was added: that would change parameter shapes and require
retraining. There is no mean/attention hybrid pooling in this release.

For expert logits z_k and gate weights alpha, the released mixture uses
`softmax(z_k)` with unit expert temperatures. The original cooperative loss
clamps/renormalizes alpha and the resulting probabilities for numerical stability,
then uses `log(p_mix)` as input to weighted smoothed cross-entropy. Label smoothing
is 0.05. Class weights are balanced weights computed on the training split only.

The additional conflict term is `w_i = 1 + 0.5 * JS(p1,p2,p3)`. It weights
`KL(r || alpha)` and `sum_k r_k CE(z_k,y)` only. Responsibilities are detached;
the conflict weights themselves retain gradients, as in the research source.
Xenium loss coefficients are lambda_gate=0.3, lambda_exp=0.3, lambda_lb=0.1.
Load balancing penalizes the squared distance of batch-average alpha from 1/3.

## Packaging changes

- Retained only the main three-expert class and its dataset/backbone/loss dependencies.
- Fixed the public model to the saved E1-query cosine-attention configuration.
- Made forward outputs directly expose the same probability mixture previously
  applied by the `coop_moe` branch of the research trainer. The source model's
  otherwise-unused logit-mixture output is not exposed as a prediction.
- Introduced a focused JSON-config runner, standalone evaluation, explicit split
  manifests, output-directory checks, and finite-loss checks.
- Retained trainable state-dict names; release checkpoints omit frozen encoders.
- Exported attention maps only for selected examples to limit validation memory.
- Preserved the original frozen-backbone gradient behavior. As in the research
  source, recursive `model.train()` also changes backbone module training flags;
  freezing parameters does not itself force every stochastic layer into eval mode.
  Numerical source regression is performed in eval mode with dropout disabled.

The Visium HD config is an interface template derived from the saved cooperative
skin run (8 epochs, tau3=2, lambda_lb=0.05), with multiclass `cell_type` labels and
slide-group splitting. It is not a recovered in-house experiment split.

The release's CPU tests establish mathematical and execution consistency. They
do not validate the manuscript's full-data metrics, GPU throughput, or external
backbone weight availability. This release does not retrain models or fabricate
new results to fill those gaps.
