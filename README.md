# rbyrct-geology

Synthetic adaptive-tomography probe for resolving a small inclusion in heterogeneous geological material.

## Question
Can adaptive projection selection preserve a small inclusion against a strongly heterogeneous background using fewer views?

## V0 phantom
Rock-like circular core with correlated texture, mineral blobs, fractures, and one compact high-density inclusion.

## Metrics
MSE, SSIM, inclusion IoU, contrast-to-noise ratio (CNR), localization error, ray count, runtime.

## Run
```bash
python experiments/experiment_001.py
```
