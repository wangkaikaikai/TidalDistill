# TidalDistill

TidalDistill is a multilevel teacher–student knowledge-distillation framework for dynamic tidal volume (VT) estimation from non-gas cardiopulmonary exercise testing variables and subject-level characteristics across clinical populations.

This repository provides the inference implementations of the Multi-scale Relational Teacher (MRT) and the Dual-branch Adaptive Estimator (DAE), together with reference pretrained checkpoints. DAE is the model used for inference after distillation.

## Model overview

MRT integrates multi-scale temporal encoding, variable-specific lag alignment, variable-relation modelling, and learnable wavelet representations. DAE combines static-conditioned branch-specific feature selection, temporal encoding, variable-relation fusion, and gated trend–residual decoding. Both models accept a 32-step window with 17 input features and estimate VT at the final window position. Intermediate outputs are retained for model interpretation.

## Repository contents

- `src/tidaldistill/model.py`: MRT and DAE architectures and checkpoint loaders.
- `weights/dae_reference.pth`: reference DAE checkpoint stored as a PyTorch state dictionary.
- `weights/mrt_reference.pth`: reference MRT checkpoint stored as a PyTorch state dictionary.
- `weights/SHA256SUMS`: SHA-256 checksums for verifying the checkpoints.
- `preprocessing.json`: input feature order, categorical encodings, and training-set normalisation parameters required by the checkpoints.
- `pyproject.toml`: package metadata and dependencies.

## Input and output

The required input tensor shape is `[batch, 32, 17]`. Features must follow the exact order listed in `preprocessing.json`. Continuous inputs are standardised with the provided training-set parameters. Categorical inputs use the specified integer encodings and are not standardised.

Continuous predictions are obtained by sliding the input window and retaining the DAE prediction at each window endpoint.

Model outputs are in the standardised target space. Use the target mean and scale in `preprocessing.json` to restore predictions to litres.

## Data

No participant-level or clinical data are included. Data access is subject to the data availability statement in the associated manuscript, institutional approvals, participant consent, and applicable privacy requirements.

## License

The source code and model checkpoints are released under the MIT License.

## Citation

Citation metadata will be updated when the article bibliographic record and DOI are available.
