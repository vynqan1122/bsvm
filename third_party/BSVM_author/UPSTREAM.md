# Upstream BSVM source snapshot

Source: <https://github.com/MojtabaMohasel/BSVM.git>

- Pinned commit: `ee5a7ae7ade4977b41d604dd989b31f4e356d342`
- Commit date: 2025-11-14
- Upstream files retained: `BSVM_linear.py`, `BSVM_poly.py`, `BSVM_RBF.py`,
  `BSVM_sigmoid.py`.
- Added extension: `BSVM_extended.py`.

The upstream repository contains visualization scripts using synthetic 2-D
data, not the complete Table 2 experiment runner. The extension keeps the
author's `Initialsolution -> masterproblem -> extend_samples` control flow in a
reusable estimator and adds the requested priority functions, binary-tree
insertion, local CSV loading, paper split, hyperparameter search and result
tables elsewhere in this package.

At the pinned commit the upstream repository did not include a license file.
The original files are included here solely as a traceable research snapshot;
check with the authors before redistributing them beyond this comparison work.

