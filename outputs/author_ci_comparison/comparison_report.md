# Author BSVM vs three new c_i functions

## Aggregate deltas versus author_original

| variant | algorithm | comparisons | objective_mean_delta | objective_wins | objective_ties | objective_losses | support_vectors_mean_delta | fewer_support_vectors | model_fits_mean_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| robust_hybrid | New 1 - robust_hybrid | 6 | -0.0161 | 3 | 0 | 3 | 1.1667 | 2 | -486.3333 |
| user_formula_1 | New 2 - user_formula_1 | 6 | -0.0177 | 2 | 0 | 4 | -68.6667 | 3 | -517.3333 |
| user_formula_2 | New 3 - user_formula_2 | 6 | -0.0135 | 2 | 1 | 3 | 4.3333 | 2 | -442.6667 |

## Per-combination deltas

| dataset | paper_group | kernel | variant | algorithm | objective_test_delta | accuracy_test_delta | balanced_accuracy_test_delta | f1_macro_test_delta | minority_f1_test_delta | support_vectors_delta | model_fits_delta | elapsed_tuning_seconds_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | experiment_2 | linear | robust_hybrid | New 1 - robust_hybrid | -0.0455 | -0.0455 | -0.0333 | -0.0165 | 0.0000 | 1 | 8 | 0.2576 |
| cloud | experiment_2 | linear | user_formula_1 | New 2 - user_formula_1 | -0.0455 | -0.0455 | 0.0048 | 0.0769 | 0.2000 | 2 | 28 | 0.1521 |
| cloud | experiment_2 | linear | user_formula_2 | New 3 - user_formula_2 | -0.0455 | -0.0455 | 0.0048 | 0.0769 | 0.2000 | 2 | 28 | 0.1340 |
| cloud | experiment_2 | poly | robust_hybrid | New 1 - robust_hybrid | 0.0455 | 0.0455 | 0.0714 | 0.0762 | 0.1282 | -6 | -14 | -0.5569 |
| cloud | experiment_2 | poly | user_formula_1 | New 2 - user_formula_1 | 0.0455 | 0.0455 | 0.0333 | 0.0341 | 0.0303 | -11 | -22 | -0.4815 |
| cloud | experiment_2 | poly | user_formula_2 | New 3 - user_formula_2 | 0.0455 | 0.0455 | 0.0333 | 0.0341 | 0.0303 | -11 | -22 | -0.4229 |
| cloud | experiment_2 | sigmoid | robust_hybrid | New 1 - robust_hybrid | -0.0455 | -0.0455 | -0.0333 | -0.0288 | -0.0222 | 2 | -13 | 2.3065 |
| cloud | experiment_2 | sigmoid | user_formula_1 | New 2 - user_formula_1 | -0.0455 | -0.0455 | -0.0333 | -0.0288 | -0.0222 | 2 | -13 | 2.2699 |
| cloud | experiment_2 | sigmoid | user_formula_2 | New 3 - user_formula_2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 64 | 2.2111 |
| cloud | experiment_2 | rbf | robust_hybrid | New 1 - robust_hybrid | -0.0909 | -0.0909 | -0.1429 | -0.1754 | -0.3077 | 26 | -28 | 0.3326 |
| cloud | experiment_2 | rbf | user_formula_1 | New 2 - user_formula_1 | -0.0909 | -0.0909 | -0.1429 | -0.1754 | -0.3077 | 27 | -32 | 0.4913 |
| cloud | experiment_2 | rbf | user_formula_2 | New 3 - user_formula_2 | -0.0909 | -0.0909 | -0.1429 | -0.1754 | -0.3077 | 27 | -36 | 0.1454 |
| dataset | multiclass | linear | robust_hybrid | New 1 - robust_hybrid | 0.0325 | 0.0325 | 0.0325 | 0.0305 | - | -38 | -277 | 143.3778 |
| dataset | multiclass | linear | user_formula_1 | New 2 - user_formula_1 | 0.0350 | 0.0350 | 0.0350 | 0.0325 | - | -35 | -335 | 184.4250 |
| dataset | multiclass | linear | user_formula_2 | New 3 - user_formula_2 | 0.0125 | 0.0125 | 0.0125 | 0.0099 | - | -50 | 113 | 271.0483 |
| dataset | multiclass | poly | robust_hybrid | New 1 - robust_hybrid | 0.0075 | 0.0075 | 0.0075 | 0.0063 | - | 22 | -2594 | -8848.9338 |
| dataset | multiclass | poly | user_formula_1 | New 2 - user_formula_1 | -0.0050 | -0.0050 | -0.0050 | -0.0066 | - | -397 | -2730 | -9146.9089 |
| dataset | multiclass | poly | user_formula_2 | New 3 - user_formula_2 | -0.0025 | -0.0025 | -0.0025 | -0.0037 | - | 58 | -2803 | 234.9968 |

## Full results (compact view)

| dataset | paper_group | kernel | algorithm | status | validation_score | objective_test | accuracy_test | f1_macro_test | minority_f1_test | support_vectors | model_fits | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | experiment_2 | linear | Author original BSVM | ok | 0.7222 | 0.6818 | 0.6818 | 0.4054 | 0.0000 | 6 | 37 | - |
| cloud | experiment_2 | linear | New 1 - robust_hybrid | ok | 0.7222 | 0.6364 | 0.6364 | 0.3889 | 0.0000 | 7 | 45 | - |
| cloud | experiment_2 | linear | New 2 - user_formula_1 | ok | 0.7222 | 0.6364 | 0.6364 | 0.4824 | 0.2000 | 8 | 65 | - |
| cloud | experiment_2 | linear | New 3 - user_formula_2 | ok | 0.7222 | 0.6364 | 0.6364 | 0.4824 | 0.2000 | 8 | 65 | - |
| cloud | experiment_2 | poly | Author original BSVM | ok | 0.8333 | 0.6364 | 0.6364 | 0.5417 | 0.3333 | 26 | 61 | - |
| cloud | experiment_2 | poly | New 1 - robust_hybrid | ok | 0.7778 | 0.6818 | 0.6818 | 0.6179 | 0.4615 | 20 | 47 | - |
| cloud | experiment_2 | poly | New 2 - user_formula_1 | ok | 0.7778 | 0.6818 | 0.6818 | 0.5758 | 0.3636 | 15 | 39 | - |
| cloud | experiment_2 | poly | New 3 - user_formula_2 | ok | 0.7778 | 0.6818 | 0.6818 | 0.5758 | 0.3636 | 15 | 39 | - |
| cloud | experiment_2 | sigmoid | Author original BSVM | ok | 0.7778 | 0.6818 | 0.6818 | 0.5111 | 0.2222 | 2 | 92 | - |
| cloud | experiment_2 | sigmoid | New 1 - robust_hybrid | ok | 0.7778 | 0.6364 | 0.6364 | 0.4824 | 0.2000 | 4 | 79 | - |
| cloud | experiment_2 | sigmoid | New 2 - user_formula_1 | ok | 0.7778 | 0.6364 | 0.6364 | 0.4824 | 0.2000 | 4 | 79 | - |
| cloud | experiment_2 | sigmoid | New 3 - user_formula_2 | ok | 0.7778 | 0.6818 | 0.6818 | 0.5111 | 0.2222 | 2 | 156 | - |
| cloud | experiment_2 | rbf | Author original BSVM | ok | 0.7222 | 0.5909 | 0.5909 | 0.5087 | 0.3077 | 2 | 75 | - |
| cloud | experiment_2 | rbf | New 1 - robust_hybrid | ok | 0.7778 | 0.5000 | 0.5000 | 0.3333 | 0.0000 | 28 | 47 | - |
| cloud | experiment_2 | rbf | New 2 - user_formula_1 | ok | 0.7778 | 0.5000 | 0.5000 | 0.3333 | 0.0000 | 29 | 43 | - |
| cloud | experiment_2 | rbf | New 3 - user_formula_2 | ok | 0.7778 | 0.5000 | 0.5000 | 0.3333 | 0.0000 | 29 | 39 | - |
| dataset | multiclass | linear | Author original BSVM | ok | 0.7125 | 0.7450 | 0.7450 | 0.7367 | - | 762 | 1263 | - |
| dataset | multiclass | linear | New 1 - robust_hybrid | ok | 0.7063 | 0.7775 | 0.7775 | 0.7672 | - | 724 | 986 | - |
| dataset | multiclass | linear | New 2 - user_formula_1 | ok | 0.7031 | 0.7800 | 0.7800 | 0.7692 | - | 727 | 928 | - |
| dataset | multiclass | linear | New 3 - user_formula_2 | ok | 0.7094 | 0.7575 | 0.7575 | 0.7466 | - | 712 | 1376 | - |
| dataset | multiclass | poly | Author original BSVM | ok | 0.7594 | 0.7725 | 0.7725 | 0.7700 | - | 1850 | 3385 |  |
| dataset | multiclass | poly | New 1 - robust_hybrid | ok | 0.7688 | 0.7800 | 0.7800 | 0.7763 | - | 1872 | 791 |  |
| dataset | multiclass | poly | New 2 - user_formula_1 | ok | 0.7500 | 0.7675 | 0.7675 | 0.7635 | - | 1453 | 655 |  |
| dataset | multiclass | poly | New 3 - user_formula_2 | ok | 0.7531 | 0.7700 | 0.7700 | 0.7663 | - | 1908 | 582 |  |
| dataset | multiclass | sigmoid | Author original BSVM | ok | 0.6844 | 0.7625 | 0.7625 | 0.7506 | - | 782 | 4455 |  |
