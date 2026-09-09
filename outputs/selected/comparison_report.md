# Author BSVM vs three new c_i functions

## Aggregate deltas versus author_original

| variant | algorithm | comparisons | objective_mean_delta | objective_wins | objective_ties | objective_losses | support_vectors_mean_delta | fewer_support_vectors | model_fits_mean_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| robust_hybrid | New 1 - robust_hybrid | 6 | -0.0349 | 2 | 1 | 3 | 7.3333 | 1 | 10.0000 |
| user_formula_1 | New 2 - user_formula_1 | 6 | -0.0853 | 1 | 0 | 5 | 3.3333 | 2 | 18.1667 |
| user_formula_2 | New 3 - user_formula_2 | 6 | -0.0712 | 1 | 0 | 5 | 3.8333 | 2 | 17.1667 |

## Per-combination deltas

| dataset | paper_group | kernel | variant | algorithm | objective_test_delta | accuracy_test_delta | balanced_accuracy_test_delta | f1_macro_test_delta | minority_f1_test_delta | support_vectors_delta | model_fits_delta | elapsed_tuning_seconds_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | experiment_2 | linear | robust_hybrid | New 1 - robust_hybrid | 0.0000 | 0.0000 | 0.0381 | 0.1057 | 0.2222 | 3 | 24 | 0.0966 |
| cloud | experiment_2 | linear | user_formula_1 | New 2 - user_formula_1 | -0.0455 | -0.0455 | 0.0048 | 0.0769 | 0.2000 | 2 | 28 | 0.0767 |
| cloud | experiment_2 | linear | user_formula_2 | New 3 - user_formula_2 | -0.0455 | -0.0455 | 0.0048 | 0.0769 | 0.2000 | 2 | 28 | 0.1862 |
| cloud | experiment_2 | rbf | robust_hybrid | New 1 - robust_hybrid | -0.1818 | -0.1818 | -0.1333 | -0.0721 | 0.0000 | 21 | 10 | 0.0277 |
| cloud | experiment_2 | rbf | user_formula_1 | New 2 - user_formula_1 | -0.1818 | -0.1818 | -0.1333 | -0.0721 | 0.0000 | 22 | 6 | 0.0381 |
| cloud | experiment_2 | rbf | user_formula_2 | New 3 - user_formula_2 | -0.1818 | -0.1818 | -0.1333 | -0.0721 | 0.0000 | 22 | 2 | 0.1248 |
| fruitfly | experiment_1 | linear | robust_hybrid | New 1 - robust_hybrid | 0.0694 | 0.0000 | 0.0167 | 0.0255 | 0.0694 | 1 | 7 | 0.1264 |
| fruitfly | experiment_1 | linear | user_formula_1 | New 2 - user_formula_1 | 0.1250 | 0.0000 | 0.0333 | 0.0429 | 0.1250 | -3 | 1 | 0.0948 |
| fruitfly | experiment_1 | linear | user_formula_2 | New 3 - user_formula_2 | 0.2250 | 0.0800 | 0.1167 | 0.1262 | 0.2250 | -2 | 1 | 0.0412 |
| fruitfly | experiment_1 | rbf | robust_hybrid | New 1 - robust_hybrid | 0.1030 | 0.2400 | 0.1833 | 0.2587 | 0.1030 | 18 | 0 | -0.2606 |
| fruitfly | experiment_1 | rbf | user_formula_1 | New 2 - user_formula_1 | -0.1697 | 0.0000 | -0.0667 | 0.0152 | -0.1697 | 6 | 70 | -0.3170 |
| fruitfly | experiment_1 | rbf | user_formula_2 | New 3 - user_formula_2 | -0.1048 | -0.0800 | -0.1000 | -0.0660 | -0.1048 | 8 | 68 | -0.3459 |
| fruitfly | experiment_2 | linear | robust_hybrid | New 1 - robust_hybrid | -0.0800 | -0.0800 | -0.0833 | -0.1040 | -0.1538 | -1 | 48 | 0.1147 |
| fruitfly | experiment_2 | linear | user_formula_1 | New 2 - user_formula_1 | -0.0800 | -0.0800 | -0.0833 | -0.1040 | -0.1538 | 0 | 26 | 0.0531 |
| fruitfly | experiment_2 | linear | user_formula_2 | New 3 - user_formula_2 | -0.0800 | -0.0800 | -0.0833 | -0.1040 | -0.1538 | 0 | 26 | 0.0100 |
| fruitfly | experiment_2 | rbf | robust_hybrid | New 1 - robust_hybrid | -0.1200 | -0.1200 | 0.0167 | 0.0728 | 0.5000 | 2 | -29 | -0.4521 |
| fruitfly | experiment_2 | rbf | user_formula_1 | New 2 - user_formula_1 | -0.1600 | -0.1600 | -0.0500 | 0.0410 | 0.4000 | -7 | -22 | -0.4779 |
| fruitfly | experiment_2 | rbf | user_formula_2 | New 3 - user_formula_2 | -0.2400 | -0.2400 | -0.0833 | -0.0805 | 0.4516 | -7 | -22 | -0.4580 |

## Full results (compact view)

| dataset | paper_group | kernel | algorithm | status | validation_score | objective_test | accuracy_test | f1_macro_test | minority_f1_test | support_vectors | model_fits | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | experiment_2 | linear | Author original BSVM | ok | 0.7222 | 0.6818 | 0.6818 | 0.4054 | 0.0000 | 6 | 37 | - |
| cloud | experiment_2 | linear | New 1 - robust_hybrid | ok | 0.7222 | 0.6818 | 0.6818 | 0.5111 | 0.2222 | 9 | 61 | - |
| cloud | experiment_2 | linear | New 2 - user_formula_1 | ok | 0.7222 | 0.6364 | 0.6364 | 0.4824 | 0.2000 | 8 | 65 | - |
| cloud | experiment_2 | linear | New 3 - user_formula_2 | ok | 0.7222 | 0.6364 | 0.6364 | 0.4824 | 0.2000 | 8 | 65 | - |
| cloud | experiment_2 | rbf | Author original BSVM | ok | 0.7222 | 0.6818 | 0.6818 | 0.4054 | 0.0000 | 7 | 37 | - |
| cloud | experiment_2 | rbf | New 1 - robust_hybrid | ok | 0.7778 | 0.5000 | 0.5000 | 0.3333 | 0.0000 | 28 | 47 | - |
| cloud | experiment_2 | rbf | New 2 - user_formula_1 | ok | 0.7778 | 0.5000 | 0.5000 | 0.3333 | 0.0000 | 29 | 43 | - |
| cloud | experiment_2 | rbf | New 3 - user_formula_2 | ok | 0.7778 | 0.5000 | 0.5000 | 0.3333 | 0.0000 | 29 | 39 | - |
| fruitfly | experiment_1 | linear | Author original BSVM | ok | 0.1667 | 0.3750 | 0.6000 | 0.5404 | 0.3750 | 11 | 88 | - |
| fruitfly | experiment_1 | linear | New 1 - robust_hybrid | ok | 0.4615 | 0.4444 | 0.6000 | 0.5660 | 0.4444 | 12 | 95 | - |
| fruitfly | experiment_1 | linear | New 2 - user_formula_1 | ok | 0.2857 | 0.5000 | 0.6000 | 0.5833 | 0.5000 | 8 | 89 | - |
| fruitfly | experiment_1 | linear | New 3 - user_formula_2 | ok | 0.1818 | 0.6000 | 0.6800 | 0.6667 | 0.6000 | 9 | 89 | - |
| fruitfly | experiment_1 | rbf | Author original BSVM | ok | 0.5000 | 0.5333 | 0.4400 | 0.4167 | 0.5333 | 2 | 97 | - |
| fruitfly | experiment_1 | rbf | New 1 - robust_hybrid | ok | 0.5882 | 0.6364 | 0.6800 | 0.6753 | 0.6364 | 20 | 97 | - |
| fruitfly | experiment_1 | rbf | New 2 - user_formula_1 | ok | 0.6667 | 0.3636 | 0.4400 | 0.4318 | 0.3636 | 8 | 167 | - |
| fruitfly | experiment_1 | rbf | New 3 - user_formula_2 | ok | 0.6250 | 0.4286 | 0.3600 | 0.3506 | 0.4286 | 10 | 165 | - |
| fruitfly | experiment_2 | linear | Author original BSVM | ok | 0.6000 | 0.6400 | 0.6400 | 0.5322 | 0.3077 | 9 | 49 | - |
| fruitfly | experiment_2 | linear | New 1 - robust_hybrid | ok | 0.5500 | 0.5600 | 0.5600 | 0.4283 | 0.1538 | 8 | 97 | - |
| fruitfly | experiment_2 | linear | New 2 - user_formula_1 | ok | 0.5500 | 0.5600 | 0.5600 | 0.4283 | 0.1538 | 9 | 75 | - |
| fruitfly | experiment_2 | linear | New 3 - user_formula_2 | ok | 0.6000 | 0.5600 | 0.5600 | 0.4283 | 0.1538 | 9 | 75 | - |
| fruitfly | experiment_2 | rbf | Author original BSVM | ok | 0.6000 | 0.5600 | 0.5600 | 0.3590 | 0.0000 | 13 | 191 | - |
| fruitfly | experiment_2 | rbf | New 1 - robust_hybrid | ok | 0.6500 | 0.4400 | 0.4400 | 0.4318 | 0.5000 | 15 | 162 | - |
| fruitfly | experiment_2 | rbf | New 2 - user_formula_1 | ok | 0.6000 | 0.4000 | 0.4000 | 0.4000 | 0.4000 | 6 | 169 | - |
| fruitfly | experiment_2 | rbf | New 3 - user_formula_2 | ok | 0.7000 | 0.3200 | 0.3200 | 0.2784 | 0.4516 | 6 | 169 | - |
