# Author BSVM vs three new c_i functions

## Aggregate deltas versus author_original

| variant | algorithm | comparisons | objective_mean_delta | objective_wins | objective_ties | objective_losses | support_vectors_mean_delta | fewer_support_vectors | model_fits_mean_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| robust_hybrid | New 1 - robust_hybrid | 1 | 0.1000 | 1 | 0 | 0 | -3.0000 | 1 | -1.0000 |
| user_formula_1 | New 2 - user_formula_1 | 1 | 0.1000 | 1 | 0 | 0 | -5.0000 | 1 | -12.0000 |
| user_formula_2 | New 3 - user_formula_2 | 1 | 0.0000 | 0 | 1 | 0 | -6.0000 | 1 | -10.0000 |

## Per-combination deltas

| dataset | paper_group | kernel | variant | algorithm | objective_test_delta | accuracy_test_delta | balanced_accuracy_test_delta | f1_macro_test_delta | minority_f1_test_delta | support_vectors_delta | model_fits_delta | elapsed_tuning_seconds_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fruitfly | experiment_1 | linear | robust_hybrid | New 1 - robust_hybrid | 0.1000 | 0.0000 | 0.0286 | 0.0143 | 0.1000 | -3 | -1 | 2.3003 |
| fruitfly | experiment_1 | linear | user_formula_1 | New 2 - user_formula_1 | 0.1000 | 0.0000 | 0.0286 | 0.0143 | 0.1000 | -5 | -12 | 0.0998 |
| fruitfly | experiment_1 | linear | user_formula_2 | New 3 - user_formula_2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -6 | -10 | 0.0168 |

## Full results (compact view)

| dataset | paper_group | kernel | algorithm | status | validation_score | objective_test | accuracy_test | f1_macro_test | minority_f1_test | support_vectors | model_fits | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fruitfly | experiment_1 | linear | Author original BSVM | ok | 0.4444 | 0.4000 | 0.5000 | 0.4857 | 0.4000 | 11 | 35 |  |
| fruitfly | experiment_1 | linear | New 1 - robust_hybrid | ok | 0.6000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 8 | 34 |  |
| fruitfly | experiment_1 | linear | New 2 - user_formula_1 | ok | 0.6000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 6 | 23 |  |
| fruitfly | experiment_1 | linear | New 3 - user_formula_2 | ok | 0.2857 | 0.4000 | 0.5000 | 0.4857 | 0.4000 | 5 | 25 |  |
