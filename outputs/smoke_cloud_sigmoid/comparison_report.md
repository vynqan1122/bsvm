# Author BSVM vs three new c_i functions

## Aggregate deltas versus author_original

| variant | algorithm | comparisons | objective_mean_delta | objective_wins | objective_ties | objective_losses | support_vectors_mean_delta | fewer_support_vectors | model_fits_mean_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| robust_hybrid | New 1 - robust_hybrid | 1 | 0.0000 | 0 | 1 | 0 | 0.0000 | 0 | 44.0000 |
| user_formula_1 | New 2 - user_formula_1 | 1 | 0.0000 | 0 | 1 | 0 | 0.0000 | 0 | 44.0000 |
| user_formula_2 | New 3 - user_formula_2 | 1 | 0.0000 | 0 | 1 | 0 | 0.0000 | 0 | 44.0000 |

## Per-combination deltas

| dataset | paper_group | kernel | variant | algorithm | objective_test_delta | accuracy_test_delta | balanced_accuracy_test_delta | f1_macro_test_delta | minority_f1_test_delta | support_vectors_delta | model_fits_delta | elapsed_tuning_seconds_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | experiment_2 | sigmoid | robust_hybrid | New 1 - robust_hybrid | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 44 | 0.2605 |
| cloud | experiment_2 | sigmoid | user_formula_1 | New 2 - user_formula_1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 44 | 0.1587 |
| cloud | experiment_2 | sigmoid | user_formula_2 | New 3 - user_formula_2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 44 | 0.0671 |

## Full results (compact view)

| dataset | paper_group | kernel | algorithm | status | validation_score | objective_test | accuracy_test | f1_macro_test | minority_f1_test | support_vectors | model_fits | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | experiment_2 | sigmoid | Author original BSVM | ok | 0.3333 | 0.3636 | 0.3636 | 0.3583 | 0.4167 | 2 | 84 |  |
| cloud | experiment_2 | sigmoid | New 1 - robust_hybrid | ok | 0.3333 | 0.3636 | 0.3636 | 0.3583 | 0.4167 | 2 | 128 |  |
| cloud | experiment_2 | sigmoid | New 2 - user_formula_1 | ok | 0.3333 | 0.3636 | 0.3636 | 0.3583 | 0.4167 | 2 | 128 |  |
| cloud | experiment_2 | sigmoid | New 3 - user_formula_2 | ok | 0.3333 | 0.3636 | 0.3636 | 0.3583 | 0.4167 | 2 | 128 |  |
