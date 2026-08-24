# BSVM của tác giả và ba hàm ưu tiên `c_i` mới

Gói này dùng snapshot mã GitHub chính thức của Mohasel và Koosha, mô-đun hóa
ba hàm `Initialsolution -> masterproblem -> extend_samples`, rồi thêm:

1. Hàm `robust_hybrid` của phiên bản Binary-Tree BSVM trước đó.
2. Công thức `user_formula_1` ưu tiên signed margin gần 1.
3. Công thức `user_formula_2` bổ sung validation-gain proxy $g_i$.
4. Cây nhị phân thử nửa candidate tốt nhất rồi chia đôi khối không khả thi.
5. Runner dùng local CSV, mọi kernel, split và grid search của bài báo, có
   `--resume` và xuất bảng so sánh sau từng tổ hợp.

Nguồn khoa học:

- Bài báo: *A robust and lightweight support vector machine for imbalanced
  and noisy data via Benders decomposition*, Neurocomputing 671 (2026),
  132629, <https://doi.org/10.1016/j.neucom.2026.132629>.
- Mã tác giả: <https://github.com/MojtabaMohasel/BSVM.git>, commit được ghim
  `ee5a7ae7ade4977b41d604dd989b31f4e356d342`.

Đặc tả đầy đủ từng ký hiệu, số mũ và ranh giới validation/test nằm trong
[`docs/AUTHOR_CI_FORMULAS.md`](docs/AUTHOR_CI_FORMULAS.md).

## Bốn dòng được so sánh

| `variant` | Hàm ưu tiên | Cách thêm candidate |
|---|---|---|
| `author_original` | $\widetilde\alpha_y/(|f(x_i)|+\varepsilon)$ theo Thuật toán 4 | tuần tự, giống mã tác giả |
| `robust_hybrid` | class weight x gần boundary x label reliability x density | cây nhị phân |
| `user_formula_1` | $R[\widetilde\alpha_y^p r_i^\beta\rho_i^\gamma e^{-|1-m_i|/\tau}]$ | cây nhị phân |
| `user_formula_2` | hàm 1 nhân $g_i^\delta$, với $g_i$ tính từ validation k-NN | cây nhị phân |

Lưu ý: công thức (13) trong bài báo in $|f|/|w_y|$, nhưng dòng 9 và 13 của
Thuật toán 4 dùng $w_y/|f|$ rồi sắp giảm dần. Baseline dùng phiên bản của
Thuật toán 4 và mã GitHub, vì nó đúng với diễn giải "ưu tiên điểm gần boundary".

## 1. Cài đặt từ đầu

Yêu cầu Python 3.10 trở lên. Trong PowerShell, đứng ở thư mục đã giải nén:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,dev]"
python -m pytest
```

Linux/macOS dùng:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,dev]"
python -m pytest
```

## 2. Chuẩn hóa dataset

ZIP đầy đủ đã có `data/csv/*.csv`. Nếu muốn tạo lại từ ARFF/TXT/XLSX đã tải
vào `data/`:

```powershell
python examples/convert_data_to_csv.py
```

Mỗi CSV phải có cột nhãn `target`. Dataset không nhận đúng target được khai
báo trong `data/target_overrides.csv`.

Workspace hiện có 17 dataset CSV. So với Table 2, `credit-approval` (OpenML
29) chưa có trong thư mục local; runner chỉ chạy những file thực sự tồn tại.

## 3. Smoke test đủ bốn biến thể

Lệnh nhanh nhất để xác nhận môi trường, công thức $g_i$, cây nhị phân và bảng
delta đều hoạt động:

```powershell
python examples/run_author_ci_comparison.py `
  --datasets fruitfly `
  --kernels linear `
  --paper-groups experiment_1 `
  --search none `
  --max-rows 60 `
  --output-dir outputs\author_ci_smoke
```

## 4. Chạy giống protocol bài báo

Lệnh sau chạy mọi CSV hiện có, bốn kernel, đủ bốn biến thể và grid ở Table 3:

```powershell
python examples/run_author_ci_comparison.py `
  --search paper `
  --paper-groups all `
  --resume
```

Protocol được giữ:

- 80% train+validation, 20% holdout test, stratified, `random_state=42`.
- Trong 80% đầu: 80% train và 20% validation; tương đương 64/16/20 toàn bộ.
- Experiment 1: class weight cân bằng, chọn theo minority-F1.
- Experiment 2: equal class weights, chọn theo accuracy.
- Grid Table 3: $C=[0.1,1,10,100]$, degree `[2,3,4,5]`, coef0
  `[0,0.5,1]`, gamma `[0.001,0.01,0.1,1]` theo kernel phù hợp.
- Mô hình có validation score tốt nhất được đánh giá trên test; test không
  tham gia tạo $c_i$, chọn tham số hay sửa initial core.
- `--paper-groups all` chạy `fruitfly` và `tecator` ở cả experiment 1 và 2,
  vì Table 2 dùng lại hai dataset này với hai mục tiêu khác nhau.

Full grid có thể rất lâu, đặc biệt với `poly`, `sigmoid`, Fashion-MNIST và OVR.
Mỗi tổ hợp được ghi ngay xuống đĩa; dừng rồi chạy lại cùng lệnh `--resume`.

## 5. Tối ưu performance và ít support vector

Protocol bài báo chọn theo F1/accuracy và chỉ dùng số support vector để phá hòa.
Để bật mục tiêu mở rộng

\[
J=Score_{validation}-\lambda\frac{\#SV}{n_{train}},
\]

chạy:

```powershell
python examples/run_author_ci_comparison.py `
  --search paper `
  --selection-objective performance_sv `
  --sv-penalty 0.05 `
  --output-dir outputs\author_ci_performance_sv `
  --resume
```

`--sv-penalty` càng lớn thì mô hình càng ưu tiên ít support vector. Không chọn
giá trị này trên test set.

## 6. Chỉ chạy một số data, kernel hoặc hàm

```powershell
python examples/run_author_ci_comparison.py `
  --datasets fruitfly,cloud,leukemia `
  --kernels linear,rbf `
  --variants author_original,user_formula_1,user_formula_2 `
  --search fast `
  --paper-groups auto `
  --output-dir outputs\selected `
  --resume
```

Bạn cũng có thể sửa trực tiếp các hằng `DEFAULT_KERNELS`, `VARIANT_LABELS` và
`DATASET_PROTOCOLS` ở đầu `examples/run_author_ci_comparison.py`.

Các tham số công thức:

```text
--p 1.0
--beta 1.0
--gamma-power 1.0
--delta 1.0
--tau 0          # 0 = tự ước lượng robust
--n-neighbors 7
```

## 7. Chạy bản demo đã sửa trực tiếp từ mã tác giả

Các file upstream nguyên gốc nằm trong `third_party/BSVM_author/`. File thêm
`BSVM_extended.py` chạy lại toy 2-D của tác giả nhưng cho phép chọn kernel và
hàm mới:

```powershell
python third_party\BSVM_author\BSVM_extended.py `
  --kernel rbf `
  --variant user_formula_2
```

Đối chiếu commit và thay đổi xem `third_party/BSVM_author/UPSTREAM.md`.
Ánh xạ từng hàm upstream sang extension nằm ở
`docs/AUTHOR_CODE_MAPPING.md`.

## 8. Notebook

```powershell
jupyter notebook notebooks\02_author_ci_comparison.ipynb
```

Notebook có cell chỉnh dataset/kernel/search, gọi runner và đọc bốn bảng đầu
ra. Chạy script CLI thuận tiện hơn cho full grid và `--resume`.

## 9. File kết quả

Mặc định ghi vào `outputs/author_ci_comparison/`:

| File | Nội dung |
|---|---|
| `comparison_all.csv` | train/validation/test metrics, best hyperparameters, SV, model fits |
| `tuning_results.csv` | từng cấu hình grid, lỗi cấu hình và selection utility |
| `comparison_deltas_vs_original.csv` | mỗi hàm mới trừ `author_original` trên cùng dataset/group/kernel |
| `summary_by_variant.csv` | mean delta, win/tie/loss, số lần giảm support vector |
| `comparison_report.md` | bản Markdown gọn để đưa vào báo cáo |
| `run_config.json` | toàn bộ tham số để tái lập lần chạy |
| `author_ci_comparison_smoke.xlsx` | workbook mẫu đi kèm ZIP; có Results, delta bằng công thức, Summary và Protocol |

Trong bảng delta:

- metric delta dương: hàm mới tốt hơn baseline.
- `support_vectors_delta < 0`: hàm mới dùng ít support vector hơn.
- `model_fits_delta < 0`: hàm mới fit ít subproblem hơn.
- Không kết luận từ smoke output; cần full grid và toàn bộ dataset đã khóa.

## 10. Xử lý lỗi kernel

Mã tác giả giả định initial core luôn chứa hai lớp và tách được. Với sigmoid
hoặc một số cấu hình poly, giả định này có thể sai. Mặc định extension thu nhỏ
core theo signed margin, bảo toàn hai lớp và ghi `initial_core_repairs` vào bảng.

Để kiểm tra nghiêm ngặt, tắt sửa tự động:

```powershell
python examples/run_author_ci_comparison.py --no-core-repair --resume
```

Lỗi của một grid point được ghi vào `tuning_results.csv`; các grid point khác
vẫn chạy. Một tổ hợp chỉ có `status=error` khi toàn bộ grid đều thất bại.

## 11. Cấu trúc chính

```text
src/roch_bsvm/author_bsvm.py             mô-đun hóa control flow mã tác giả
src/roch_bsvm/scoring.py                 bốn họ priority và các thành phần
examples/run_author_ci_comparison.py     runner paper protocol / all data / all kernel
third_party/BSVM_author/                 snapshot GitHub và demo đã sửa
docs/AUTHOR_CI_FORMULAS.md               đặc tả toán học
notebooks/02_author_ci_comparison.ipynb  notebook điều khiển runner
outputs/author_ci_comparison_smoke/      output kiểm thử mẫu
tests/                                   kiểm thử công thức, validation gain và model
```
