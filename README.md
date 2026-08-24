# BSVM của tác giả và ba hàm ưu tiên mới

Dự án giữ snapshot mã BSVM chính thức của Mohasel và Koosha, rồi mở rộng phần
xếp hạng và thêm candidate. Bốn biến thể được chạy trên cùng dữ liệu, split,
kernel và tiêu chí chọn mô hình:

- **author_original**: hàm của Thuật toán 4, thêm từng điểm.
- **robust_hybrid**: hàm mới thứ nhất, thêm theo cây nhị phân.
- **user_formula_1**: hàm mới thứ hai, ưu tiên signed margin gần 1.
- **user_formula_2**: hàm mới thứ ba, có validation-gain proxy.

Nguồn bài báo: <https://doi.org/10.1016/j.neucom.2026.132629>.
Nguồn code: <https://github.com/MojtabaMohasel/BSVM>, commit
**ee5a7ae7ade4977b41d604dd989b31f4e356d342**.

## Ký hiệu chung

Với candidate $(x_i,y_i)$:

$$
f(x_i)=\sum_{s\in SV}\lambda_s y_s K(x_s,x_i)+b
$$

$$
m_i=y_i f(x_i)
$$

- $x_i$: vector đặc trưng của candidate thứ i.
- $y_i$: nhãn; với bài toán nhị phân, $y_i\in\{-1,+1\}$.
- $K(x_s,x_i)$: kernel linear, RBF, polynomial hoặc sigmoid.
- $f(x_i)$: decision value của SVM hiện tại.
- $m_i$: signed margin; $m_i=1$ ở đường margin, $m_i=0$ ở decision boundary,
  và $m_i<0$ nghĩa là đang phân lớp sai.
- $SV$: tập support vector; $\lambda_s$ và $b$ là hệ số và bias của SVM.
- $\varepsilon$: hằng số ổn định số, mặc định $10^{-8}$.

## Hàm của tác giả và ba hàm mới

Mọi công thức trả về **độ ưu tiên**: giá trị càng lớn thì điểm được thử càng
sớm.

### 1. author_original — hàm của tác giả

$$
c_i^{author}
=
\frac{\widetilde{\alpha}_{y_i}}
{|f(x_i)|+\varepsilon}
$$

Hàm ưu tiên điểm gần decision boundary và lớp thiểu số. Candidate được thêm
tuần tự như control flow của mã tác giả.

Lưu ý: công thức (13) của bài báo in $|f(x_i)|/|w_{y_i}|$, nhưng dòng 9 và 13
của Thuật toán 4 dùng tỷ số nghịch đảo rồi sắp giảm dần. Baseline dùng phiên
bản của Thuật toán 4. Snapshot GitHub thực tế dùng
$1/(|f(x_i)|+10^{-6})$ và bỏ qua đối số weights; extension nhân class weight
để bám sát dòng 9.

### 2. robust_hybrid — hàm mới thứ nhất

$$
c_i^{hybrid}
=
\widetilde{\alpha}_{y_i}^{p}
\exp\left(-\beta\frac{|f(x_i)|}{T}\right)
r_i^{\gamma}
\rho_i^{\delta}
$$

Hàm ưu tiên điểm gần **decision boundary**, đồng thời giảm ưu tiên điểm có
nhãn không phù hợp với lân cận hoặc là outlier cô lập.

### 3. user_formula_1 — hàm mới thứ hai

$$
c_i^{(1)}
=
R\left[
\widetilde{\alpha}_{y_i}^{p}
r_i^{\beta}
\rho_i^{\gamma}
\exp\left(-\frac{|1-m_i|}{\tau}\right)
\right]
$$

Số hạng mũ đạt cực đại tại $m_i=1$, nên hàm ưu tiên điểm gần **đường margin
đơn vị**, không phải decision boundary.

### 4. user_formula_2 — hàm mới thứ ba

Validation-gain proxy:

$$
g_i
=
\frac{1}{k}
\sum_{x_j\in N_k^{Val}(x_i)}
\mathbf{1}(y_j=y_i)
$$

Công thức đầy đủ:

$$
c_i^{(2)}
=
R\left[
\widetilde{\alpha}_{y_i}^{p}
r_i^{\beta}
\rho_i^{\gamma}
\exp\left(-\frac{|1-y_i f(x_i)|}{\tau}\right)
g_i^{\delta}
\right]
$$

$g_i$ chỉ đọc validation, không đọc test. Candidate gần nhiều validation
sample cùng lớp có ưu tiên cao hơn.

## Giải thích đầy đủ tham số

Class weight:

$$
\alpha_y=\frac{n}{K n_y}
$$

$$
\widetilde{\alpha}_y
=
\frac{\alpha_y}{\min_c\alpha_c}
$$

- $n$: tổng số mẫu train; $K$: số lớp; $n_y$: số mẫu train của lớp y.
- $\alpha_y$: class weight, ưu tiên lớp thiểu số.
- $\widetilde{\alpha}_y$: class weight đã chuẩn hóa; không đổi tỷ lệ giữa lớp.
- $p$: số mũ điều khiển độ mạnh của class weight.

Label reliability:

$$
r_i
=
\frac{1}{k}
\sum_{x_\ell\in N_k^{train}(x_i)}
\mathbf{1}(y_\ell=y_i)
$$

$r_i$ là tỷ lệ láng giềng train cùng nhãn; candidate không tính là láng giềng
của chính nó. Giá trị thấp gợi ý nhiễu nhãn.

Local density:

$$
\rho_i
=
\frac{1}
{1+D_i/(s_{y_i}+\varepsilon)}
$$

$D_i$ là khoảng cách trung bình tới k láng giềng cùng lớp; $s_y$ là median
của các $D_j$ trong lớp y. $\rho_i$ thấp với outlier cô lập.

- $\beta$: số mũ của $r_i$ trong hai công thức người dùng; trong
  robust_hybrid, nó điều khiển mức giảm theo khoảng cách boundary.
- $\gamma$: số mũ của $\rho_i$ trong hai công thức người dùng; trong
  robust_hybrid, nó là số mũ của $r_i$.
- $\delta$: số mũ của $g_i$ trong user_formula_2; trong robust_hybrid, nó là
  số mũ của $\rho_i$.
- $k$: số láng giềng của $r_i$, $\rho_i$ và $g_i$; CLI là
  **--n-neighbors**.
- $T$: temperature của robust_hybrid; tự động lấy median $|f(x_i)|$ với sàn
  0.25.
- $\tau$: temperature của hai công thức người dùng. Khi **--tau 0**:

$$
\tau
=
\max\left\{
\operatorname{median}_i|1-m_i|,
0.25,
\varepsilon
\right\}
$$

- $R[\cdot]$: rank normalization:

$$
R(z_i)
=
\frac{\operatorname{rank}_{ascending}(z_i)}
{|H|}
$$

  $H$ là pool candidate hiện tại. Code tính score trong log-space rồi lấy
  percentile rank để tránh underflow.
- $\mathbf{1}(\cdot)$: hàm chỉ báo, bằng 1 khi điều kiện đúng, ngược lại bằng
  0.
- $N_k^{train}(x_i)$: k láng giềng gần nhất trong train.
- $N_k^{Val}(x_i)$: k láng giềng gần nhất trong validation.

Đặc tả chi tiết nằm ở
[docs/AUTHOR_CI_FORMULAS.md](docs/AUTHOR_CI_FORMULAS.md).

## Cây nhị phân thêm candidate

Ba hàm mới sắp giảm dần theo $c_i$, thử nửa tốt nhất, nhận cả khối nếu khả
thi; nếu không thì chia đôi đệ quy:

~~~text
ordered = sort(candidates, by=c_i, descending=True)

try(block):
    fit SVM on core + block
    if mọi điểm được phân lớp đúng:
        nhận toàn bộ block
    elif block chỉ có một điểm:
        loại điểm đó
    else:
        try(nửa tốt hơn)
        try(nửa còn lại)
~~~

author_original vẫn thêm tuần tự để không trộn tác động của hàm ưu tiên với
tác động của cây.

## Cài đặt và chạy

Yêu cầu Python 3.10 trở lên:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,dev]"
python -m pytest
~~~

Chuyển ARFF, CSV, TSV, TXT, DATA hoặc XLSX trong data sang CSV chuẩn:

~~~powershell
python examples/convert_data_to_csv.py
~~~

Chạy mọi local CSV, bốn kernel, bốn biến thể và grid của bài:

~~~powershell
python examples/run_author_ci_comparison.py --search paper --paper-groups all --resume
~~~

Chỉ chạy một số data/kernel:

~~~powershell
python examples/run_author_ci_comparison.py --datasets fruitfly,cloud --kernels linear,rbf --variants author_original,robust_hybrid,user_formula_1,user_formula_2 --search fast --output-dir outputs\selected --resume
~~~

Tham số có thể chỉnh:

~~~text
--p 1.0
--beta 1.0
--gamma-power 1.0
--delta 1.0
--tau 0
--n-neighbors 7
~~~

Protocol mặc định là 64% train, 16% validation, 20% holdout test,
stratified, seed 42. Experiment 1 chọn minority-F1 với balanced class weight;
Experiment 2 chọn accuracy với equal weights. Test không tham gia tính $c_i$
hay chọn tham số.

## Performance và số support vector

Chế độ mở rộng tối ưu:

$$
J
=
Score_{validation}
-\lambda\frac{\#SV}{n_{train}}
$$

$\lambda$ là mức phạt, $\#SV$ là số support vector, $n_{train}$ là số mẫu
train.

~~~powershell
python examples/run_author_ci_comparison.py --search paper --selection-objective performance_sv --sv-penalty 0.05 --resume
~~~

## Output và notebook

Runner ghi comparison_all.csv, tuning_results.csv,
comparison_deltas_vs_original.csv, summary_by_variant.csv,
comparison_report.md và run_config.json trong outputs/author_ci_comparison.

~~~powershell
jupyter notebook notebooks\02_author_ci_comparison.ipynb
~~~

Demo trên cấu trúc code tác giả:

~~~powershell
python third_party\BSVM_author\BSVM_extended.py --kernel rbf --variant user_formula_2
~~~

Xem [docs/AUTHOR_CODE_MAPPING.md](docs/AUTHOR_CODE_MAPPING.md) và
[third_party/BSVM_author/UPSTREAM.md](third_party/BSVM_author/UPSTREAM.md).

## Lưu ý

- user_formula_2 dùng nhãn validation để tính $g_i$; phải nêu rõ khi báo cáo.
- Không chọn số mũ, $\tau$, $k$ hoặc $\lambda$ trên test.
- Smoke output chỉ chứng minh code chạy, không đủ để kết luận hàm mới tốt hơn.
- Sigmoid hoặc poly có thể làm initial core không tách được. Core repair mặc
  định bảo toàn hai lớp, thu nhỏ core và ghi initial_core_repairs.
