# Công thức so sánh trên mã BSVM của tác giả

Tài liệu này là đặc tả toán học của bốn dòng trong bảng so sánh. Mọi hàm đều
trả về **độ ưu tiên**: giá trị càng lớn thì candidate càng được thử sớm.

## 1. Ký hiệu chung

Với candidate $(x_i,y_i)$, trong bài toán nhị phân $y_i\in\{-1,+1\}$:

\[
f(x_i)=\sum_{s\in SV}\lambda_s y_s K(x_s,x_i)+b,
\qquad
m_i=y_i f(x_i).
\]

- $f(x_i)$: decision value của SVM hiện tại.
- $m_i$: signed margin. $m_i=1$ nằm trên đường biên margin; $m_i=0$
  nằm trên decision boundary; $m_i<0$ bị phân lớp sai.
- $\alpha_y=n/(K n_y)$: class weight của công thức (37), với $K$ lớp và
  $n_y$ mẫu của lớp $y$. Code chuẩn hóa
  $\widetilde\alpha_y=\alpha_y/\min_c\alpha_c$; tỷ lệ giữa các lớp không đổi.
- $\varepsilon>0$: hằng số ổn định số, mặc định $10^{-8}$.

### Label reliability $r_i$

\[
r_i=\frac{1}{k}\sum_{x_\ell\in N_k^{train}(x_i)}
\mathbf 1(y_\ell=y_i).
\]

Láng giềng là các điểm train gần nhất, không tính chính $x_i$. $r_i$ cao
khi nhãn candidate phù hợp với vùng train xung quanh; điểm nghi nhiễu nhãn có
$r_i$ thấp.

### Local density $\rho_i$

Với $D_i$ là khoảng cách trung bình từ $x_i$ đến $k$ láng giềng cùng lớp
và $s_y=\operatorname{median}_{j:y_j=y}D_j$:

\[
\rho_i=\frac{1}{1+D_i/(s_{y_i}+\varepsilon)}.
\]

Điểm nằm trong cụm có $\rho_i$ cao; outlier cô lập có $\rho_i$ thấp.

### Toán tử $R[\cdot]$

Vì tích nhiều lũy thừa có thể rất nhỏ, code tính trong log-space rồi dùng
percentile rank:

\[
R(z_i)=\frac{\operatorname{rank}_{ascending}(z_i)}{|H|}.
\]

$R$ đơn điệu nên giữ nguyên thứ tự candidate, đồng thời tránh underflow và
làm kết quả ổn định hơn giữa các thang đo.

## 2. Baseline: `author_original`

Thuật toán 4 trong bài báo và mã GitHub sắp giảm dần theo nghịch đảo khoảng
cách tới decision boundary:

\[
c_i^{author}=\frac{\widetilde\alpha_{y_i}}
{|f(x_i)|+\varepsilon}.
\]

Candidate được thử từng điểm (`sequential`). Có một khác biệt trong bài báo:
công thức (13) in $|f(x_i)|/|w_{y_i}|$, nhưng dòng 9 và 13 của Thuật toán 4
dùng $w_{y_i}/|f(x_i)|$ rồi sắp giảm dần. Code dùng phiên bản của Thuật toán
4 vì phù hợp với diễn giải "điểm gần boundary và lớp thiểu số được ưu tiên".

## 3. Hàm mới 1: `robust_hybrid`

Đây là hàm của phiên bản Binary-Tree BSVM đã xây dựng trước đó:

\[
c_i^{hybrid}=
\widetilde\alpha_{y_i}^{p}
\exp\!\left(-\beta\frac{|f(x_i)|}{T}\right)
r_i^{\gamma}\rho_i^{\delta}.
\]

Nó ưu tiên điểm gần **decision boundary** $f=0$, đồng thời giảm ưu tiên
candidate có nhãn cục bộ kém tin cậy hoặc có mật độ thấp. $T$ mặc định là
median của $|f(x_i)|$, có sàn 0.25.

## 4. Hàm mới 2: `user_formula_1`

Theo công thức người dùng cung cấp:

\[
c_i^{(1)}=R\!\left[
\widetilde\alpha_{y_i}^{p}
r_i^{\beta}\rho_i^{\gamma}
\exp\!\left(-\frac{|1-m_i|}{\tau}\right)
\right].
\]

Khác `robust_hybrid`, số hạng mũ đạt cực đại tại $m_i=1$, tức là ưu tiên
điểm gần **đường margin đơn vị**, không phải điểm gần decision boundary.

Nếu không truyền `--tau`, code dùng:

\[
\tau=\max\{\operatorname{median}_i|1-m_i|,\ 0.25,\ \varepsilon\}.
\]

## 5. Hàm mới 3: `user_formula_2`

Đặt:

\[
u_i=\exp\!\left(-\frac{|1-m_i|}{\tau}\right)
\]

và validation-gain proxy:

\[
g_i=\frac{1}{k}\sum_{x_j\in N_k^{Val}(x_i)}
\mathbf 1(y_j=y_i).
\]

$N_k^{Val}(x_i)$ là $k$ điểm validation gần $x_i$ nhất. Khi candidate
nằm gần nhiều validation sample cùng lớp, $g_i$ cao; candidate là outlier
hoặc gần vùng validation trái nhãn có $g_i$ thấp.

Công thức đầy đủ:

\[
c_i^{(2)}=R\!\left[
\widetilde\alpha_{y_i}^{p}
r_i^{\beta}\rho_i^{\gamma}
u_i g_i^{\delta}
\right].
\]

Code chặn $g_i$ ở $\varepsilon$ khi lấy log; giá trị thành phần được báo
cáo vẫn là $g_i$ gốc trong đoạn $[0,1]$.

### Ranh giới dữ liệu

`user_formula_2` được phép đọc feature và label của **validation**, đúng định
nghĩa $g_i$. Test set không được truyền vào hàm ưu tiên. Vì vậy đây là mô
hình validation-guided/transductive ở giai đoạn chọn candidate; khi báo cáo
kết quả phải nêu rõ nó dùng thêm thông tin validation so với ba dòng còn lại.

## 6. Các số mũ

| Tham số CLI | Vai trò |
|---|---|
| `--p` | mức ưu tiên lớp thiểu số qua $\alpha_{y_i}$ |
| `--beta` | số mũ $r_i$ ở hai công thức người dùng; độ mạnh boundary ở `robust_hybrid` |
| `--gamma-power` | số mũ $\rho_i$ ở hai công thức người dùng; số mũ $r_i$ ở `robust_hybrid` |
| `--delta` | số mũ $g_i$ ở hàm 2; số mũ $\rho_i$ ở `robust_hybrid` |
| `--tau` | temperature; 0 nghĩa là tự ước lượng robust |
| `--n-neighbors` | $k$ dùng cho $r_i,\rho_i,g_i$ |

Giá trị mặc định của các số mũ là 1. Không nên chọn chúng trên test set.

## 7. Thêm candidate theo cây nhị phân

Ba hàm mới dùng cùng một cơ chế, để khác biệt chính nằm ở thứ tự $c_i$:

```text
ordered = sort(candidates, c_i, descending=True)
try(best half)
try(remaining half)

try(block):
    fit SVM on core + block
    if every point is classified correctly:
        accept the whole block
    elif block has one point:
        reject that point
    else:
        try(first half)
        try(second half)
```

Baseline tác giả vẫn thử tuần tự. Sau khi boundary thay đổi, code tính lại
priority từ candidate còn lại như `masterproblem`/`extend_samples` của tác giả.

## 8. Chọn mô hình theo performance và số support vector

Chế độ đúng protocol bài báo (`--selection-objective paper`) tối ưu
minority-F1 ở experiment 1 và accuracy ở experiment 2; nếu bằng điểm, mô hình
ít support vector hơn thắng.

Chế độ mở rộng:

\[
J=Score_{validation}-\lambda\frac{\#SV}{n_{train}}
\]

được bật bằng `--selection-objective performance_sv --sv-penalty 0.05`.
Tăng $\lambda$ nếu muốn phạt kích thước mô hình mạnh hơn.
