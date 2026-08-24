# Đặc tả toán học của bốn hàm ưu tiên

Tài liệu mô tả các công thức trong src/roch_bsvm/scoring.py. Mọi hàm trả về
độ ưu tiên; giá trị lớn hơn được xếp trước.

## 1. Ký hiệu chung

$$
f(x_i)
=
\sum_{s\in SV}\lambda_s y_s K(x_s,x_i)+b
$$

$$
m_i=y_i f(x_i)
$$

$f(x_i)$ là decision value. $m_i$ là signed margin: $m_i=1$ ở đường margin,
$m_i=0$ ở decision boundary, và $m_i<0$ đang bị phân lớp sai.

$$
\alpha_y=\frac{n}{K n_y}
$$

$$
\widetilde{\alpha}_y
=
\frac{\alpha_y}{\min_c\alpha_c}
$$

$n$ là số mẫu train, $K$ là số lớp, $n_y$ là số mẫu lớp y.

$$
r_i
=
\frac{1}{k}
\sum_{x_\ell\in N_k^{train}(x_i)}
\mathbf{1}(y_\ell=y_i)
$$

$r_i$ là label reliability. Candidate không được tính là láng giềng của
chính nó.

$$
D_i
=
\frac{1}{k}
\sum_{x_\ell\in N_{k,y_i}^{train}(x_i)}
\lVert x_i-x_\ell\rVert_2
$$

$$
s_y
=
\operatorname{median}_{j:y_j=y}D_j
$$

$$
\rho_i
=
\frac{1}{1+D_i/(s_{y_i}+\varepsilon)}
$$

$\rho_i$ là local density; giá trị thấp biểu thị outlier cô lập.

$$
R(z_i)
=
\frac{\operatorname{rank}_{ascending}(z_i)}
{|H|}
$$

$H$ là pool candidate. Code tính trong log-space rồi rank để tránh underflow.

## 2. author_original

$$
c_i^{author}
=
\frac{\widetilde{\alpha}_{y_i}}
{|f(x_i)|+\varepsilon}
$$

Baseline thử tuần tự. Công thức (13) in tỷ số ngược lại, nhưng Thuật toán 4
dùng công thức trên rồi sắp giảm dần.

## 3. robust_hybrid

$$
c_i^{hybrid}
=
\widetilde{\alpha}_{y_i}^{p}
\exp\left(-\beta\frac{|f(x_i)|}{T}\right)
r_i^{\gamma}
\rho_i^{\delta}
$$

$T$ tự động là median $|f(x_i)|$ với sàn 0.25.

## 4. user_formula_1

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

Nếu tau bằng 0:

$$
\tau
=
\max\left\{
\operatorname{median}_i|1-m_i|,
0.25,
\varepsilon
\right\}
$$

## 5. user_formula_2

$$
g_i
=
\frac{1}{k}
\sum_{x_j\in N_k^{Val}(x_i)}
\mathbf{1}(y_j=y_i)
$$

$$
u_i
=
\exp\left(-\frac{|1-m_i|}{\tau}\right)
$$

$$
c_i^{(2)}
=
R\left[
\widetilde{\alpha}_{y_i}^{p}
r_i^{\beta}
\rho_i^{\gamma}
u_i
g_i^{\delta}
\right]
$$

$g_i$ đọc feature và label validation, không đọc test. Code chỉ chặn nó dưới
bởi $\varepsilon$ khi lấy log; giá trị báo cáo vẫn trong $[0,1]$.

## 6. Tham số

| Tên CLI/code | Ý nghĩa |
|---|---|
| p | độ mạnh class weight |
| beta | số mũ reliability trong hai công thức người dùng; độ mạnh boundary trong robust_hybrid |
| gamma-power | số mũ density trong hai công thức người dùng; số mũ reliability trong robust_hybrid |
| delta | số mũ validation gain trong user_formula_2; số mũ density trong robust_hybrid |
| tau | temperature của hai công thức người dùng; 0 là tự ước lượng |
| n-neighbors | số láng giềng k |
| epsilon | hằng số ổn định số |
| sv-penalty | hệ số lambda phạt support vector |

Không chọn các tham số bằng test set.

## 7. Cây nhị phân

~~~text
ordered = sort(candidates, by=c_i, descending=True)

try(block):
    fit SVM on core + block
    if every point is classified correctly:
        accept block
    elif block has one point:
        reject point
    else:
        try(first half)
        try(second half)
~~~

Ba hàm mới dùng cây nhị phân; baseline thử tuần tự. Khi boundary đổi, priority
của pool còn lại được tính lại.

## 8. Mục tiêu mở rộng

$$
J
=
Score_{validation}
-\lambda\frac{\#SV}{n_{train}}
$$

$\lambda$ là sv-penalty, $\#SV$ là số support vector, $n_{train}$ là số mẫu
train.
