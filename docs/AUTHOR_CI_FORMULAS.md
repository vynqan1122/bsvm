# Đặc tả toán học của bốn hàm ưu tiên

Tài liệu mô tả các công thức trong src/roch_bsvm/scoring.py. Mọi hàm trả về
độ ưu tiên; giá trị lớn hơn được xếp trước.

## 1. Ký hiệu chung

```math
f(x_i)
=
\sum_{s\in SV}\lambda_s y_s K(x_s,x_i)+b
```

```math
m_i=y_i f(x_i)
```

$f(x_i)$ là decision value. $m_i$ là signed margin: $m_i=1$ ở đường margin,
$m_i=0$ ở decision boundary, và $m_i<0$ đang bị phân lớp sai.

```math
\alpha_y=\frac{n}{K n_y}
```

```math
\widetilde{\alpha}_y
=
\frac{\alpha_y}{\min_c\alpha_c}
```

$n$ là số mẫu train, $K$ là số lớp, $n_y$ là số mẫu lớp y.

```math
r_i
=
\frac{1}{k}
\sum_{x_\ell\in N_k^{\mathrm{train}}(x_i)}
\mathbf{1}(y_\ell=y_i)
```

$r_i$ là label reliability. Candidate không được tính là láng giềng của
chính nó.

```math
D_i
=
\frac{1}{k}
\sum_{x_\ell\in N_{k,y_i}^{\mathrm{train}}(x_i)}
\lVert x_i-x_\ell\rVert_2
```

```math
s_y
=
\mathrm{median}_{j:y_j=y}D_j
```

```math
\rho_i
=
\frac{1}{1+D_i/(s_{y_i}+\varepsilon)}
```

$\rho_i$ là local density; giá trị thấp biểu thị outlier cô lập.

```math
R(z_i)
=
\frac{\mathrm{rank}_{\mathrm{ascending}}(z_i)}
{\lvert H\rvert}
```

$H$ là pool candidate. Code tính trong log-space rồi rank để tránh underflow.

## 2. author_original

```math
c_i^{\mathrm{author}}
=
\frac{\widetilde{\alpha}_{y_i}}
{\lvert f(x_i)\rvert+\varepsilon}
```

Baseline thử tuần tự. Công thức (13) in tỷ số ngược lại, nhưng Thuật toán 4
dùng công thức trên rồi sắp giảm dần.

## 3. robust_hybrid

```math
c_i^{\mathrm{hybrid}}
=
\widetilde{\alpha}_{y_i}^{p}
\exp\left(-\beta\frac{\lvert f(x_i)\rvert}{T}\right)
(a_r+(1-a_r)r_i)^{\gamma}
(a_\rho+(1-a_\rho)\rho_i)^{\delta}
```

$a_r=a_\rho=0.05$ theo code hybrid mặc định. $T$ tự động là median
$|f(x_i)|$ với sàn 0.25; có thể đặt qua `temperature` trong JSON/API hoặc
`--tau` dương của runner cũ.

## 4. user_formula_1

```math
c_i^{(1)} = R\left[\widetilde{\alpha}_{y_i}^{p} \cdot \widehat r_i^{\beta} \cdot \rho_i^{\gamma} \cdot \exp\left(-\frac{\lvert 1-m_i\rvert}{\tau}\right)\right]
```

$\widehat r_i=a+(1-a)r_i$, `local_floor=a`, mặc định 0. Profile
`conservative` đặt 0.05. Các yếu tố được clip dưới bởi epsilon khi lấy log.

Trong CLI cũ, `--tau 0` là auto; trong JSON/API phải bỏ key tau để auto:

```math
\tau
=
\max\left\{
\mathrm{median}_i\lvert 1-m_i\rvert,
0.25,
\varepsilon
\right\}
```

## 5. user_formula_2

```math
g_i
=
\frac{1}{k}
\sum_{x_j\in N_k^{\mathrm{Ref}}(x_i)}
\mathbf{1}(y_j=y_i)
```

```math
u_i
=
\exp\left(-\frac{\lvert 1-m_i\rvert}{\tau}\right)
```

```math
c_i^{(2)} = R\left[\widetilde{\alpha}_{y_i}^{p} \cdot \widehat r_i^{\beta} \cdot \rho_i^{\gamma} \cdot u_i \cdot \widehat g_i^{\delta}\right]
```

$\widehat g_i=b+(1-b)g_i$, `gain_floor=b`, mặc định 0; profile conservative
đặt 0.05. Hai sàn mới nhận giá trị hữu hạn trong [0,1]. Sàn 1 loại tác động
của yếu tố tương ứng. `g_validation_gain` vẫn báo tỷ lệ thô, còn
`g_validation_gain_effective` báo yếu tố sau smoothing/clip.

$g_i$ đọc feature và label tập tham chiếu, không đọc test. Đây là proxy đồng
nhãn, không phải mức tăng metric đo trực tiếp. Runner cũ dùng cùng validation
để tính g và chọn tham số; runner bảng mới mặc định tách riêng hai tập này.
Rank được tính trực tiếp từ log-product để exp underflow không đổi thứ tự.

## 6. Tham số

| Tên CLI/code | Ý nghĩa |
|---|---|
| p | độ mạnh class weight |
| beta | số mũ reliability trong hai công thức người dùng; độ mạnh boundary trong robust_hybrid |
| gamma-power | số mũ density trong hai công thức người dùng; số mũ reliability trong robust_hybrid |
| delta | số mũ validation gain trong user_formula_2; số mũ density trong robust_hybrid |
| tau | temperature; CLI cũ 0 là auto, JSON/API bỏ key để auto |
| n-neighbors | số láng giềng k |
| eps | hằng số ổn định số; cờ CLI đúng là --eps |
| local_floor | sàn tuyến tính reliability; mới cho công thức 1/2, mặc định 0 |
| gain_floor | sàn tuyến tính g; chỉ công thức 2, mặc định 0 |
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

```math
J
=
\mathrm{Score}_{\mathrm{validation}}
-\lambda\frac{N_{\mathrm{SV}}}{n_{\mathrm{train}}}
```

$\lambda$ là sv-penalty, $N_{\mathrm{SV}}$ là số support vector, $n_{train}$ là số mẫu
train.
