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

## Kết quả hiện có có tốt không?

Các thư mục `outputs/smoke_*` chỉ có **12 kết quả, 2 dataset, 1 seed (42)**.
Fruitfly được lấy 60/125 mẫu, chỉ có 12 mẫu test. Đây là bằng chứng thử nghiệm
ban đầu, chưa đủ để kết luận một công thức luôn tốt hơn hoặc chống nhiễu tốt hơn.

| Lần chạy | Hàm | F1 thiểu số test | Accuracy test | SV | Số fit ghi nhận |
|---|---|---:|---:|---:|---:|
| fruitfly, linear, C=10 | author_original | 0.3636 | 0.4167 | 8 | 21 |
| cùng cấu hình | robust_hybrid | 0.4000 | 0.5000 | 6 | 25 |
| cùng cấu hình | user_formula_1 | **0.5000** | 0.5000 | 6 | 23 |
| cùng cấu hình | user_formula_2 | 0.4000 | 0.5000 | **5** | 25 |
| fruitfly, linear, chọn C từ {1,10}, phạt SV | author_original | 0.4000 | 0.5000 | 11 | 35 |
| cùng quy trình | robust_hybrid | **0.5000** | 0.5000 | 8 | 34 |
| cùng quy trình | user_formula_1 | **0.5000** | 0.5000 | 6 | 23 |
| cùng quy trình | user_formula_2 | 0.4000 | 0.5000 | **5** | 25 |
| cloud, sigmoid, C=10, gamma=scale | author_original | 0.4167 | 0.3636 | 2 | 84 |
| cùng cấu hình | cả ba hàm mới, từng hàm | 0.4167 | 0.3636 | 2 | 128 |

**Nên ưu tiên kiểm chứng `user_formula_1`**: trên fruitfly C=10, F1 tăng
13.64 điểm phần trăm và SV giảm 25% so với baseline. `user_formula_2` gọn hơn
(5 thay vì 8 SV, giảm 37.5%) nhưng F1 thấp hơn công thức 1. Hybrid chưa cho
thấy lợi thế ổn định so với công thức 1. Cloud-sigmoid chỉ giữ 2 điểm, không
nhận thêm candidate nào, cả bốn mô hình đều sửa initial core; đây là trường hợp
cấu hình kernel không hiệu quả, không phải bằng chứng rằng 2 SV là tốt.

Tên `smoke_paper` dễ gây hiểu nhầm: thực tế nó dùng `search=none`. So với nó,
`smoke_performance_sv` vừa đổi cách chọn mô hình vừa đổi grid C. Trong chính
grid `{1,10}`, bỏ mức phạt 0.05 vẫn chọn cùng C cho cả bốn mô hình (kể cả tie-break
ít SV). Vì vậy chưa thể quy cải thiện cho số hạng phạt SV.

Xem số liệu chi tiết, thời gian và giới hạn diễn giải tại
[docs/OUTPUT_ANALYSIS.md](docs/OUTPUT_ANALYSIS.md).

## Ký hiệu chung

Với candidate $(x_i,y_i)$:

```math
f(x_i)=\sum_{s\in SV}\lambda_s y_s K(x_s,x_i)+b
```

```math
m_i=y_i f(x_i)
```

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

```math
c_i^{\mathrm{author}}
=
\frac{\widetilde{\alpha}_{y_i}}
{\lvert f(x_i)\rvert+\varepsilon}
```

Hàm ưu tiên điểm gần decision boundary và lớp thiểu số. Candidate được thêm
tuần tự như control flow của mã tác giả.

Lưu ý: công thức (13) của bài báo in $|f(x_i)|/|w_{y_i}|$, nhưng dòng 9 và 13
của Thuật toán 4 dùng tỷ số nghịch đảo rồi sắp giảm dần. Baseline dùng phiên
bản của Thuật toán 4. Snapshot GitHub thực tế dùng
$1/(|f(x_i)|+10^{-6})$ và bỏ qua đối số weights; extension nhân class weight
để bám sát dòng 9.

### 2. robust_hybrid — hàm mới thứ nhất

```math
c_i^{\mathrm{hybrid}}
=
\widetilde{\alpha}_{y_i}^{p}
\exp\left(-\beta\frac{\lvert f(x_i)\rvert}{T}\right)
\widehat r_i^{\gamma}\widehat \rho_i^{\delta}
```

Hàm ưu tiên điểm gần **decision boundary**, đồng thời giảm ưu tiên điểm có
nhãn không phù hợp với lân cận hoặc là outlier cô lập.
Với hybrid, $\widehat r_i=0.05+0.95r_i$ và
$\widehat\rho_i=0.05+0.95\rho_i$ theo implementation. Với các công thức
người dùng, implementation dùng $\max(\varepsilon,\widehat r_i)$,
$\max(\varepsilon,\widehat g_i)$ và $\max(\varepsilon,\rho_i)$ trước khi
lấy log; các công thức dưới đây viết tắt bước clip này.

### 3. user_formula_1 — hàm mới thứ hai

```math
c_i^{(1)} = R\left[\widetilde{\alpha}_{y_i}^{p} \cdot \widehat r_i^{\beta} \cdot \rho_i^{\gamma} \cdot \exp\left(-\frac{\lvert 1-m_i\rvert}{\tau}\right)\right]
```

Số hạng mũ đạt cực đại tại $m_i=1$, nên hàm ưu tiên điểm gần **đường margin
đơn vị**, không phải decision boundary.

Đặt $\widehat r_i=a+(1-a)r_i$, với `local_floor=a`, rồi dùng
$\max(\varepsilon,\widehat r_i)$. Mặc định $a=0$
giữ công thức cũ; profile đề xuất dùng $a=0.05$. Điều này tránh hạ ưu tiên
gần như về 0 chỉ vì k láng giềng không có điểm cùng nhãn. Điểm thiểu số hợp
lệ nằm giữa lớp đa số cũng có thể có $r_i$ thấp, không nhất thiết là nhiễu.

### 4. user_formula_2 — hàm mới thứ ba

Validation-gain proxy:

```math
g_i
=
\frac{1}{k}
\sum_{x_j\in N_k^{\mathrm{Ref}}(x_i)}
\mathbf{1}(y_j=y_i)
```

Công thức đầy đủ:

```math
c_i^{(2)} = R\left[\widetilde{\alpha}_{y_i}^{p} \cdot \widehat r_i^{\beta} \cdot \rho_i^{\gamma} \cdot \exp\left(-\frac{\lvert 1-y_i f(x_i)\rvert}{\tau}\right) \cdot \widehat g_i^{\delta}\right]
```

$\widehat g_i=b+(1-b)g_i$, với `gain_floor=b`, rồi dùng
$\max(\varepsilon,\widehat g_i)$. Mặc định $b=0$, profile đề xuất
dùng $b=0.05$. $g_i$ là **tỷ lệ đồng nhãn trên tập tham chiếu**, chưa phải mức
tăng F1/accuracy thực sau khi thêm candidate. Không nên gọi nó là validation
gain đo trực tiếp. Chỉ số này có thể tiếp tục ưu tiên lớp đa số khi tập tham
chiếu mất cân bằng.

Runner cũ dùng `Ref=validation` và cũng chọn tham số trên validation đó.
Runner bảng mới mặc định chia riêng `Ref` và `validation-select`; test luôn
độc lập. Việc làm mềm và tách tập là thay đổi phương pháp cần báo cáo rõ.

## Giải thích đầy đủ tham số

Class weight:

```math
\alpha_y=\frac{n}{K n_y}
```

```math
\widetilde{\alpha}_y
=
\frac{\alpha_y}{\min_c\alpha_c}
```

- $n$: tổng số mẫu train; $K$: số lớp; $n_y$: số mẫu train của lớp y.
- $\alpha_y$: class weight, ưu tiên lớp thiểu số.
- $\widetilde{\alpha}_y$: class weight đã chuẩn hóa; không đổi tỷ lệ giữa lớp.
- $p$: số mũ điều khiển độ mạnh của class weight.

Label reliability:

```math
r_i
=
\frac{1}{k}
\sum_{x_\ell\in N_k^{\mathrm{train}}(x_i)}
\mathbf{1}(y_\ell=y_i)
```

$r_i$ là tỷ lệ láng giềng train cùng nhãn; candidate không tính là láng giềng
của chính nó. Giá trị thấp gợi ý nhiễu nhãn.

Local density:

```math
\rho_i
=
\frac{1}
{1+D_i/(s_{y_i}+\varepsilon)}
```

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
  0.25. CLI cũ `--tau` dương cũng đặt $T$; profile JSON dùng `temperature`.
- $\tau$: temperature của hai công thức người dùng. Khi **--tau 0**:

```math
\tau
=
\max\left\{
\mathrm{median}_i\lvert 1-m_i\rvert,
0.25,
\varepsilon
\right\}
```

- $R[\cdot]$: rank normalization:

```math
R(z_i)
=
\frac{\mathrm{rank}_{\mathrm{ascending}}(z_i)}
{\lvert H\rvert}
```

  $H$ là pool được xếp hạng hiện tại. Code rank trực tiếp log-product để
  tránh underflow. Chỉ dùng `exp(log_raw-max)` cho giá trị chẩn đoán, không
  dùng nó để rank. Các giá trị bằng nhau được phá hòa theo thứ tự ổn định.
- $\mathbf{1}(\cdot)$: hàm chỉ báo, bằng 1 khi điều kiện đúng, ngược lại bằng
  0.
- $N_k^{train}(x_i)$: k láng giềng gần nhất trong train.
- $N_k^{Ref}(x_i)$: k láng giềng gần nhất trong tập tham chiếu của $g_i$.

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

author_original mặc định thêm tuần tự, còn các hàm mới dùng cây. Vì thế so
sánh này đo cả **công thức và cách thêm candidate**. Runner bảng cho phép
`--insertion-policy sequential` hoặc `binary_tree` áp dụng cùng cách thêm cho
bốn BSVM để làm ablation; baseline lúc đó phải ghi là đã đổi insertion.

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

Protocol của runner cũ là 64% train, 16% validation, 20% holdout test,
stratified, seed 42. Experiment 1 chọn minority-F1 với balanced class weight;
Experiment 2 chọn accuracy với equal weights. Test không tham gia tính $c_i$
hay chọn tham số.

## Cấu hình riêng từng hàm

Runner cũ `run_author_ci_comparison.py` dùng các cờ ở bảng dưới. Runner mới
`run_paper_table.py` đọc [config/ci_profiles.json](config/ci_profiles.json)
để mỗi hàm có tham số riêng trong cùng một thí nghiệm.

| Ý nghĩa | Cờ CLI cũ | Key JSON hybrid | Key JSON công thức 1/2 |
|---|---|---|---|
| Số láng giềng | `--n-neighbors` | `n_neighbors` | `n_neighbors` |
| Số mũ class weight | `--p` | `class_power` | `p` |
| Độ mạnh ưu tiên boundary | `--beta` cho hybrid | `boundary_power` | Không có; target là signed margin 1 |
| Số mũ reliability | `--gamma-power` cho hybrid, `--beta` cho 1/2 | `local_power` | `beta` |
| Số mũ density | `--delta` cho hybrid, `--gamma-power` cho 1/2 | `density_power` | `gamma` |
| Số mũ đồng nhãn tham chiếu | `--delta` cho 2 | Không dùng | `delta`, chỉ công thức 2 |
| Temperature | `--tau` | `temperature` | `tau` |
| Sàn reliability | Chỉ có qua JSON/API mới | `local_floor` | `local_floor` |
| Sàn đồng nhãn tham chiếu | Chỉ có qua JSON/API mới | Không dùng | `gain_floor`, chỉ công thức 2 |
| Ổn định số | `--eps` | `eps` | `eps` |

**Không nhầm JSON `gamma` của công thức 1/2 với gamma kernel RBF.**
`author_original` chỉ dùng `eps` trong priority; các số mũ ở đây không chỉnh
baseline này. Khi equal class weights, mọi class term bằng 1 nên chỉnh `p`
không có tác dụng. `delta` không có tác dụng với công thức 1.

Profile `legacy` khớp tham số các smoke output (mọi số mũ 1, k=7, auto
temperature; hybrid giữ sàn implementation 0.05, còn hai công thức người dùng
có sàn 0). Profile `conservative` là **khởi điểm để
thử**, chưa phải cấu hình tối ưu:

| Hàm | p | Mũ boundary | Mũ reliability | Mũ density | Mũ g | k | Temperature | Sàn r / g |
|---|---:|---:|---:|---:|---:|---:|---|---|
| author_original | Không dùng | Nghịch đảo | Không dùng | Không dùng | Không dùng | Không dùng | Không dùng | Không dùng |
| robust_hybrid | 1 | 1 | 1 | 0.5 | Không dùng | 7 | Auto | r=.05, rho=.05 |
| user_formula_1 | 1 | Không dùng | 1 | 0.5 | Không dùng | 7 | Auto | 0.05 / không dùng |
| user_formula_2 | 1 | Không dùng | 1 | 0.5 | 0.5 | 7 | Auto | 0.05 / 0.05 |

Trong JSON/API, **bỏ key `tau`/`temperature` để dùng auto**, không ghi 0.
Chỉ CLI của runner cũ chuyển `--tau 0` thành auto. Temperature dương là giá
trị cố định, ví dụ `"tau": 0.5`.

Nên điều chỉnh theo thứ tự dưới đây, mỗi lần giữ nguyên các yếu tố còn lại:

1. Chạy `legacy` trên dữ liệu đầy đủ và RBF/linear để lấy mốc. Sau đó so với
   `conservative` bằng cùng split, kernel grid và insertion policy.
2. Chọn k trong `{3,5,7,11}`; tập nhỏ hoặc lớp thiểu số ít mẫu nên thử 3/5
   trước. k=7 trên validation chỉ 10 mẫu của fruitfly smoke lấy tới 70% tập
   này, nên $g_i$ ít còn tính cục bộ. Code giới hạn k theo số điểm khả dụng.
3. Nếu điểm thiểu số bị giảm ưu tiên nhiều, thử mũ reliability `{0,0.5,1}` và
   sàn `{0,0.05,0.1}`. Nếu có outlier cô lập, thử mũ density `{0,0.5,1}`.
   Mũ 0 là ablation bỏ một yếu tố, không phải giá trị tốt mặc định.
4. Hybrid: thử boundary power `{0.5,1,2}`. Công thức 1/2: thử tau auto hoặc
   `{0.5,1,2}`. Tau nhỏ ưu tiên rất hẹp quanh margin 1; tau lớn làm ảnh hưởng
   margin yếu hơn. Giữ target 1 nếu mục tiêu là chọn điểm gần support margin.
5. Công thức 2: thử delta `{0,0.25,0.5,1}` trên validation-select độc lập.
   Delta 0 loại ảnh hưởng g về mặt ranking nhưng code vẫn tính g; dùng công
   thức 1 cùng tham số để so chi phí khi bỏ hẳn việc tính g.
6. Chỉ Exp1 cần thử p `{0.5,1,1.5,2}`. Khi tối ưu độ gọn, bắt đầu lambda
   `{0,0.01,0.05,0.1}`; chọn trên validation, báo cáo cả F1/accuracy và SV.

`--search paper` **chỉ dò kernel**, không tự dò profile/k/số mũ/sàn. Muốn
so profile, sao chép một mục trong JSON, đổi tên, sửa một nhóm tham số rồi
chạy `--profile ten_moi --output-dir outputs/ten_moi`. Chọn profile trên
validation; không chọn dựa trên test vừa xem. Muốn kết luận sau nhiều vòng
thử nghiệm cần outer CV hoặc một test cuối chưa dùng để điều chỉnh.

## Mỗi bảng bài báo một file .sh

Chạy bằng Bash trên Linux, WSL hoặc Git Bash. Các script tìm Python trong
`.venv-linux`, `.venv`, hoặc biến `PYTHON`; luôn chuyển về thư mục project.
Để cài môi trường mới trong Linux/WSL:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Các bảng giữ model của bảng gốc và thêm ba thuật toán mới. Thuật toán được
đặt thành **các dòng** để bảng dễ đọc; kết quả có cả CSV và Markdown.

| Bảng | Script | Mô hình / nội dung |
|---|---|---|
| 4 | [table_4.sh](scripts/paper_tables/table_4.sh) | Soft Margin, Weighted, NuSVC, BSVM tác giả + 3 hàm; RBF, F1 từng lớp train/test |
| 5 | [table_5.sh](scripts/paper_tables/table_5.sh) | Soft Margin, NuSVC, BSVM tác giả + 3 hàm; RBF, accuracy và macro R/P/F1 |
| 6 | [table_6.sh](scripts/paper_tables/table_6.sh) | Cùng 6 mô hình, hai protocol; thời gian fit/dự đoán và SV |
| C.8 | [table_c8.sh](scripts/paper_tables/table_c8.sh) | BSVM tác giả + 3 hàm; F1 từng lớp, 4 kernel |
| C.9 | [table_c9.sh](scripts/paper_tables/table_c9.sh) | Soft Margin + 3 hàm; F1 từng lớp, 4 kernel |
| C.10 | [table_c10.sh](scripts/paper_tables/table_c10.sh) | BSVM tác giả + 3 hàm; 9 dataset Exp2 + steel, 4 kernel |
| C.11 | [table_c11.sh](scripts/paper_tables/table_c11.sh) | Soft Margin + 3 hàm; cùng dataset/chỉ số C.10 |
| C.12 | [table_c12.sh](scripts/paper_tables/table_c12.sh) | Soft Margin, BSVM tác giả + 3 hàm; Fashion subset/glass, OVR, 4 kernel |

Xem kế hoạch và chạy thử một bảng nhỏ trước:

```bash
bash scripts/paper_tables/table_4.sh --dry-run
bash scripts/paper_tables/table_4.sh --datasets fruitfly --search none --max-rows 60 --output-dir outputs/table4_smoke
```

Chạy bảng đầy đủ theo dữ liệu đang có và grid Table 3:

```bash
bash scripts/paper_tables/table_4.sh --resume
bash scripts/paper_tables/table_5.sh --resume
bash scripts/paper_tables/table_6.sh --resume
bash scripts/paper_tables/table_c8.sh --resume
bash scripts/paper_tables/table_c9.sh --resume
bash scripts/paper_tables/table_c10.sh --resume
bash scripts/paper_tables/table_c11.sh --resume
bash scripts/paper_tables/table_c12.sh --resume
```

Mặc định mới: `search=paper`, không cắt dữ liệu, `profile=conservative`,
`gain-protocol=independent`, seed 42. Full grid có thể chạy lâu, đặc biệt
leukemia, polynomial và Fashion OVR. `--search fast` giảm grid để khảo sát.
Các bảng chạy độc lập; fit trùng giữa hai bảng chưa dùng cache chung.

Ví dụ đối chiếu profile, nhiều seed và giữ chung insertion:

```bash
bash scripts/paper_tables/table_4.sh --profile legacy --gain-protocol independent --insertion-policy sequential --seeds 42,43,44,45,46 --output-dir outputs/table4_legacy_sequential --resume
bash scripts/paper_tables/table_4.sh --profile conservative --gain-protocol independent --insertion-policy sequential --seeds 42,43,44,45,46 --output-dir outputs/table4_conservative_sequential --resume
```

Chế độ sát split trong phần mô tả bài báo và runner cũ:

```bash
bash scripts/paper_tables/table_5.sh --profile legacy --gain-protocol shared_validation --output-dir outputs/table5_legacy_shared --resume
```

| Protocol | Train | Ref để tính g | Validation chọn tham số | Test |
|---|---:|---:|---:|---:|
| `shared_validation` | 64% | Dùng chung 16% validation | 16% | 20% |
| `independent` | 64% | 8% | 8% | 20% |

Các tỷ lệ thực tế chịu làm tròn và stratification; output ghi số mẫu thực.
Mọi mô hình trong cùng lần chạy dùng cùng các tập train/select/test. Formula 2
dùng thêm nhãn Ref và phải khai báo điều đó trong báo cáo. Đây không phải
so sánh cùng lượng nhãn tham gia học. Dataset rất nhỏ như shuttle (15 mẫu)
không đủ để chia Ref/select giữ đủ lớp: runner ghi lỗi split, không tự đổi
protocol. Có thể chạy `shared_validation` trong thư mục riêng và nêu giới hạn
của Formula 2, hoặc dùng thiết kế cross-fitting trong nghiên cứu tiếp theo.

Output mỗi script ở `outputs/paper_tables/table_<id>/`: `table.csv`,
`table.md`, `results_long.csv`, `tuning_results.csv`, `paired_deltas.csv`,
`run_config.json` và `completeness.json`. File cuối ghi số tổ hợp thành công,
thiếu dataset và lỗi. Script có thể hoàn thành việc xuất bảng nhưng bảng vẫn
chưa đủ dữ liệu; xem cảnh báo `INCOMPLETE` và `requested_plan_complete`.
Các kết quả có seed riêng, không lấy seed có test tốt nhất.
Chỉ số thiếu/lỗi giữ trạng thái rõ ràng, không chuyển thành điểm 0.

Runner mới kiểm tra fingerprint cấu hình/dữ liệu khi `--resume`; đổi profile,
seed hoặc protocol phải dùng output-dir mới. **Runner cũ chưa có bảo vệ này**:
không dùng lại output-dir cũ với `--resume` sau khi đổi tham số.

## Mức độ đối chiếu với bài báo

- Đây là chạy lại theo cấu trúc bảng và protocol được ghi rõ, không cam kết
  tái tạo từng số đã in trong bài. Seed, preprocessing và các chỉnh sửa BSVM
  có thể khác thực nghiệm gốc. `author_original` theo Algorithm 4, không phải
  snapshot upstream chạy nguyên vẹn.
- Bảng 4/C.8/C.9 thiếu `credit-approval` (OpenML 29) trong workspace. Runner
  giữ hàng `missing_dataset`; bổ sung CSV có target vào data-dir để chạy đủ.
- Fashion hiện có 2.000 mẫu, 100 predictor, 10 lớp (OpenML 44701), đúng subset
  được liệt kê trong bài; không phải Fashion-MNIST đầy đủ 70.000 ảnh.
- Polynomial của runner bảng dùng `gamma=1` để khớp $(x^Tz+c)^d$ trong bài.
  Runner cũ dùng `gamma=scale`. RBF/sigmoid dùng gamma của grid Table 3.
- Soft Margin dùng equal penalties; Weighted dùng balanced; NuSVC dùng
  balanced ở Exp1 và equal ở Exp2. Giá trị nu không khả thi được ghi trong
  tuning log, không tự thay grid bài báo.
- Thời gian fit, toàn bộ tuning, predict batch và predict mỗi mẫu là các đại
  lượng khác nhau. Không so thời gian trên máy này với thời gian HPC của bài.
  SV multiclass là tổng qua OVR, không phải số mẫu SV phân biệt.
- Bảng 1–3 và A.7 là ký hiệu/dataset/grid/tổng quan, không cần script train.
  Thí nghiệm nhiễu có kiểm soát là Figure 6–7, chưa nằm trong tám script bảng.
  Kết quả bảng chưa chứng minh riêng khả năng chống nhiễu nhãn/đặc trưng.

## Performance và số support vector

Chế độ mở rộng tối ưu:

```math
J
=
\mathrm{Score}_{\mathrm{validation}}
-\lambda\frac{N_{\mathrm{SV}}}{n_{\mathrm{train}}}
```

$\lambda$ là mức phạt, $N_{\mathrm{SV}}$ là số support vector, $n_{train}$ là số mẫu
train.

~~~powershell
python examples/run_author_ci_comparison.py --search paper --selection-objective performance_sv --sv-penalty 0.05 --output-dir outputs/author_ci_performance_sv --resume
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

