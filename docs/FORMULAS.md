# Giai thich cac cong thuc moi trong Binary-Tree BSVM

Tai lieu nay giai thich phan cai tien `c_i` va cach them diem theo cay nhi phan
trong source code. Ky hieu dung trong code:

- `y_i` la nhan da ma hoa ve `{-1, +1}`.
- `f(x_i)` la gia tri decision function cua SVM hien tai.
- `|f(x_i)|` la khoang cach theo diem so toi bien phan tach: cang nho thi diem
  cang nam gan bien.
- `w_y` la trong so lop. Voi `class_weight="balanced"`,
  `w_y = n / (2 n_y)`, nen lop thieu so co trong so lon hon.
- Moi cong thuc `c_i` trong code deu la do uu tien: `c_i` cang lon thi diem
  duoc thu them vao tap kha thi cang som.

## 1. Baseline `paper_priority`

Cong thuc:

```text
c_i = class_term_i / (|f(x_i)| + eps)
```

Trong do:

```text
class_term_i = w_{y_i} / min_y(w_y)
```

Y nghia:

- Diem cua lop thieu so duoc tang uu tien nho `class_term_i`.
- Diem gan bien phan tach duoc thu truoc vi `|f(x_i)|` nho.
- `eps` tranh chia cho 0 khi diem nam rat sat bien.

Day la baseline gan voi thuat toan goc. Diem yeu la neu mot diem nhieu nam gan
bien, no cung co `c_i` rat lon va co the duoc thu qua som.

## 2. Cong thuc mem hon `boundary_exp`

Cong thuc:

```text
c_i = class_term_i^alpha * exp(- beta * |f(x_i)| / T)
```

Trong code:

- `alpha` la `class_power`.
- `beta` la `boundary_power`.
- `T` la `temperature`, mac dinh lay median cua `|f(x_i)|` tren tap ung vien,
  co san so duoi de tranh qua nho.

Y nghia:

- Van uu tien lop thieu so va diem gan bien.
- Thay vi `1 / |f(x_i)|`, ham mu `exp(-|f|/T)` giam tron va khong bi bung vo
  han tai bien.
- Neu `beta` lon, code uu tien manh cac diem gan bien hon.
- Neu `alpha` lon, code uu tien lop thieu so manh hon.

Cong thuc nay thuong on dinh hon `paper_priority` khi du lieu co diem sat bien
hoac co outlier.

## 3. Cong thuc mac dinh `robust_hybrid`

Cong thuc:

```text
c_i =
  class_term_i^alpha
  * boundary_i^beta
  * local_term_i^gamma
  * density_term_i^delta
```

Voi:

```text
boundary_i = exp(-|f(x_i)| / T)
local_term_i = local_floor + (1 - local_floor) * q_i
density_term_i = density_floor + (1 - density_floor) * r_i
```

### Thanh phan `q_i`: do nhat quan nhan cuc bo

`q_i` la ty le `k` lang gieng gan nhat co cung nhan voi diem `i`.

```text
q_i = so lang gieng cung nhan / k
```

Neu mot diem co nhan phu hop voi vung xung quanh, `q_i` gan 1. Neu no co ve
bi gan nhan sai, `q_i` gan 0. Thanh phan nay giup giam uu tien diem nhieu nhan.

### Thanh phan `r_i`: do tin cay mat do noi lop

Voi tung lop, code tinh khoang cach trung binh tu diem `i` toi cac lang gieng
cung lop, sau do so voi khoang cach noi lop dien hinh cua lop do.

```text
r_i = 1 / (1 + mean_same_class_distance_i / median_typical_distance_class)
```

Neu diem nam trong cum cung lop, khoang cach nho va `r_i` cao. Neu diem bi co
lap, `r_i` thap. Thanh phan nay lam giam uu tien outlier.

### Vai tro cua cac so mu

- `alpha = class_power`: tang/giam muc uu tien lop thieu so.
- `beta = boundary_power`: tang/giam muc uu tien diem gan bien.
- `gamma = local_power`: tang/giam muc phat diem co nhan khong nhat quan.
- `delta = density_power`: tang/giam muc phat diem bi co lap.

Mac dinh trong `tune_ci_strategy` thu nhieu cau hinh, vi khong co cong thuc nao
tot nhat cho moi dataset.

## 4. Chon cong thuc `c_i` tren validation

Ham `tune_ci_strategy(...)` chay nhieu cau hinh:

```text
paper_priority
boundary_exp
robust_hybrid voi k = 5
robust_hybrid voi k = 9 va phat nhieu manh hon
```

Voi bai toan nhi phan, diem validation mac dinh la `minority_f1`, vi bai bao
tap trung vao du lieu mat can bang. Voi bai toan nhieu lop trong script batch,
code dung One-vs-Rest va chon theo `macro_f1`.

Sau khi chon cau hinh tot nhat, model duoc fit lai tren `train + validation`.

## 5. Them diem theo cay nhi phan

Thuat toan goc thu them ung vien gan nhu tung diem mot. Phien ban moi sap xep
ung vien theo `c_i`, sau do thu theo khoi:

```text
ordered = sort(candidates, key=c_i, descending=True)

try_block(block):
    fit SVM tren core + block
    neu block kha thi:
        nhan ca block
    nguoc lai neu block chi co 1 diem:
        loai diem do
    nguoc lai:
        chia block lam hai nua
        try_block(nua uu tien cao hon)
        try_block(nua con lai)
```

Khi nhieu diem tot co the vao cung luc, cach nay tiet kiem so lan thu. Khi khoi
co diem xung dot, code chia nho den khi tim duoc diem nao nen nhan/loai.

## 6. Cach doc bang so sanh

Trong `outputs/all_data_kernels/comparison_all.csv`:

- `accuracy`, `balanced_accuracy`, `macro_f1`, `minority_f1`: chat luong du doan.
- `ci_strategy`: cong thuc `c_i` da dung.
- `insertion_strategy`: `sequential` cho baseline, `binary_tree` cho ban cai tien.
- `batch_attempts`: so lan thu them ung vien/khoi ung vien.
- `model_fits`: so lan fit SVM phu. Chi so nay cang cao thi chay cang lau.
- `support_vectors`: so support vector cua model cuoi.
- `binary_problems`: bang 1 voi binary; bang so lop voi One-vs-Rest multiclass.

Trong `comparison_deltas.csv`, cac cot `*_delta` duoc tinh:

```text
delta = ket qua cai tien - ket qua baseline
```

Vi du `minority_f1_delta > 0` nghia la ban cai tien tot hon baseline tren F1 cua
lop thieu so o dataset-kernel do.

## 7. Goi y thuc nghiem

- Chay `linear` truoc de kiem tra pipeline.
- Sau do chay `rbf`; day thuong manh hon nhung ton thoi gian hon.
- `poly` va `sigmoid` nen chay sau cung.
- Voi dataset lon, dung `--max-rows` de smoke test truoc, sau do bo tham so nay
  khi can ket qua day du.
- Dung `--resume` de chay tiep neu bi dung giua chung.
- Bat `--tune-ci` khi can ket qua nghiem tuc hon, nhung chi phi co the tang
  khoang 4 lan vi moi cau hinh `c_i` can fit rieng.
