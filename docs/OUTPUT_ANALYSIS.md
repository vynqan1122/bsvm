# Phân tích output hiện có và hướng cải tiến BSVM

Nguồn là ba bộ kết quả local `outputs/smoke_paper`,
`outputs/smoke_performance_sv`, `outputs/smoke_cloud_sigmoid`, mỗi bộ gồm
`comparison_all.csv`, `tuning_results.csv` và `run_config.json`. Có 12 mô
hình được chọn và 16 cấu hình kernel đã thử, tất cả có trạng thái `ok`.
Các bảng dưới giữ nguyên số đo cũ, không thay bằng kết quả sau chỉnh sửa.

## 1. Phạm vi thực sự của kết quả

| Bộ output | Dataset / kernel | Số mẫu | Train / validation / test | Search | Tiêu chí chọn |
|---|---|---:|---|---|---|
| smoke_paper | fruitfly / linear | 60 trong 125 | 38 / 10 / 12 | none: C=10 | Minority F1 |
| smoke_performance_sv | fruitfly / linear | 60 trong 125 | 38 / 10 / 12 | fast: C=1,10 | Minority F1 − 0.05 × SV/38 |
| smoke_cloud_sigmoid | cloud / sigmoid | 108 | 68 / 18 / 22 | none: C=10, gamma=scale, coef0=0 | Accuracy |

Tất cả dùng seed 42, k=7, p=beta=gamma-power=delta=1, auto temperature.
Không có nhiều seed, full paper grid, kiểm tra nhiễu có kiểm soát, hoặc
khoảng tin cậy. Không trung bình minority F1 của Exp1 với accuracy của Exp2.
Hai lần fruitfly dùng cùng split, không phải hai bằng chứng độc lập.

## 2. Fruitfly với C cố định

| Hàm | F1 minority validation | F1 minority test | ΔF1 test (điểm %) | Accuracy test | Macro F1 test | SV | Giảm SV | Fit (s) | Tổng tuning (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| author_original | 0.4000 | 0.3636 | 0.00 | 0.4167 | 0.4126 | 8 | 0% | 0.0845 | 0.0945 |
| robust_hybrid | 0.5000 | 0.4000 | +3.64 | 0.5000 | 0.4857 | 6 | 25% | 2.2563 | 2.2609 |
| user_formula_1 | 0.6000 | 0.5000 | +13.64 | 0.5000 | 0.5000 | 6 | 25% | 0.0872 | 0.0918 |
| user_formula_2 | 0.2857 | 0.4000 | +3.64 | 0.5000 | 0.4857 | 5 | 37.5% | 0.0739 | 0.0775 |

`user_formula_1` là ứng viên nên kiểm chứng trước: F1 tốt hơn và ít SV hơn
baseline ở cấu hình này. `user_formula_2` đánh đổi 1 SV ít hơn công thức 1
để có F1 thấp hơn 0.10. Xét riêng F1 test và SV, hai công thức này nằm trên
biên Pareto quan sát được; không dùng biên Pareto trên test để chọn tham số.

Accuracy tăng từ 5/12 lên 6/12 dự đoán đúng: chỉ **một mẫu**. Chênh lệch F1
13.64 điểm phần trăm không phải tăng 13.64% tương đối và chưa chứng minh
cải thiện có ý nghĩa thống kê.

Hybrid ghi nhận hơn 2 giây dù có số lần fit gần các hàm khác. Có thể có chi
phí khởi tạo thư viện/kNN hoặc hiệu ứng thứ tự chạy, nhưng output không đủ
để xác định nguyên nhân. Không kết luận hybrid luôn chậm hơn hàng chục lần
từ một phép đo nhỏ; cần warm-up, lặp thời gian và đổi thứ tự model.

## 3. Fruitfly với tiêu chí performance_sv

| Hàm | C chọn | F1 validation | Utility validation | F1 test | ΔF1 test (điểm %) | SV | Giảm SV so baseline | Fit (s) | Tổng tuning (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| author_original | 1 | 0.4444 | 0.4300 | 0.4000 | 0.00 | 11 | 0% | 0.0839 | 0.1442 |
| robust_hybrid | 1 | 0.6000 | 0.5895 | 0.5000 | +10.00 | 8 | 27.27% | 2.3494 | 2.4445 |
| user_formula_1 | 10 | 0.6000 | 0.5921 | 0.5000 | +10.00 | 6 | 45.45% | 0.1026 | 0.2441 |
| user_formula_2 | 10 | 0.2857 | 0.2791 | 0.4000 | 0.00 | 5 | 54.55% | 0.0698 | 0.1610 |

Các phần trăm giảm SV dùng baseline của **cùng bộ chạy**, không lấy 8 SV từ
bộ C=10 để so với 11 SV của baseline đã tune C.

Kiểm tra lại 8 dòng tuning với cùng quy tắc chọn `(utility, validation_score,
-SV)` cho kết quả:

| Hàm | C khi lambda=0 | C khi lambda=0.05 | Lambda có đổi lựa chọn? |
|---|---:|---:|---|
| author_original | 1 | 1 | Không |
| robust_hybrid | 1 | 1 | Không |
| user_formula_1 | 10 | 10 | Không |
| user_formula_2 | 10 | 10 | Không; F1 validation hòa, tie-break chọn ít SV |

Vì vậy không thể nói mức phạt 0.05 đã cải thiện kết quả. Tác động quan sát
giữa hai thư mục còn có thay đổi grid C. Với grid này, utility của hai C chỉ
hòa tại lambda khoảng 0.563 cho author và 1.9 cho hybrid; đây là phân tích
hậu nghiệm trên validation, **không phải khuyến nghị dùng lambda lớn đó**.

## 4. Cloud với sigmoid

| Hàm | Accuracy validation | Accuracy test | Balanced accuracy test | Macro F1 test | SV | Model fits | Fit (s) | Tổng tuning (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| author_original | 0.3333 | 0.3636 | 0.4571 | 0.3583 | 2 | 84 | 0.3124 | 0.3163 |
| robust_hybrid | 0.3333 | 0.3636 | 0.4571 | 0.3583 | 2 | 128 | 0.5722 | 0.5769 |
| user_formula_1 | 0.3333 | 0.3636 | 0.4571 | 0.3583 | 2 | 128 | 0.4700 | 0.4750 |
| user_formula_2 | 0.3333 | 0.3636 | 0.4571 | 0.3583 | 2 | 128 | 0.3779 | 0.3834 |

Cả bốn có `selected_total=2`, `accepted_candidates=0`,
`rejected_candidates=66`, `initial_core_repairs=1`. Các hàm mới cần thêm
44 lần fit được ghi nhận (+52.38%) mà không cải thiện chỉ số. Nên khảo sát
linear/RBF và grid C/gamma trước khi chỉnh số mũ để cứu riêng cấu hình
sigmoid này. Không suy rộng kết quả thành mọi sigmoid đều không tốt.

## 5. Cần sửa gì trong cách viết và dùng công thức?

| Vấn đề | Chỉnh sửa / thí nghiệm đề xuất | Đã triển khai |
|---|---|---|
| README hybrid thiếu sàn của implementation | Viết `(0.05+0.95r)^gamma (0.05+0.95rho)^delta` | Đã sửa mô tả |
| r=0 có thể là một điểm thiểu số hợp lệ | Cho phép `r_hat=a+(1-a)r`; thử a=0.05 | Có `local_floor`, mặc định 0 cho công thức 1/2 |
| g=0 làm công thức 2 phạt quá mạnh | Cho phép `g_hat=b+(1-b)g`; thử b=0.05, delta=0.5 | Có `gain_floor`, mặc định 0 |
| Gọi g là gain có thể gây hiểu sai | Ghi rõ là tỷ lệ đồng nhãn tham chiếu; chưa đo chênh lệch F1 sau thêm điểm | Đã sửa mô tả |
| Rank sau exp vẫn có thể underflow | Rank log-product trực tiếp | Đã sửa code; giữ raw product để chẩn đoán |
| Formula 2 học và chọn tham số trên cùng validation | Chia train/reference/validation/test riêng | Runner bảng mới có `independent` |
| Formula và insertion thay đổi cùng lúc | Chạy cùng sequential hoặc cùng binary_tree | Có `--insertion-policy` |
| Số mũ mặc định 1 chưa được tune | Profile riêng mỗi hàm; chọn trên validation độc lập | Có JSON `legacy`/`conservative`, chưa auto-search số mũ |

Các thay đổi smoothing là giả thuyết cần kiểm chứng. Nếu các điểm bất đồng
nhãn chủ yếu là nhiễu thật, smoothing có thể làm ưu tiên nhiễu tăng lên và
giảm chất lượng. Vì thế cần ablation sàn=0 so với sàn=0.05 trên nhiều dataset
và seed, không tuyên bố công thức mới tốt hơn trước khi có thí nghiệm đó.

## 6. Cách đọc bảng mới

Bảng mới giữ riêng dataset, protocol, algorithm, kernel và seed. Test chỉ
được dùng để báo cáo sau khi chọn mô hình trên validation. Các model baseline
được chạy lại cùng dữ liệu/split, không chép số báo từ PDF cạnh số máy này.

`fit_time_seconds` đo fit của cấu hình được chọn; `tuning_seconds` là toàn
bộ tìm kernel và đánh giá validation; thời gian dự đoán là median các lần
lặp sau warm-up, có đơn vị batch và mỗi mẫu. Với binary BSVM, trường
`model_fits` cũ bỏ qua một lần SVC initial, nên runner mới còn có
`total_model_fits`. Với multiclass OVR, SV là tổng số incidences qua các
bài toán nhị phân, có thể vượt số mẫu train.

Các bảng có nhiều seed ghi mean và sample standard deviation; một seed
không có ước lượng độ lệch chuẩn. Những seed lặp holdout vẫn có tập test
chồng lấn, không phải các quan sát độc lập để tùy ý áp dụng kiểm định. Khi
cần Wilcoxon như bài báo, ghép cặp kết quả theo dataset/protocol và một
quy tắc tổng hợp seed/kernel đã định trước; không coi mọi kernel là một
dataset mới.

Hướng dẫn tham số, ma trận 8 script và các lệnh chạy nằm trong [README](../README.md).
