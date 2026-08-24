# Đối chiếu mã GitHub tác giả và extension

Snapshot upstream: <https://github.com/MojtabaMohasel/BSVM.git>, commit
`ee5a7ae7ade4977b41d604dd989b31f4e356d342`.

| Mã tác giả | Extension | Giữ nguyên / thay đổi |
|---|---|---|
| `Initialsolution(X, y, pipeline)` | `AuthorExtendedBSVMClassifier.fit` phần initial core | Fit SVC trên full train; candidate là điểm sai nhãn hoặc có signed margin dưới 1 |
| `masterproblem(...)` | vòng `while pool` | Tính lại boundary và priority sau khi nhận candidate |
| `extend_samples(...)` | `_candidate_priority` và nhánh `sequential` | Baseline thử lần lượt, nhận candidate đầu tiên không gây misclassification |
| bốn file theo kernel | tham số `kernel` | Một implementation dùng cho linear/poly/sigmoid/RBF |
| toy 2-D | `third_party/BSVM_author/BSVM_extended.py` | Vẫn chạy toy; thêm CLI chọn kernel và priority |
| không có runner Table 2 | `examples/run_author_ci_comparison.py` | Thêm local CSV, paper split/grid, OVR, resume và bảng kết quả |

## Các chỉnh sửa cần thiết

1. Upstream nhận đối số `weights` nhưng priority thực tế là
   `1/(abs(decision)+1e-6)`. Extension dùng
   `alpha_y/(abs(decision)+eps)` ở `author_original`, đúng dòng 9 của Thuật
   toán 4 và class weight công thức (37). Khi equal weights, thứ tự trùng
   upstream.
2. Upstream chỉ kiểm tra `predict(Xnew)==ynew` sau trial. Extension giữ điều
   kiện này và bổ sung dung sai signed margin bằng 0 để xử lý tie số học.
3. Bài báo giả định initial core khả thi và có hai lớp. Extension có core
   repair xác định trước: bảo toàn hai lớp, loại dần điểm core có signed margin
   thấp. Mọi lần sửa được ghi vào `initial_core_repairs`; dùng
   `--no-core-repair` để tắt.
4. `user_formula_2` cần validation feature và label để tính `g_i`. Validation
   được chuyển theo từng bài toán OVR; test không được truyền vào estimator.
5. Ba hàm mới dùng binary-tree insertion. Baseline tiếp tục dùng sequential
   insertion để đại diện thuật toán gốc.

Các file upstream nguyên gốc không bị sửa đè. Mọi thay đổi chạy được nằm ở
`BSVM_extended.py`, `src/roch_bsvm/author_bsvm.py` và runner, giúp kiểm tra diff
và nguồn gốc rõ ràng.

