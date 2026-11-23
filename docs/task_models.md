# Giao việc: bảng đặc trưng và huấn luyện mô hình

## 1. Mục tiêu

So sánh các nhóm mô hình trên **cùng dữ liệu IMU**: tuyến tính (Logistic), cây (Random Forest, Gradient Boosting), mạng nơ-ron (CNN1D, RNN, CNN + RNN). Song song, chuẩn bị bảng đặc trưng ToF/nhiệt cho bước sau.

## 2. Tình trạng hiện tại

- Code tiền xử lý đã được tách thành 3 file:
  - `src/preprocess.py`: làm sạch dữ liệu.
  - `src/features_imu.py`: đặc trưng IMU theo từng bước thời gian. Đã gộp đặc trưng của Dũng với `lin_acc`, `grav`, `angvel` từ CNN1D.
  - `src/features_tof.py`: đặc trưng ToF/nhiệt theo từng bước thời gian.
- Chạy `python src/preprocess.py` để tạo `data/processed/train.parquet`.
- Còn thiếu bảng **mỗi chuỗi một dòng**, là đầu vào cho mô hình cây và mô hình tuyến tính.
- Mốc so sánh: CNN1D chỉ dùng IMU đạt CV **0,779** (1 mô hình). Chi tiết trong `reports/cnn1d/report_imu.md`.

## 3. Phân công

| Người làm | Ngày 1–2 | Ngày 3–6 |
|---|---|---|
| Dũng | Bảng đặc trưng IMU | Random Forest, Gradient Boosting |
| Huy | Bảng đặc trưng ToF/nhiệt | Logistic (trên bảng IMU của Dũng), RNN thuần |
| Mạnh | CNN + RNN | CNN + RNN |

## 4. Quy ước chung

1. Dùng `folds/folds.csv` và `src/metric.py`. **Chỉ chọn cấu hình theo CV.** Cuối đợt mới chấm holdout (`split == "test"`), và chỉ chấm 1 lần.
2. Không dùng các cột `phase`, `behavior`, `orientation`, `sequence_type`, vì tập test không có.
3. Bảng đặc trưng: mỗi chuỗi một dòng, có `sequence_id`. Dũng và Huy thống nhất tên file và tên cột từ ngày 1, vì Huy cần bảng IMU từ ngày 3.
4. Mỗi mô hình lưu file OOF gồm các cột `sequence_id, fold, gesture, pred` (giống CNN1D), kèm file xác suất `.npy`. Hai file này dùng để vẽ confusion matrix và ghép các mô hình về sau.
5. Mỗi bước giải thích được: làm gì, vì sao, kiểm chứng thế nào. Ghi lại cả các lần thử thất bại.

## 5. Gợi ý rút ra từ CNN1D

- **`angvel` giúp nhiều nhất** (+0,04). Các BFRB có tư thế cổ tay gần giống nhau, chủ yếu khác nhau ở cách xoay cổ tay.
- **Bỏ quaternion `rot` làm điểm tăng** (0,740 → 0,750).
  - Giả thuyết: `rot` chứa hướng người đang quay mặt trong phòng, giúp mô hình "nhận ra người" thay vì nhận ra cử chỉ.
  - Đặc trưng tính từ `rot` như `rot_yaw` và các góc Euler nên được kiểm tra với cả mô hình cây.
- **Jerk và độ lớn gia tốc không giúp CNN**, vì CNN tự tính được hiệu giữa hai bước liên tiếp. Cây thì không tự tính được, nên với cây các đặc trưng này có thể vẫn có ích.
- **Pha Gesture luôn nằm ở cuối chuỗi.** Thống kê trên đoạn cuối chuỗi có thể có ích.
- **Chỉ đổi seed đã làm CV dao động khoảng 0,006.** Chênh lệch nhỏ hơn mức này cần chạy thêm seed mới kết luận được.
- **Với RNN:**
  - Nên dùng cùng đầu vào với CNN1D để so sánh công bằng: 11 cột `lin_acc_x/y/z, rot_missing, grav_x/y/z, angvel_x/y/z, angvel_mag`, `max_len` 128.
  - Có thể dùng lại `src/cnn1d/prepare.py`.
  - Hàm `make_padded` đệm ở **đầu** chuỗi, nên trạng thái ẩn ở bước cuối luôn ứng với dữ liệu thật.
