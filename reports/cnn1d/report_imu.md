# Báo cáo: tối ưu CNN1D chỉ dùng IMU

## 1. Mục tiêu
- Đẩy mô hình CNN1D **chỉ dùng cảm biến IMU** lên mức tốt nhất có thể, làm mốc để sau đó đo xem ToF/nhiệt độ đóng góp thêm bao nhiêu.

## 2. Cách đánh giá
- **Chọn cấu hình chỉ dựa trên CV 5 fold theo người** (65 người dev).
- **Holdout (16 người) chỉ chấm 1 lần** ở cuối. Nếu chấm nhiều lần, ta sẽ vô tình chọn theo holdout và điểm bị lạc quan.
- **Mỗi lần chỉ đổi 1 thứ** so với cấu hình tốt nhất hiện tại, để biết chính xác thay đổi nào có tác dụng.
- Một thay đổi được giữ khi tăng rõ hơn nhiễu và tốt hơn ở đa số fold.
- Mỗi bước dưới đây trình bày theo: **Quan sát → Giả thuyết → Thử → Kết quả → Rút ra**. Cuối mỗi bước có bảng tổng hợp số liệu.

## 3. Mô hình nền

### 3.1. Base

**Cấu hình**
- Input: `lin_acc_x/y/z`, `rot_w/x/y/z`, `rot_missing`. Có lật tay trái, không sửa dấu quaternion.
- Conv1d [32, 64, 128], kernel 5, ReLU, MaxPool(2), Global Avg Pool, Linear 18.
- Adam lr 1e-3, batch 64, max_len 128 (giữ cuối chuỗi), tối đa 200 epoch, early stopping patience 20 theo val score.
- 5 fold theo người, 1 seed (42).

**Kết quả**

| | Score | Binary F1 | Macro F1 |
|---|---|---|---|
| CV val | 0,6927 ± 0,011 | 0,947 | 0,439 |
| Train (epoch tốt nhất) | 0,870 | | |

- Epoch tốt nhất 63–80, dừng ở 83–100.
- So với input `acc` gốc (~0,66), dùng `lin_acc` tăng ~0,03; macro F1 từ ~0,38 lên ~0,44.

**Nhận xét**
- Overfit rõ: val loss thấp nhất ở epoch ~30 rồi tăng (1,53 → ~2,1), train loss vẫn giảm.
- Train score 0,87 so với val 0,69.
- Val score vẫn nhích nhẹ sau epoch 30 dù val loss tăng → mô hình ngày càng tự tin quá mức.
- Điểm yếu chính: macro F1 (phân biệt 8 BFRB); binary F1 đã cao.

### 3.2. Label smoothing

**Giả thuyết:** mô hình tự tin quá mức (val loss tăng trong khi score vẫn nhích) → làm "mềm" nhãn để mô hình bớt chắc chắn.

| LS | CV val | Binary F1 | Macro F1 | Val loss tăng sau đáy | Số fold hơn Base |
|---|---|---|---|---|---|
| 0 (Base) | 0,6927 ± 0,011 | 0,947 | 0,439 | +12–22% | |
| 0,05 | 0,6975 ± 0,014 | 0,949 | 0,447 | +1–5% | 3/5 |
| 0,1 | 0,6972 ± 0,009 | 0,950 | 0,446 | +1–4% | 5/5 |
| 0,2 | 0,6963 ± 0,009 | 0,947 | 0,445 | +1–5% | 3/5 |

**Nhận xét**
- Hết quá tự tin: val loss gần như không còn tăng sau đáy, ở cả 3 giá trị.
- Score tăng ~0,004–0,005 so với Base; giữa các giá trị LS chênh ≤ 0,0012, chưa đủ để chọn giá trị tốt nhất với 1 seed.
- Overfit vẫn còn (train ~0,85–0,89, val ~0,70) vì nguyên nhân gốc là dữ liệu ít, mỗi người làm một kiểu.
- Chọn **LS = 0,1** làm nền: giá trị kinh điển, std thấp nhất, hơn Base ở cả 5 fold.

→ **Điểm xuất phát cho các bước sau: CV = 0,697.**

## 4. Quá trình tối ưu

### Bước 1: Phân tích lỗi trước khi thử

**Quan sát:** confusion matrix của mô hình nền cho thấy:
- Binary F1 đã ~0,95; lỗi gần như toàn bộ là **nhầm giữa 8 BFRB với nhau**.
- Tệ nhất: véo da cổ (đúng 23%), nhổ lông mày (26%).

**Phân tích dữ liệu:** để biết vì sao nhầm, tính hướng cổ tay trung bình (hướng trọng lực `grav`) của từng chuỗi trong pha Gesture:
- Các cặp hay nhầm có tâm hướng cổ tay chỉ lệch nhau **2–6°** (véo cổ vs gãi cổ 2°, má vs lông mi 3°).
- Trong khi các lần làm **cùng một cử chỉ** đã lệch nhau 9–15°.
- Cường độ chuyển động trung bình cũng gần như bằng nhau (véo cổ 0,081 vs gãi cổ 0,080 rad/bước).

**Rút ra:** chỉ dựa vào tư thế cổ tay hay độ mạnh động tác thì không phân biệt được. Nếu IMU còn thông tin phân biệt, nó phải nằm ở **chi tiết chuyển động theo thời gian** (xoay thế nào, theo nhịp nào). Đây là định hướng cho bước đặc trưng.

**Kiểm tra phụ (để loại trừ nguyên nhân khác):**
- max_len 128 có cắt mất dữ liệu không? Chỉ 4,9% chuỗi dài hơn 128, và pha Gesture luôn nằm ở cuối chuỗi → không phải nguyên nhân, không xử lý.

### Bước 2: Đặc trưng

**2.1. Thêm grav (hướng trọng lực)**
- **Giả thuyết:** độ nghiêng cổ tay cho biết tay đang ở gần vùng nào (đầu, mặt, cổ). rot cũng chứa thông tin này nhưng ở dạng quaternion khó học.
- **Kết quả:** 0,697 → 0,701, tốt hơn 4/5 fold. Tăng ít, chủ yếu ở binary F1 → **giữ**.
- **Rút ra:** đúng như phân tích Bước 1: hướng cổ tay giúp tách BFRB khỏi non-BFRB, nhưng không tách được các BFRB với nhau.

**2.2. Thêm angvel (vận tốc góc)**
- **Giả thuyết:** từ Bước 1, khác biệt phải nằm ở chuyển động. angvel đo cổ tay xoay bao nhiêu giữa hai bước liên tiếp, đúng loại thông tin còn thiếu.
- **Kết quả:** 0,701 → **0,740**, tốt hơn cả 5 fold. Macro F1 tăng mạnh 0,45 → 0,52 → **giữ**.
- **Rút ra:** xác nhận giả thuyết: các BFRB khác nhau ở **cách xoay cổ tay**, không phải ở tư thế.

**2.3. Thêm acc gốc** — *thất bại*
- **Giả thuyết:** acc gốc = lin_acc + trọng lực. Thử xem mô hình có tận dụng được dạng gốc không.
- **Kết quả:** 0,697 (giảm nhẹ) → **bỏ**.
- **Rút ra:** thông tin trong acc gốc đã có đủ trong lin_acc + grav, thêm vào chỉ là dư thừa.

**2.4. Thêm lin_acc_mag và jerk** — *thất bại*
- **Giả thuyết:** angvel là "đạo hàm" của hướng và đã giúp rất nhiều → thử tương tự với gia tốc: độ lớn gia tốc và jerk (đạo hàm của lin_acc).
- **Kiểm tra trước:** jerk sẽ bị sai nếu rot thiếu một đoạn giữa chuỗi (lin_acc nhảy về 0). Kiểm tra: cả 50 chuỗi thiếu rot đều thiếu toàn bộ → jerk hợp lệ.
- **Kết quả:** lin_acc_mag 0,740 (không đổi), jerk 0,736 (giảm) → **bỏ cả hai**.
- **Rút ra:** CNN tự tính được các đại lượng này từ lin_acc (phép trừ hai bước liên tiếp chính là một bộ lọc conv đơn giản). angvel khác vì nó được tính từ quaternion — phép tính mà CNN khó tự học.

**2.5. Sửa dấu quaternion** — *thất bại*
- **Giả thuyết:** q và −q là cùng một hướng; nếu dấu đổi giữa chuỗi, rot nhảy đột ngột, gây nhiễu cho mô hình.
- **Kết quả:** 0,739 (không tăng) → **bỏ**.

**2.6. Bỏ rot**
- **Quan sát:** sửa dấu không giúp → có thể vấn đề không nằm ở dấu mà ở bản thân rot.
- **Giả thuyết:** rot là hướng của đồng hồ trong **hệ tọa độ thế giới**, gồm cả việc người đó đang quay mặt về hướng nào trong phòng. Thông tin này ngẫu nhiên, không liên quan cử chỉ, và mô hình có thể học thuộc theo nó. Phần có ích của rot (độ nghiêng, chuyển động xoay) đã có trong grav và angvel.
- **Kết quả:** 0,740 → **0,750**, tốt hơn cả 5 fold; chạy lại seed khác ra 0,750 → **giữ**.
- **Lưu ý:** giả thuyết phù hợp với kết quả nhưng chưa được kiểm chứng trực tiếp.

**Bảng tổng hợp Bước 2** (mỗi dòng đổi 1 thứ so với dòng "Nền" gần nhất phía trên, 1 seed):

| Thay đổi | CV val | Macro F1 | Số fold hơn nền | Quyết định |
|---|---|---|---|---|
| Nền: LS 0,1 | 0,6972 | 0,446 | | |
| + grav | 0,7007 | 0,448 | 4/5 | Giữ |
| + angvel | **0,7400** | 0,515 | 5/5 | Giữ |
| + acc gốc | 0,6970 | 0,446 | 2/5 | Bỏ |
| Nền: + grav + angvel | 0,7400 | 0,515 | | |
| + lin_acc_mag | 0,7400 | 0,514 | 3/5 | Bỏ |
| + jerk | 0,7359 | 0,504 | 2/5 | Bỏ |
| Sửa dấu quaternion | 0,7391 | 0,511 | 2/5 | Bỏ |
| Bỏ rot | **0,7495** | 0,532 | 5/5 | Giữ |

**Input chốt:** `lin_acc_x/y/z, rot_missing, grav_x/y/z, angvel_x/y/z, angvel_mag` (11 cột).

### Bước 3: Chống overfit và kiến trúc

**Quan sát:** train 0,90 nhưng val chỉ 0,75 → overfit rõ. Đây là vấn đề chính cần xử lý.

**3.1. Dropout 0,3**
- **Lý do:** cách chống overfit trực tiếp và phổ biến nhất: tắt ngẫu nhiên một phần nơ-ron khi học, buộc mô hình không dựa vào vài chi tiết riêng của từng người.
- **Kết quả:** 0,750 → **0,765**, tốt hơn cả 5 fold. Train giảm còn 0,86, epoch tốt nhất muộn hơn (học chậm nhưng tổng quát hơn) → **giữ**.

**3.2. Dropout 0,5** — *thất bại*
- **Giả thuyết:** dropout có tác dụng → mạnh hơn có tốt hơn không?
- **Kết quả:** 0,746. Train tụt còn 0,80 → mô hình không còn học đủ (underfit) → **bỏ**, giữ 0,3.

**3.3. BatchNorm** — *thất bại*
- **Lý do thử:** BatchNorm là thành phần chuẩn trong hầu hết CNN hiện đại, thường giúp học ổn định.
- **Kết quả:** 0,732. Mô hình đạt đỉnh rất sớm (epoch 14–36) rồi overfit → **bỏ**.
- **Giả thuyết về nguyên nhân:** trung bình 47% độ dài mỗi chuỗi là phần đệm, BatchNorm tính thống kê trên cả phần đệm → kiểm tra ở mục 3.7.

**3.4. Mixup** — *thất bại*
- **Lý do thử:** các đội top cuộc thi này đều dùng mixup (trộn hai mẫu và hai nhãn theo tỉ lệ ngẫu nhiên).
- **Kết quả:** dùng riêng 0,753 (+0,003, chỉ 3/5 fold); kết hợp dropout 0,761 (thấp hơn dropout riêng) → **bỏ**.
- **Rút ra:** mixup và dropout cùng là cách chống overfit; khi đã có dropout thì mixup thừa và làm học chậm (một fold đến epoch 193 mới đạt đỉnh).

**3.5. Mô hình rộng hơn [64, 128, 256]**
- **Quan sát:** sau dropout, train chỉ còn 0,86 → mô hình đang bị giới hạn, có thể dùng thêm sức chứa mà không overfit.
- **Kết quả:** 0,765 → **0,774**, tốt hơn 4/5 fold → **giữ**.
- **Thử rộng hơn nữa [128, 256, 512]:** 0,776 (+0,002, nằm trong nhiễu) nhưng gấp ~4 lần tham số, train lên 0,97 → **bỏ**, không đáng.

**3.6. Mô hình sâu hơn** — *thất bại (2 lần)*
- **Lần 1:** thêm 1 khối [32, 64, 128, 256] → 0,763, **bỏ**. Giả thuyết: mỗi khối có MaxPool, 4 khối nén chuỗi 128 bước còn 8 → mất chi tiết chuyển động.
- **Lần 2 (kiểm tra giả thuyết trên):** sâu hơn nhưng không nén thêm — 2 lớp conv mỗi khối → 0,769 (3 seed), train lên 0,93, **bỏ**.
- **Rút ra:** giả thuyết "do nén thời gian" bị bác bỏ. Với lượng dữ liệu này, thêm độ sâu chỉ làm mô hình học thuộc nhiều hơn; tăng **chiều rộng** hiệu quả hơn.

**3.7. Masked pooling** — *thất bại*
- **Quan sát:** đo được phần đệm chiếm 47% tổng số bước. Global Average Pooling hiện lấy trung bình cả phần đệm.
- **Giả thuyết:** phần đệm làm "loãng" đặc trưng, và có thể là nguyên nhân BatchNorm thất bại.
- **Thử:** chỉ lấy trung bình trên các bước thật (đã kiểm tra mask khớp 100% với bước thật).
- **Kết quả:** 0,772 (không tăng) → **bỏ**.
- **Rút ra:** phần đệm không gây hại (có thể mô hình còn dùng nó như thông tin về độ dài chuỗi). Nguyên nhân BatchNorm thất bại vẫn chưa rõ.

**3.8. Kernel size**
- **Lý do:** kernel 5 là giá trị chọn ban đầu, chưa kiểm tra. Thử hai phía: 3 và 7.
- **Kết quả (1 seed):** kernel 3 = 0,778, kernel 5 = 0,774, kernel 7 = 0,771 → xu hướng kernel nhỏ tốt hơn.
- **Vấn đề:** chênh chỉ 0,004 → cần biết nhiễu do seed. Chạy thêm 2 seed:

| | Seed 42 | Seed 0 | Seed 1 | Trung bình |
|---|---|---|---|---|
| Kernel 5 | 0,7741 | 0,7703 | 0,7766 | 0,7737 |
| Kernel 3 | 0,7777 | 0,7757 | 0,7804 | **0,7779** |

- Kernel 3 thắng ở cả 3 seed → **giữ**.
- **Phát hiện quan trọng:** cùng cấu hình, đổi seed làm CV dao động tới **0,006** — lớn hơn nhiều so với ước tính ban đầu (0,0005, do 2 lần đo đầu tình cờ trùng nhau). Từ đây mọi thay đổi đều chạy 3 seed.
- **Rà lại các quyết định trước:** các thay đổi đã giữ đều tăng ≥ 0,009, vượt xa nhiễu → vẫn chắc chắn. Một số thay đổi đã bỏ (±0,002) nằm trong nhiễu → kết luận đúng là "không thấy lợi rõ", chứ không phải "có hại"; bỏ vì chúng làm mô hình phức tạp hơn mà không có lợi đã chứng minh.

**Bảng tổng hợp Bước 3** (1 seed trừ khi ghi khác):

| Thay đổi | CV val | Train | Số fold hơn nền | Quyết định |
|---|---|---|---|---|
| Nền: bỏ rot | 0,7495 | 0,90 | | |
| + BatchNorm | 0,7317 | 0,90 | 0/5 | Bỏ |
| + Dropout 0,3 | **0,7651** | 0,86 | 5/5 | Giữ |
| + Mixup 0,4 | 0,7527 | 0,88 | 3/5 | Bỏ |
| Nền: + dropout 0,3 | 0,7651 | 0,86 | | |
| Dropout 0,5 | 0,7455 | 0,80 | 0/5 | Bỏ |
| + Mixup 0,4 | 0,7607 | 0,85 | 2/5 | Bỏ |
| Rộng [64, 128, 256] | **0,7741** | 0,91 | 4/5 | Giữ |
| Sâu [32, 64, 128, 256] | 0,7631 | 0,89 | 2/5 | Bỏ |
| Nền: rộng [64, 128, 256] | 0,7741 | 0,91 | | |
| Rộng hơn [128, 256, 512] | 0,7761 | 0,97 | 4/5 | Bỏ |
| Masked pooling | 0,7719 | 0,94 | 2/5 | Bỏ |
| Kernel 7 | 0,7710 | 0,94 | 2/5 | Bỏ |
| Kernel 3 | **0,7779** (3 seed) | 0,89 | 4/5 | Giữ |
| 2 conv mỗi khối (so với kernel 3) | 0,7685 (3 seed) | 0,93 | | Bỏ |

### Bước 4: Tinh chỉnh huấn luyện (mỗi thay đổi 3 seed)

**4.1. Lịch học cosine** — *thất bại*
- **Lý do thử:** giảm dần learning rate về cuối thường giúp mô hình hội tụ tốt hơn. Cho chạy đủ 100 epoch (không dừng sớm), vì với patience 20 việc học sẽ dừng khi lr mới giảm ~20%, cosine gần như không có tác dụng.
- **Kết quả:** 0,773 (thua cả 3 seed). Train chỉ 0,87 → lr giảm quá sớm, mô hình chưa học đủ → **bỏ**.

**4.2. Augmentation: nhân hệ số + nhiễu** — *thất bại*
- **Quan sát:** vẫn còn overfit (train 0,89, val 0,78).
- **Giả thuyết:** mỗi người làm động tác mạnh/nhẹ khác nhau → nhân mỗi kênh với hệ số ngẫu nhiên quanh 1 và thêm nhiễu nhỏ sẽ giúp mô hình bớt phụ thuộc vào từng người.
- **Kết quả:** 0,776 (không tăng). Train vẫn 0,90 → augmentation không làm mô hình khó học thuộc hơn → **bỏ**.
- **Rút ra:** độ mạnh/nhẹ không phải khác biệt chính giữa các người.

**4.3. Augmentation: co giãn thời gian ±20%** — *thất bại*
- **Giả thuyết (rút từ 4.2):** nếu không phải độ mạnh thì có thể là **tốc độ** — mỗi người làm nhanh/chậm khác nhau.
- **Kết quả:** 0,772 (thua cả 3 seed) → **bỏ**.
- **Rút ra:** giả thuyết về tốc độ cũng không được ủng hộ.

**Bảng tổng hợp Bước 4:**

| Thay đổi so với nền kernel 3 | Trung bình 3 seed | Train | Quyết định |
|---|---|---|---|
| Nền | 0,7779 | 0,89 | |
| Cosine LR, 100 epoch | 0,7734 | 0,87 | Bỏ |
| Augmentation: nhân hệ số + nhiễu (0,1) | 0,7764 | 0,90 | Bỏ |
| Time stretch ±20% | 0,7719 | 0,85 | Bỏ |

**Quyết định dừng:** 4 lượt liên tiếp (masked pooling, cosine, 2 kiểu augmentation, độ sâu) không cải thiện. Tiếp tục tinh chỉnh nhỏ thì nhiều khả năng chỉ là chọn theo nhiễu → **kết luận mô hình đã chạm trần ~0,778, chốt cấu hình**.

### Bước 5: Chốt và đánh giá cuối
- **Ensemble 5 seed:** lấy trung bình xác suất của 5 mô hình với seed khác nhau. Lý do: đã đo được mỗi seed cho kết quả lệch nhau ~0,006, tức mỗi mô hình sai hơi khác nhau → trung bình lại sẽ ổn định hơn.
- Cấu hình cuối được quyết định **trước khi** nhìn holdout, rồi chấm holdout 1 lần.

## 5. Kết quả cuối

**Cấu hình**
- Input: `lin_acc_x/y/z, rot_missing, grav_x/y/z, angvel_x/y/z, angvel_mag`, lật tay trái.
- Conv1d [64, 128, 256], kernel 3, ReLU, MaxPool(2), Dropout 0,3, Global Avg Pool, Linear 18.
- Adam lr 1e-3, label smoothing 0,1, batch 64, max_len 128, early stopping patience 20.
- 5 seed (42, 0, 1, 2, 3); holdout chấm 1 lần duy nhất.

| | Score | Binary F1 | Macro F1 |
|---|---|---|---|
| Base ban đầu | 0,693 | 0,947 | 0,439 |
| CV, 1 seed (trung bình 5 seed) | 0,779 | 0,973 | 0,586 |
| **OOF, ensemble 5 seed** | **0,790** | 0,976 | 0,605 |
| Holdout, 1 seed (5 mô hình fold) | 0,813–0,826 | 0,99 | 0,63–0,66 |
| Holdout, ensemble 5 seed (25 mô hình) | 0,829 | 0,994 | 0,665 |

**Holdout cao hơn CV — có đáng tin không?**
- Kiểm tra rò rỉ: không có người nào nằm ở cả dev và holdout; chuẩn hóa chỉ dùng thống kê của tập train → không rò rỉ.
- Lấy ngẫu nhiên 16 người trong dev (2000 lần), score dao động 0,76–0,81; 0,83 nằm trong nhóm 0,2% cao nhất → nhóm 16 người holdout **dễ hơn trung bình**, cộng thêm lợi ích từ việc lấy trung bình 25 mô hình.
- → **Điểm nên báo cáo: ~0,79** (tính trên 65 người, đáng tin hơn 16 người).

## 6. Kết luận

**Những gì có tác dụng** (tổng +0,10 so với Base):

| Thay đổi | Tăng | Dựa vào |
|---|---|---|
| angvel | +0,04 | Phân tích lỗi: BFRB giống nhau về tư thế → cần chuyển động |
| Dropout 0,3 | +0,016 | Khoảng cách train–val lớn |
| Ensemble 5 seed | +0,011 | Đo được các seed sai khác nhau |
| Bỏ rot | +0,01 | rot chứa hướng quay mặt ngẫu nhiên |
| Rộng hơn | +0,009 | Sau dropout, train giảm → còn chỗ tăng sức chứa |
| Label smoothing, kernel 3, grav | +0,004 mỗi cái | Mô hình quá tự tin; kernel chưa được kiểm tra; tư thế cổ tay |

**Bài học từ các lần thất bại:**
- Đặc trưng CNN tự tính được (jerk, độ lớn) không giúp; đặc trưng CNN khó tự tính (angvel từ quaternion) giúp rất nhiều.
- Kỹ thuật chống overfit không cộng dồn: đã có dropout thì mixup/augmentation không giúp thêm.
- Với dữ liệu nhỏ, tăng chiều rộng tốt hơn tăng độ sâu.
- Kỹ thuật phổ biến (BatchNorm, cosine, mixup) không phải lúc nào cũng hợp — phải kiểm tra trên dữ liệu của mình.
- Phải đo nhiễu trước khi kết luận: ước tính nhiễu ban đầu (0,0005) thấp hơn thực tế (0,006) hơn 10 lần.

**Hạn chế và hướng tiếp:**
- Điểm yếu còn lại: phân biệt 8 BFRB (macro F1 ~0,60). Với IMU, các cặp như véo cổ / gãi cổ gần như giống hệt nhau.
- Nguyên nhân BatchNorm thất bại và giả thuyết "bỏ rot vì hướng quay mặt" chưa được kiểm chứng trực tiếp.
- Hướng tiếp: thêm ToF/nhiệt độ (đo khoảng cách tay tới mặt/cổ) để phân biệt vị trí. Holdout đã dùng, nên từ giờ chỉ chọn cấu hình theo CV.
