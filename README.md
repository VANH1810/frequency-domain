## FrequencyDomain – Mô tả các file Python

- **[extract_feature.py](./extract_feature.py)**:
  - Đọc CSV pose/bbox theo từng PID, dùng `WaveletFeatureExtractor` để trích xuất đặc trưng theo thời gian (khoảng cách, góc, bbox), lưu các mảng `.npy` và hệ số DWT vào `.pkl` theo PID.
  - Đầu vào: CSV có cột `PID`, `FrameID`, `Bbox`, `Pose`; cấu hình kích thước khung hình.

- **[frequency_pipeline.py](./frequency_pipeline.py)**:
  - Thư viện trung tâm cho trích xuất đặc trưng miền thời gian/tần số từ skeleton: chuẩn hoá keypoint theo bbox, tính distances/angles/bbox features, PSD, DWT, CWT, năng lượng theo thang, tần số trội, và tập đặc trưng gait rút gọn theo khung.
  - Cung cấp lớp `WaveletFeatureExtractor` và nhiều hàm vẽ/tiện ích để trực quan hoá (time-series, scalogram...).

- **[Running_Walking_ML.py](./Running_Walking_ML.py)**:
  - Pipeline phân loại Run/Walk theo cửa sổ trượt dùng đặc trưng tần số wavelet hoặc bộ đặc trưng WR 10 kênh.
  - Xây dataset từ thư mục đặc trưng, train One-Class SVM cho Walk, dự đoán, hậu xử lý (quy chiếu frame theo CSV pose, luật tốc độ thấp, láng giềng), xuất CSV kết quả và đánh giá.

- **[infer_gcn.py](./infer_gcn.py)**:
  - Suy luận hành vi dựa trên GCN ONNX (`EfficentGCN-B4_asilla_secom_20250521.onnx`) từ keypoints COCO-17 → ánh xạ 14 khớp, chuẩn hoá theo frame và cửa sổ trượt.
  - Xuất CSV dự đoán cho mỗi video (nhãn gốc và nhóm Falling/Staggering/Other).

- **[infer_rtm.py](./infer_rtm.py)**:
  - Chạy phát hiện người (RTMDet qua Triton), ước lượng pose (RTMPose), tracking (ByteTrack), ghép kết quả theo frame để ghi ra CSV `FrameID, PID, Bbox, Pose` cho toàn bộ video trong thư mục.

- **[infer_rtmpose.py](./infer_rtmpose.py)**:
  - Quy trình tương tự `infer_rtm.py` nhưng đọc danh sách video cố định; minh hoạ pipeline detect→crop→pose→track và ghi CSV.

- **[infer_rtmpose2.py](./infer_rtmpose2.py)**:
  - Biến thể tối ưu hoá cho data dạng `recorded_data` (robot-dog), clamp bbox theo kích thước khung, trích crop từ track rồi ước lượng pose hàng loạt, ghi CSV ra thư mục `infer-robot-dog`.

- **[eval.py](./eval.py)**:
  - Đọc các CSV dự đoán GCN đã gộp, quy chiếu với ground-truth (Toyota v1–v4), tạo file đánh giá per-video và tính báo cáo tổng hợp (accuracy, classification report). Có logic ép nhãn Falling khi video có cả dự đoán và nhãn thật Falling.

- **[gcn_eval.py](./gcn_eval.py)**:
  - Đánh giá kết quả suy luận từ mô hình GCN: đọc dự đoán đã xuất, so khớp với ground-truth theo video/PID/frame-range, tính và in các chỉ số (accuracy, classification report, confusion matrix).

- **[RF_FrequencyDomain.py](./RF_FrequencyDomain.py)**:
  - Phân loại Run/Walk theo cửa sổ dùng đặc trưng wavelet-only (năng lượng theo băng, tỷ lệ năng lượng, entropy, sparsity, big-fraction...) trên các kênh WR; build dataset cửa sổ, train One-Class SVM (chuẩn hoá), dự đoán, xuất CSV và đánh giá (có thể dùng event-level metrics).

- **[visualization_freqeuency_features.py](./visualization_freqeuency_features.py)**:
  - Vẽ overlay video từ CSV pose/bbox và (tuỳ chọn) xuất các biểu đồ đặc trưng wavelet theo PID.

- **[visualization_gait_features.py](./visualization_gait_features.py)**:
  - Vẽ/ghi video có khung/bộ xương và khối thông tin dự đoán/nhãn thật trên từng bbox; hỗ trợ 14 hoặc ≥17 keypoints; dựng map dự đoán/gold theo từng frame từ CSV kết quả và CSV ground-truth.

- **[visualization_running_walking.py](./visualization_running_walking.py)**:
  - Duyệt thư mục video/csv và vẽ plots time-series cho bộ đặc trưng gait 10 kênh (WR) theo từng PID, xuất ảnh; có khung annotate video (comment sẵn).

- **[gait_features.py](./gait_features.py)**:
  - Tính 11 nhóm đặc trưng gait từ chuỗi skeleton: tốc độ chuẩn hoá, góc đầu/thân, độ rộng vai/nhún vai, phối hợp/văng tay, tần số/ tốc độ vung tay, cadence/độ dài bước, góc gối, dao động dọc, jerk hông, proxy động năng; kịch bản batch demo và lưu CSV.

- **[extract_time_series_feature.py](./extract_time_series_feature.py)**:
  - Trích xuất các đặc trưng time-series WR (10 kênh) cho từng PID từ CSV pose bất kỳ: parse `POSES`/`SCORE`, tự tính `bbox` từ pose, chạy `WaveletFeatureExtractor.extract_wr_frame_features`, lưu `wr_feats(.npy/.csv)` và tên kênh.

- **[main.py](./main.py)**:
  - Script tiện ích: quét thư mục CSV, với mỗi file trích DWT theo PID bằng `WaveletFeatureExtractor` và lưu JSON hệ số DWT có cấu trúc rõ ràng.

- **[rename.py](./rename.py)**:
  - Đổi tên file CSV trong cây `data/client_v4_video/5FPS_all_csv` bằng cách bỏ đuôi trung gian `.MP4.csv` → `.csv`.

- **[test.py](./test.py)**:
  - Hợp nhất các file dự đoán GCN từ nhiều nguồn vào thư mục `merged_predictions`: copy theo danh sách ưu tiên và sao chép giữ nguyên cấu trúc; in log tiến trình.



