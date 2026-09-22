Dưới đây là bảng tổng hợp chi tiết 3 bài báo theo tiến trình thời gian phát triển của hướng nghiên cứu **Personal Voice Activity Detection (Personal VAD / PVAD)**:

---

### 1. Năm 2019 – 2020: Đặt nền móng cho Personal VAD (Proof-of-Concept)

* **Tên bài báo:** *Personal VAD: Speaker-Conditioned Voice Activity Detection*

* **Tác giả & Đơn vị:** Shaojin Ding, Quan Wang, Shuo-yiin Chang, Li Wan, Ignacio Lopez Moreno (Google Inc., Texas A&M University).


* **Thời gian:** arXiv tháng 08/2019 (cập nhật v4 tháng 04/2020).



#### Nội dung chính

* Đề xuất bài toán **Personal VAD (PVAD)** đầu tiên nhằm phát hiện hoạt động giọng nói ở cấp độ khung hình (*frame-level*) chỉ dành riêng cho một người nói mục tiêu (*target speaker*).


* Đóng vai trò như bộ cổng lọc (*gating module*) cho hệ thống nhận dạng giọng nói on-device (streaming ASR), giúp thiết bị không bị kích hoạt ngoài ý muốn bởi người khác hoặc tiếng ồn xung quanh mà không bắt buộc phải dùng từ khóa đánh thức (*wake-word*).



#### Phát hiện & Hạn chế của phương pháp cũ

* Ghép nối đơn thuần mô hình VAD chuẩn và mô hình nhận dạng người nói (Speaker Verification - SV) độc lập (baseline Score Combination) hoạt động kém hiệu quả. Lý do: SV thường chạy trên cấp độ cửa sổ/đoạn (*window-based/segment-based*), không tối ưu cho frame-level và mô hình SV quá lớn để chạy liên tục on-device.



#### Những phát hiện mới & Phương án xử lý

* **Phân loại 3 lớp (3-class classification):** Mở rộng nhãn phân loại khung hình thành: Không phải tiếng nói (*non-speech - ns*), Tiếng nói của người mục tiêu (*target speaker speech - tss*), và Tiếng nói của người khác (*non-target speaker speech - ntss*).


* **4 kiến trúc thử nghiệm:**
1. *Score Combination (SC):* Chạy độc lập VAD và SV rồi nhân điểm (baseline).


2. *Score Conditioned Training (ST):* Nối điểm tương đồng cosine từ SV vào đặc trưng âm học.


3. *Embedding Conditioned Training (ET):* Trực tiếp nối d-vector embedding (256-dim) của người mục tiêu vào đặc trưng âm học. Cách này đóng vai trò như chưng cất tri thức (*knowledge distillation*), không cần chạy mô hình SV khi inference.


4. *Score and Embedding Conditioned Training (SET):* Nối cả điểm cosine và embedding.




* **Hàm mất mát Weighted Pairwise Loss (WPL):** Nhận thấy nhầm lẫn giữa *ns* và *ntss* ít nghiêm trọng hơn nhầm lẫn liên quan đến *tss*, nhóm tác giả giảm trọng số phạt lỗi giữa *<ns, ntss>* ($w_{<ns, ntss>} = 0.1$) thay vì dùng Cross Entropy thông thường.


* **Mô hình siêu nhẹ:** Sử dụng mạng 2 lớp LSTM (64 cells) kết hợp lượng tử hóa 8-bit (*int8 quantization*).



#### Kết quả

* Cấu hình **ET (Embedding-conditioned)** chỉ có khoảng **130K tham số (~130 KB sau lượng tử hóa)**, nhỏ hơn ~40 lần so với các hệ thống cần chạy SV trực tiếp.


* Đạt Average Precision (AP) cho lớp *tss* là **0.955** (không có nhiễu/MTR) và **0.916** (có MTR) khi dùng WPL, vượt trội so với baseline SC (0.886 / 0.777).


* Khi đánh giá trên tác vụ VAD thông thường (chỉ có người mục tiêu), PVAD đạt hiệu năng tương đương standard VAD (AP speech đạt 0.991 vs 0.992).



---

### 2. Năm 2022: Tối ưu hóa triển khai thực tế trên thiết bị (On-Device Production)

* **Tên bài báo:** *Personal VAD 2.0: Optimizing Personal Voice Activity Detection for On-Device Speech Recognition*

* **Tác giả & Đơn vị:** Shaojin Ding, Rajeev Rikhye, Qiao Liang, Yanzhang He, Quan Wang, Arun Narayanan, Tom O'Malley, Ian McGraw (Google LLC).


* **Thời gian:** Interspeech 2022 (arXiv tháng 04/2022, v3 tháng 06/2022).



#### Nội dung chính

* Giải quyết các rào cản đưa PVAD từ nghiên cứu lý thuyết vào sản phẩm thực tế (hệ thống ASR on-device luôn chạy nền): yêu cầu độ trễ cực thấp (*streaming*), tài nguyên tính toán giới hạn, và hoạt động tốt ngay cả khi **chưa/không đăng ký giọng nói** (*enrollment-less*).



#### Phát hiện & Hạn chế của phiên bản cũ (PVAD 1.0)

* Việc nối trực tiếp embedding với đặc trưng âm học ở tầng đầu vào bị giới hạn dung lượng biểu diễn do sự phân kỳ về phân phối và độ lớn của hai loại đặc trưng.


* PVAD 1.0 bắt buộc phải có giọng nói đăng ký trước; nếu người dùng bỏ qua bước đăng ký (*enrollment-less*), mô hình hoàn toàn thất bại (WER > 100%).


* Chưa tối ưu hóa triệt để độ trễ streaming trên môi trường production.



#### Những phát hiện mới & Phương án xử lý

* **Cải tiến điều biến Speaker Embedding (Modulation):**
* *FiLM (Feature-wise Linear Modulation):* Áp dụng biến đổi affine (scale $\gamma$ và shift $\beta$) lên đầu ra của các khối trích xuất đặc trưng dựa trên speaker embedding.


* *Speaker Pre-net:* Sử dụng một mạng pre-net nhỏ để trích xuất embedding trực tiếp từ âm thanh đầu vào, tính độ tương đồng cosine với enrolled embedding, rồi đưa điểm này qua FiLM.




* **Khung huấn luyện thích ứng Enrollment & Enrollment-less (Joint Training):**
* Trong quá trình huấn luyện, ngẫu nhiên (xác suất $p_0 = 0.2$) thay thế target speaker embedding bằng **vector 0**, đồng thời gộp nhãn *ntss* thành *tss*. Cách này buộc mô hình tự suy thoái về hoạt động như một Standard VAD thông thường khi không có profile người dùng.




* **Backbone Conformer & Tối ưu hóa Streaming:** Thay thế LSTM bằng Conformer thu nhỏ (4 layers, causal convolution, 31 left-context, không có right-context để triệt tiêu độ trễ tương lai) và áp dụng lượng tử hóa 8-bit dynamic range (TFLite).



#### Kết quả

* Đánh giá trực tiếp qua độ lỗi từ (Word Error Rate - WER) của hệ thống ASR trên dữ liệu thực tế:


* **Kịch bản hội thoại có nhiều người (Concat):** Giảm WER từ 41.0% (PVAD 1.0) xuống **27.2% - 32.7%** (PVAD 2.0), cắt giảm mạnh lỗi chèn từ (*insertion error*) do giọng người ngoài.


* **Kịch bản không có profile đăng ký (Enrollment-less):** Xử lý triệt để lỗi của PVAD 1.0, đạt WER ngang ngửa Standard VAD (7.0% trên Voice Search và 10.1% trên Non-Concat).




* Kích thước mô hình Conformer sau lượng tử hóa 8-bit chỉ còn **1.0 MB** (giảm 75% so với bản float 4.0 MB), tốc độ tính toán nhanh và phù hợp với phần cứng di động.



---

### 3. Năm 2026: Tự mở rộng Embedding khi dữ liệu đăng ký cực ngắn (Short Enrollment)

* **Tên bài báo:** *Adaptive Speaker Embedding Self-Augmentation for Personal Voice Activity Detection with Short Enrollment Speech*

* **Tác giả & Đơn vị:** Fuyuan Feng, Wenbin Zhang, Yu Gao, Longting Xu, Xiaofeng Mou, Yi Xu (Đại học Đông Hoa & Trung tâm Nghiên cứu AI Midea Group).


* **Thời gian:** arXiv tháng 01/2026.



#### Nội dung chính

* Giải quyết thách thức lớn trong triển khai thực tế (như nhà thông minh): **Mẫu giọng đăng ký quá ngắn** (chỉ là từ khóa đánh thức 0.5s – 1.5s) chứa rất ít thông tin đặc trưng của người nói, cộng thêm độ lệch miền (*domain mismatch*) giữa âm thanh đăng ký cự ly gần (điện thoại) và âm thanh mic trường xa (*far-field*).



#### Phát hiện & Hạn chế của các phương pháp trước

* Khi độ dài file đăng ký giảm từ 1.5s xuống 0.5s, hiệu năng của PVAD 2.0 tụt giảm nghiêm trọng (F1-score từ ~86% tụt xuống ~80% trong môi trường sạch và còn ~76% khi có nhiễu).


* Các nghiên cứu trước xem embedding đăng ký là một vector tĩnh bất biến, không thích ứng được với sự biến thiên trạng thái giọng nói theo thời gian (*temporal vocal dynamics*) trong các phiên hội thoại dài.



#### Những phát hiện mới & Phương án xử lý

* **Speaker Embedding Self-Augmentation (Tự tăng cường từ luồng âm thanh hỗn hợp):**
* Tận dụng sự thật rằng giọng của người mục tiêu đã có sẵn trong âm thanh hỗn hợp đầu vào.


* Dùng cửa sổ trượt dài (1s, bước nhảy 0.2s) và encoder pre-trained CAM++ trích xuất embedding khung hình từ âm thanh hỗn hợp, tìm khung hình có độ tương đồng cosine cao nhất với $E_{enroll}$ (gọi là keyframe embedding $E_{selected}$).


* Hợp nhất $E_{enroll}$ và $E_{selected}$ bằng phép cộng (*additive fusion*: $E_{augmented} = E_{enroll} + E_{selected}$). Kết quả chứng minh phép cộng bảo toàn phân phối tốt hơn và cho hiệu quả cao hơn phép nối (*concatenation*).




* **Thích ứng dài hạn có hiệu chỉnh phần dư (Long-term Adaptation with Residual Correction):**
* Nếu tiếp tục cộng dồn trực tiếp qua nhiều segment (1 đến 10), lỗi và nhiễu (giọng người khác bị chọn nhầm) sẽ tích lũy khiến Precision và F1 lao dốc.


* Đề xuất công thức cập nhật lặp có neo giữ embedding gốc:

$$E_{avg\_add} = 0.5(E_{augmented\_n-1} + E_{selected})$$



$$E_{augmented\_n} = \lambda E_{enroll} + (1 - \lambda) E_{avg\_add}$$




*(với $\lambda \in [0.05, 0.1]$ đóng vai trò giữ đặc trưng chuẩn từ file đăng ký ban đầu)*.





#### Kết quả

* **Cải thiện ngoạn mục ở mẫu đăng ký ngắn:** Ở mẫu 0.5s trong môi trường có nhiễu, việc tự tăng cường ở phân đoạn đầu tiên giúp tăng F1-score từ 76.35% lên 78.90% (Segment 1) và đạt 81.32% (Segment 2).


* **Hiệu quả tương đương mẫu dài:** Sau 5 vòng lặp cập nhật ($E_{augmented\_5}$), mô hình chỉ cần mẫu đăng ký **0.5s** ban đầu nhưng đạt hiệu năng ngang ngửa việc có mẫu đăng ký đầy đủ chuẩn **7.4s**.


* Giải quyết triệt để sự nhầm lẫn giữa những người nói cùng giới tính (*same-gender speaker confusion*) trong môi trường nhiễu mà **không cần tăng thêm tham số mô hình PVAD**.



---
