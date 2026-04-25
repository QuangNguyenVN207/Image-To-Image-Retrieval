# Image-To-Image-Retrieval

Image-to-Image Retrieval nhận đầu vào là một hình ảnh truy vấn và nhiệm vụ của hệ thống là tìm ra các hình ảnh tương đồng trong bộ dữ liệu có trước, dựa trên các đặc trưng thị giác như màu sắc, hình dạng, kết cấu hoặc đặc trưng ngữ nghĩa ở mức cao.

Đây là một bài toán có tính thực tiễn cao và được ứng dụng rộng rãi trong thực tế. Hiện nay, nhiều công cụ tìm kiếm hình ảnh như Google Images, Pinterest hay các hệ thống gợi ý sản phẩm trong thương mại điện tử đều khai thác bài toán này nhằm hỗ trợ người dùng tìm kiếm thông tin một cách trực quan và hiệu quả.

Trong project này, ta tập trung vào mức độ đơn giản nhất của bài toán, cụ thể là phương pháp texture-based, trong đó sự tương đồng giữa các hình ảnh được đánh giá chủ yếu dựa trên các đặc trưng mức thấp như kết cấu và phân bố cường độ pixel.

# Cách 1: So sánh độ tương đồng dựa trên pixel

Một hình ảnh màu có thể được biểu diễn dưới dạng ba ma trận hai chiều tương ứng với ba kênh màu RGB, trong đó mỗi phần tử có giá trị nằm trong khoảng [0,255]. Do đó, cách tiếp cận đơn giản nhất để đánh giá mức độ tương đồng giữa hai hình ảnh là so sánh trực tiếp giá trị các pixel tại cùng vị trí không gian.
Cụ thể, hai hình ảnh được đưa về cùng kích thước, sau đó độ tương đồng được tính toán bằng cách so sánh các giá trị pixel tương ứng trên từng kênh màu. Các phép đo khoảng cách phổ biến như L1 distance hoặc L2 distance có thể được sử dụng để lượng hóa mức độ khác biệt giữa hai hình ảnh.

# Cách 2: Feature extraction (trích xuất đặc trưng)

Ta dễ dàng nhận thấy rằng phương pháp truy vấn ảnh dựa trên việc so sánh trực tiếp các giá trị pixel trong không gian ảnh không phải là một cách làm tối ưu.

Trong bối cảnh đó, thay vì so sánh ảnh tại mức pixel, ta có thể so sánh các đặc trưng (features) mang tính tổng quát hơn của ảnh. Các đặc trưng này có thể biểu diễn thông tin về màu sắc, kết cấu (texture), hình dạng (shape) hoặc các thuộc tính thị giác quan trọng khác. Quá trình trích xuất những thông tin này từ ảnh đầu vào được gọi là feature extraction.

Trong số các phương pháptruyền thống, một kỹ thuật phổ biến và trực quan là Color Histogram, trong đó ảnh được biểu diễn thông qua phân bố tần suất của các giá trị màu sắc, thay vì phụ thuộc vào vị trí cụ thể của từng pixel.

Color Histogram là một phương pháp trích xuất đặc trưng ảnh dựa trên phân bố tần suất của các giá trị màu sắc trong ảnh. Thay vì quan tâm đến vị trí cụ thể của từng pixel, phương pháp này chỉ xét xem mỗi mức màu xuất hiện bao nhiêu lần

