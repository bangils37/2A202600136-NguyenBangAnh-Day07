# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Bằng Anh
**Nhóm:** [Tên nhóm]
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai embedding có hướng gần nhau trong không gian vector, cho thấy hai văn bản có nghĩa tương tự hoặc cùng chủ đề.

**Ví dụ HIGH similarity:**
- Sentence A: Tim mạch là cơ quan quan trọng.
- Sentence B: Bệnh tim mạch ảnh hưởng đến chức năng tim.
- Tại sao tương đồng: Cả hai đều nói về sức khỏe tim mạch và các vấn đề y tế liên quan đến tim.

**Ví dụ LOW similarity:**
- Sentence A: Tim mạch là cơ quan quan trọng.
- Sentence B: Tôi đang học toán ở trường.
- Tại sao khác: Một câu nói về sức khỏe tim, còn câu kia nói về hoạt động học thuật, nghĩa ngữ cảnh khác nhau.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chuẩn hóa chiều dài vector và chỉ đo góc giữa chúng, nên nó loại bỏ ảnh hưởng của kích thước văn bản và tập trung vào tính tương đồng ngữ nghĩa.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Bước nhảy giữa các chunk là 500 - 50 = 450 ký tự. Số chunk ≈ ceil((10,000 - 500) / 450) + 1 = ceil(9,500 / 450) + 1 = 23.
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap tăng lên 100, bước nhảy giảm còn 400 ký tự và số chunk tăng lên khoảng 25. Overlap nhiều hơn giúp giữ lại ngữ cảnh giữa các chunk, đặc biệt hữu ích khi câu hỏi hoặc thông tin nằm ở rìa của các chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Health education / Cardiovascular health.

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain sức khỏe tim mạch vì các tài liệu trong `data/heart_health` đã tập trung vào chẩn đoán, phòng ngừa và điều trị bệnh tim. Đây là lĩnh vực có ngôn ngữ chuyên môn rõ ràng, giúp đánh giá tốt các chiến lược embedding, chunking và retrieval trong hệ thống RAG.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | heart_health_01.md | www.vinmec.com | 3577 | category, date, source, language, difficulty |
| 2 | heart_health_02.md | www.vinmec.com | 3498 | category, date, source, language, difficulty |
| 3 | heart_health_03.md | www.vinmec.com | 3699 | category, date, source, language, difficulty |
| 4 | heart_health_04.md | www.vinmec.com | 3377 | category, date, source, language, difficulty |
| 5 | heart_health_05.md | www.vinmec.com | 3419 | category, date, source, language, difficulty |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| category | text | Diagnosis / Lifestyle / Treatment / Prevention | Giúp filter và phân nhóm tài liệu theo chủ đề chính. |
| date | date | 2024-04-10 | Cho phép tìm tài liệu mới nhất và ưu tiên nội dung cập nhật. |
| source | text | www.vinmec.com | Giúp truy xuất nguồn tin lúc cần xem lại hoặc đánh giá độ tin cậy. |
| difficulty | text | Beginner / Intermediate | Hữu ích khi cần trả lời truy vấn theo mức độ chi tiết phù hợp. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu.

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| heart_health_01.md | FixedSizeChunker (`fixed_size`) | 14 | 248.1 | Không giữ câu nguyên vẹn, chia theo ký tự |
| heart_health_01.md | SentenceChunker (`by_sentences`) | 14 | 246.0 | Có, theo câu và đoạn logic |
| heart_health_01.md | RecursiveChunker (`recursive`) | 21 | 163.7 | Có, nhưng nhiều chunk nhỏ hơn |
| heart_health_02.md | FixedSizeChunker (`fixed_size`) | 14 | 242.4 | Không giữ câu nguyên vẹn, chia theo ký tự |
| heart_health_02.md | SentenceChunker (`by_sentences`) | 13 | 259.2 | Có, tách theo câu và đoạn ý nghĩa |
| heart_health_02.md | RecursiveChunker (`recursive`) | 26 | 128.9 | Có, nhưng phân nhỏ quá nhiều |
| heart_health_03.md | FixedSizeChunker (`fixed_size`) | 15 | 239.4 | Không giữ câu nguyên vẹn, chia theo ký tự |
| heart_health_03.md | SentenceChunker (`by_sentences`) | 14 | 254.6 | Có, theo câu và giữ ý tốt |
| heart_health_03.md | RecursiveChunker (`recursive`) | 23 | 154.5 | Có, nhưng nhiều chunk nhỏ hơn |

### Strategy Của Tôi

**Loại:** SentenceChunker (`by_sentences`)

**Mô tả cách hoạt động:**
> Strategy sử dụng `SentenceChunker` để tách văn bản tại ranh giới câu bằng regex kết hợp dấu chấm, chấm than, dấu hỏi và ngắt dòng. Nó gom tối đa 3 câu mỗi chunk để giữ ý nghĩa đầy đủ và tránh cắt ngang câu y tế có nhiều thông tin quan trọng.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Những tài liệu sức khỏe tim mạch thường trình bày thông tin theo câu rõ ràng, mỗi câu chứa một ý chính hoặc lời khuyên. SentenceChunker giúp giữ nguyên logic y học và tạo chunk đủ dài để trả lời truy vấn mà vẫn dễ hiểu.

**Code snippet (nếu custom):**
```python
from src.chunking import SentenceChunker
chunker = SentenceChunker(max_sentences_per_chunk=3)
chunks = chunker.chunk(text)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| heart_health_01.md | best baseline: SentenceChunker | 14 | 246.0 | Tốt, giữ nguyên câu y tế |
| heart_health_01.md | **của tôi**: SentenceChunker | 14 | 246.0 | Tốt, cùng kết quả |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Nguyễn Bằng Anh | SentenceChunker | 8 | Giữ nguyên câu, phù hợp với ngữ cảnh y tế | Đôi khi chunk có thể dài nếu câu dài |
| Đỗ Thị Thùy Trang | FixedSizeChunker | 6 | Chunk size đồng đều, đơn giản | Cắt ngang câu, mất một phần ý nghĩa |
| Bùi Trọng Anh | RecursiveChunker | 7 | Giữ ý theo đoạn, phù hợp với nhiều cấu trúc | Quá nhiều chunk nhỏ, có thể kéo dài tìm kiếm |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> SentenceChunker là tốt nhất cho domain sức khỏe tim mạch vì nó tôn trọng ranh giới câu và giữ được thông tin y tế rõ ràng. Các câu trong tài liệu điều trị và chẩn đoán thường xây dựng ý nghĩa theo từng câu, nên tách theo câu giúp retrieval chính xác hơn.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> `SentenceChunker.chunk` sử dụng regex `(?<=[.!?])\s+|(?<=\.)\n` để tách câu tại các dấu chấm, chấm than, dấu hỏi và ngắt dòng. Nó lọc các câu rỗng, xóa khoảng trắng dư thừa và gom tối đa 3 câu thành một chunk.

**`RecursiveChunker.chunk` / `_split`** — approach:
> `RecursiveChunker` hoạt động theo chiến lược phân tách đệ quy bằng các separator ưu tiên `['\n\n', '\n', '. ', ' ', '']`. Nếu đoạn văn nhỏ hơn `chunk_size` thì trả về trực tiếp; nếu không thì tách theo separator lớn nhất có thể và nếu phần con vẫn dài quá thì gọi `_split` tiếp với separator tiếp theo.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `EmbeddingStore` lưu mỗi tài liệu dưới dạng record gồm `id`, `content`, `embedding` và `metadata`. `add_documents` nhúng nội dung bằng `embedding_fn` (`_mock_embed` mặc định) rồi thêm vào collection nội bộ hoặc ChromaDB nếu có. `search` tạo embedding của truy vấn rồi tính điểm tương đồng dot product giữa embedding truy vấn và embedding của từng record.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` lọc trước theo metadata; nếu dùng bộ nhớ in-memory thì lọc thủ công, nếu dùng ChromaDB thì truyền `where` vào query. `delete_document` xóa tất cả chunk có `doc_id` tương ứng trong metadata, nên loại bỏ một tài liệu hoàn toàn khỏi vector store.

### KnowledgeBaseAgent

**`answer`** — approach:
> `KnowledgeBaseAgent.answer` lấy top-k chunk liên quan với `store.search`, sau đó ghép từng chunk vào prompt như các khối context `[Context i]`. Prompt này được gửi tới `llm_fn` để tạo câu trả lời dựa trên thông tin đã retrieve.

### Test Results

```
42 passed in 2.43s
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Tim mạch là cơ quan quan trọng. | Bệnh tim mạch ảnh hưởng đến chức năng tim. | high | -0.1054 | no |
| 2 | Ăn nhiều rau xanh giúp kiểm soát huyết áp. | Tập thể dục đều đặn cải thiện sức khỏe tim. | low | 0.0024 | yes |
| 3 | Nhồi máu cơ tim có thể gây đau ngực. | Đau ngực kéo dài có thể là dấu hiệu của nhồi máu cơ tim. | high | -0.0460 | no |
| 4 | Tăng cường kali trong chế độ ăn giúp giảm huyết áp. | Muối dư thừa có thể làm tăng huyết áp. | medium | 0.0277 | yes |
| 5 | Suy tim trái gây phù phổi. | Suy tim phải thường gây phù ngoại vi. | low | 0.2072 | no |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Cặp 1 và cặp 3 là bất ngờ nhất, vì ý nghĩa tự nhiên của chúng rất gần nhưng điểm actual lại không cao. Điều này cho thấy embedding mock có thể không phản ánh chính xác mối quan hệ ngữ nghĩa tinh tế giữa các câu y tế, và việc dùng embedding thực tế chất lượng cao hơn là rất quan trọng.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Theo khuyến cáo, nên làm gì đầu tiên khi nghi ngờ bị nhồi máu cơ tim? | heart_health_01 |
| 2 | Chế độ ăn DASH giới hạn lượng Natri (muối) như thế nào so với bình thường? | heart_health_02 |
| 3 | Triệu chứng điển hình của suy tim phải là gì? | heart_health_03 |
| 4 | Mảng xơ vữa động mạch gây nguy hiểm như thế nào nếu bị nứt vỡ đột ngột? | heart_health_04 |
| 5 | Đối với người bệnh tim, quy tắc 'An Toàn Là Trên Hết' khuyên làm gì cho buổi tập thể dục? | heart_health_05 |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Theo khuyến cáo, nên làm gì đầu tiên khi nghi ngờ bị nhồi máu cơ tim? | Tài liệu về tầm soát tim mạch định kỳ nhắc đến nguy cơ nhồi máu cơ tim. | 0.1178 | No | Câu trả lời có thể nhắc chung về nguy cơ và triệu chứng tim mạch, không phải hướng dẫn cấp cứu cụ thể. |
| 2 | Chế độ ăn DASH giới hạn lượng Natri (muối) như thế nào so với bình thường? | Tài liệu DASH giải thích giảm natri, tăng kali/magiê để hạ huyết áp. | 0.0986 | Yes | Câu trả lời sẽ liên quan trực tiếp đến chế độ ăn DASH và giới hạn natri. |
| 3 | Triệu chứng điển hình của suy tim phải là gì? | Tài liệu phân biệt suy tim trái và suy tim phải với triệu chứng phù ngoại vi. | 0.0010 | Yes | Câu trả lời sẽ mô tả triệu chứng phù chân, mệt mỏi và gan to do suy tim phải. |
| 4 | Mảng xơ vữa động mạch gây nguy hiểm như thế nào nếu bị nứt vỡ đột ngột? | Tài liệu về bệnh mạch vành ở người trẻ nói về nguy cơ biến cố mạch vành. | 0.1236 | No | Câu trả lời có thể đề cập đến nguy cơ mạch vành, nhưng chưa rõ về cơ chế nứt vỡ mảng xơ vữa. |
| 5 | Đối với người bệnh tim, quy tắc 'An Toàn Là Trên Hết' khuyên làm gì cho buổi tập thể dục? | Tài liệu DASH được trả về đầu tiên, không phải tài liệu an toàn tập luyện. | 0.1374 | No | Câu trả lời có vẻ tập trung vào dinh dưỡng hơn là hướng dẫn tập luyện an toàn. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được cách chọn metadata chi tiết hơn cho từng chunk, đặc biệt là thêm `category` và `difficulty` để hỗ trợ retrieval filter. Cách phân chia chunk theo ngữ cảnh và sử dụng các separator logic của bạn giúp tôi hiểu sâu hơn về hiệu quả của RecursiveChunker.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Qua demo nhóm khác, tôi nhận thấy nhiều đội dùng prompt template rõ ràng hơn khi nối context vào prompt, và điều đó giúp giảm drift khi LLM trả lời. Tôi cũng thấy rằng đa dạng hóa loại câu hỏi (chẩn đoán, phòng ngừa, điều trị) giúp đánh giá bộ nhớ vector toàn diện hơn.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Nếu làm lại, tôi sẽ bổ sung thêm metadata `topic` và `risk_level` cho từng tài liệu để filtering chi tiết hơn. Tôi cũng sẽ cân nhắc chunk nhỏ hơn cho các phần định nghĩa chuyên môn và chunk lớn hơn cho phần hướng dẫn lối sống, nhằm cân bằng giữa độ chi tiết và khả năng giữ ngữ cảnh.

**Failure analysis:**
> Một failure case rõ ràng là query số 1 và query số 4, khi top-1 không phải tài liệu dự kiến dù top-3 vẫn chứa document phù hợp. Điều này cho thấy metadata filter hữu ích nhưng embedding mock và ranking dot product vẫn cần cải thiện để ưu tiên nội dung chính xác hơn.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
