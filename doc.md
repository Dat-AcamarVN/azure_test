# Tài liệu Kiến trúc Tìm kiếm Patent

## Tổng quan
Hệ thống patent này hỗ trợ nhiều loại tìm kiếm khác nhau, từ tìm kiếm văn bản cơ bản đến tìm kiếm vector và semantic nâng cao. Mỗi loại tìm kiếm được thiết kế để xử lý các trường hợp sử dụng cụ thể và cung cấp kết quả tối ưu.

## 1. BM25 Search (Basic Text Search)

### Cấu trúc
```python
def bm25_search(search_endpoint: str, search_key: str, search_index_name: str,
                query: str, limit: int = 10) -> List[Dict[str, Any]]
```

### Luồng hoạt động
1. **Khởi tạo**: Tạo Azure AI Search client với endpoint và key
2. **Tìm kiếm**: Sử dụng thuật toán BM25 (Best Matching 25) để tìm kiếm văn bản
3. **Xử lý kết quả**: 
   - Lấy top N kết quả theo tham số `limit`
   - Chọn các trường cần thiết: `patent_id`, `title`, `abstract`, `claims`, `patent_office`, `created_at`, `updated_at`
4. **Chuyển đổi**: Chuyển đổi kết quả thành đối tượng `PatentInfo`
5. **Trả về**: Danh sách kết quả với score BM25

### Ưu điểm
- Nhanh và hiệu quả cho tìm kiếm văn bản cơ bản
- Sử dụng thuật toán BM25 đã được tối ưu hóa
- Không cần xử lý vector

### Sử dụng khi nào
- Tìm kiếm từ khóa đơn giản
- Cần kết quả nhanh
- Tìm kiếm trong title, abstract, claims

---

## 2. Vector Search

### Cấu trúc
```python
def vector_search(search_endpoint: str, search_key: str, search_index_name: str,
                  query_text: str, vector_field: str = "combined_vector",
                  similarity_threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD, 
                  limit: int = 10) -> List[Dict[str, Any]]
```

### Luồng hoạt động
1. **Tạo embedding**: Sử dụng Azure OpenAI để chuyển đổi query thành vector
2. **Tìm kiếm vector**: 
   - Sử dụng thuật toán HNSW (Hierarchical Navigable Small World)
   - Tìm k-nearest neighbors trong không gian vector
3. **Lọc kết quả**: Áp dụng ngưỡng similarity để loại bỏ kết quả không phù hợp
4. **Xử lý kết quả**: Chuyển đổi và trả về với similarity score

### Cấu hình Vector Search
```python
# Trong Azure AI Search Index
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="myHnswAlgo",
            kind="hnsw",
            parameters=HnswParameters(m=4, ef_construction=400)
        )
    ],
    profiles=[
        VectorSearchProfile(
            name="myHnswProfile",
            algorithm_configuration_name="myHnswAlgo"
        )
    ]
)
```

### Ưu điểm
- Tìm kiếm semantic chính xác
- Hiểu được ý nghĩa của query
- Không phụ thuộc vào từ khóa chính xác

### Sử dụng khi nào
- Tìm kiếm theo ý nghĩa
- Cần kết quả có độ tương tự cao
- Tìm kiếm trong nội dung phức tạp

---

## 3. Hybrid Search

### Cấu trúc
```python
def hybrid_search(search_endpoint: str, search_key: str, search_index_name: str,
                  query: str, vector_field: str = "combined_vector",
                  similarity_threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD, 
                  limit: int = 10,
                  text_weight: float = config.HYBRID_SEARCH_TEXT_WEIGHT,
                  vector_weight: float = config.HYBRID_SEARCH_VECTOR_WEIGHT) -> List[Dict[str, Any]]
```

### Luồng hoạt động
1. **Tạo embedding**: Chuyển đổi query thành vector
2. **Tìm kiếm kết hợp**: 
   - Thực hiện tìm kiếm văn bản (BM25)
   - Thực hiện tìm kiếm vector (HNSW)
   - Azure AI Search tự động kết hợp hai loại tìm kiếm
3. **Tính toán score**: 
   - Text score từ BM25
   - Vector score từ similarity
   - Combined score = text_score × text_weight + vector_score × vector_weight
4. **Lọc và sắp xếp**: Áp dụng ngưỡng và sắp xếp theo combined score

### Cấu hình Weight
```python
# Trong config.py
HYBRID_SEARCH_TEXT_WEIGHT = 0.3      # Trọng số cho text search
HYBRID_SEARCH_VECTOR_WEIGHT = 0.7    # Trọng số cho vector search
```

### Ưu điểm
- Kết hợp ưu điểm của cả text và vector search
- Kết quả cân bằng giữa độ chính xác và relevance
- Linh hoạt trong việc điều chỉnh trọng số

### Sử dụng khi nào
- Cần kết quả tối ưu nhất
- Có thể điều chỉnh trọng số theo nhu cầu
- Tìm kiếm phức tạp với nhiều tiêu chí

---

## 4. Semantic Search

### Cấu trúc
```python
def semantic_search(search_endpoint: str, search_key: str, search_index_name: str,
                    query: str, vector_field: str = "combined_vector", 
                    limit: int = 10) -> List[Dict[str, Any]]
```

### Luồng hoạt động
1. **Tạo embedding**: Chuyển đổi query thành vector
2. **Tìm kiếm semantic**: 
   - Kết hợp tìm kiếm văn bản và vector
   - Sử dụng semantic understanding của Azure AI Search
3. **Xử lý kết quả**: Trả về với semantic score

### Ưu điểm
- Hiểu được ngữ cảnh và ý nghĩa
- Kết quả có độ chính xác cao
- Tự động xử lý synonyms và related terms

### Sử dụng khi nào
- Tìm kiếm theo ngữ cảnh
- Cần hiểu ý nghĩa sâu sắc
- Tìm kiếm trong domain chuyên môn

---

## 5. RRF Hybrid Search (Reciprocal Rank Fusion)

### Cấu trúc
```python
def search_rrf_hybrid(search_endpoint: str, search_key: str, search_index_name: str,
                      query: str, vector_field: str = "combined_vector",
                      similarity_threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD, 
                      limit: int = 10,
                      k: int = config.DEFAULT_RRF_K) -> List[Dict[str, Any]]
```

### Luồng hoạt động
1. **Tìm kiếm song song**: 
   - Thực hiện text search (BM25)
   - Thực hiện vector search
2. **Tính toán RRF score**: 
   - RRF(d) = 1/(k + r_text) + 1/(k + r_vector)
   - Trong đó r là rank của document trong mỗi loại tìm kiếm
3. **Kết hợp kết quả**: Sắp xếp theo RRF score giảm dần
4. **Trả về**: Top N kết quả với RRF score

### Công thức RRF