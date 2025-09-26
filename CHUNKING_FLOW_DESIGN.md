# Chunking Flow Design Overview

## Tổng quan hệ thống

Hệ thống Patent Search sử dụng **Azure Cosmos DB** làm database chính và **Azure AI Search** cho các phương thức tìm kiếm nâng cao. Khi áp dụng chunking, hệ thống sẽ tự động chia nhỏ các patent có nội dung lớn (>2KB) thành các đoạn văn bản nhỏ hơn để tối ưu hóa hiệu suất tìm kiếm và giảm chi phí.

## Kiến trúc tổng thể

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Azure AI Search │───▶│  Search Results │
│   (Query)       │    │   (Vector +      │    │                 │
│                 │    │    Text Index)   │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Azure Cosmos DB │
                       │  (Patent Data +  │
                       │   Chunks)        │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Azure OpenAI     │
                       │ (Embeddings)     │
                       └──────────────────┘
```

## 1. Flow: Basic Search với BM25

### Mục đích
Tìm kiếm văn bản cơ bản sử dụng thuật toán BM25 trên Azure AI Search.

### Các bước thực hiện

#### Bước 1: Nhận query từ user
- **Input**: Text query từ người dùng
- **Technology**: Web interface hoặc API endpoint

#### Bước 2: Tìm kiếm trên Azure AI Search
- **Technology**: Azure AI Search với BM25 algorithm
- **Index**: Full-text index của các patent fields (title, abstract, claims, description)
- **Process**: 
  - Phân tích query thành các từ khóa
  - Tính toán relevance score cho mỗi patent
  - Sắp xếp kết quả theo score

#### Bước 3: Kiểm tra và bổ sung từ chunks
- **Logic**: Nếu kết quả từ search index ít hơn yêu cầu
- **Technology**: Cosmos DB query với CONTAINS function
- **Process**:
  - Tìm kiếm text trong các chunks đã được chia nhỏ
  - Nhóm kết quả theo patent_id
  - Bổ sung vào danh sách kết quả cuối cùng

#### Bước 4: Trả về kết quả
- **Output**: Danh sách patents với search score
- **Format**: Patent info + relevance score + source (search_index/chunks)

### Công nghệ sử dụng
- **Azure AI Search**: BM25 algorithm, full-text indexing
- **Cosmos DB**: Text search trong chunks
- **Azure Functions/Web App**: API handling

---

## 2. Flow: Vector Search với Embeddings

### Mục đích
Tìm kiếm semantic similarity sử dụng vector embeddings.

### Các bước thực hiện

#### Bước 1: Nhận query và tạo embedding
- **Input**: Text query từ người dùng
- **Technology**: Azure OpenAI (text-embedding-ada-002)
- **Process**:
  - Gửi query text đến Azure OpenAI
  - Nhận về vector embedding 1536 dimensions
  - Chuẩn bị vector query cho Azure AI Search

#### Bước 2: Vector similarity search
- **Technology**: Azure AI Search với Vector Index
- **Index**: Vector field "combined_vector" với HNSW algorithm
- **Process**:
  - Tính toán khoảng cách cosine giữa query vector và patent vectors
  - Sắp xếp theo similarity score
  - Lọc theo threshold (mặc định: 0.7)

#### Bước 3: Kết hợp với chunk search (nếu cần)
- **Logic**: Tương tự Basic Search
- **Technology**: Cosmos DB vector search trong chunks
- **Process**: Tìm kiếm similarity trong các chunk vectors

#### Bước 4: Trả về kết quả
- **Output**: Danh sách patents với similarity score
- **Format**: Patent info + similarity score + matching chunks info

### Công nghệ sử dụng
- **Azure OpenAI**: Text embedding generation
- **Azure AI Search**: Vector index với HNSW algorithm
- **Cosmos DB**: Vector search trong chunks
- **HNSW Algorithm**: Approximate nearest neighbor search

---

## 3. Flow: Hybrid Search

### Mục đích
Kết hợp text search (BM25) và vector search để có kết quả tối ưu.

### Các bước thực hiện

#### Bước 1: Thực hiện song song
- **Text Search**: BM25 trên Azure AI Search
- **Vector Search**: Embedding + similarity search
- **Technology**: Parallel execution để tối ưu thời gian

#### Bước 2: Kết hợp kết quả
- **Method**: Reciprocal Rank Fusion (RRF)
- **Formula**: RRF = 1/(k + rank_text) + 1/(k + rank_vector)
- **Process**:
  - Gán rank cho mỗi patent trong cả hai phương thức
  - Tính RRF score
  - Sắp xếp theo RRF score giảm dần

#### Bước 3: Bổ sung từ chunks
- **Logic**: Tương tự các method khác
- **Technology**: Cosmos DB search trong chunks
- **Process**: Kết hợp với kết quả chính

#### Bước 4: Trả về kết quả
- **Output**: Danh sách patents với combined score
- **Format**: Patent info + RRF score + individual scores

### Công nghệ sử dụng
- **Azure AI Search**: BM25 + Vector search
- **Azure OpenAI**: Embedding generation
- **RRF Algorithm**: Rank fusion algorithm
- **Parallel Processing**: Concurrent search execution

---

## 4. Flow: Semantic Search với Reranker

### Mục đích
Tìm kiếm semantic với khả năng hiểu ngữ cảnh và sắp xếp lại kết quả.

### Các bước thực hiện

#### Bước 1: Semantic understanding
- **Technology**: Azure AI Search Semantic Search
- **Process**:
  - Phân tích ngữ nghĩa của query
  - Hiểu ý định người dùng
  - Mở rộng query với synonyms

#### Bước 2: Multi-modal search
- **Text Search**: Semantic text search
- **Vector Search**: Embedding similarity
- **Technology**: Azure AI Search semantic capabilities

#### Bước 3: Reranking
- **Method**: Semantic reranking
- **Process**:
  - Sử dụng language model để đánh giá relevance
  - Sắp xếp lại kết quả dựa trên semantic understanding
  - Ưu tiên kết quả có ngữ nghĩa phù hợp nhất

#### Bước 4: Bổ sung từ chunks
- **Logic**: Tương tự các method khác
- **Technology**: Cosmos DB semantic search trong chunks

#### Bước 5: Trả về kết quả
- **Output**: Danh sách patents với semantic score
- **Format**: Patent info + semantic score + reranked order

### Công nghệ sử dụng
- **Azure AI Search**: Semantic search, reranking
- **Azure OpenAI**: Language understanding
- **Semantic Models**: Context-aware search
- **Reranking Engine**: AI-powered result ordering

---

## 5. Flow: Embedding Generation

### Mục đích
Tạo vector embeddings cho patent content để hỗ trợ vector search.

### Các bước thực hiện

#### Bước 1: Chuẩn bị text content
- **Input**: Patent fields (title, abstract, claims, description)
- **Process**:
  - Kết hợp các text fields
  - Truncate nếu vượt quá 8000 characters (OpenAI limit)
  - Chuẩn bị batch processing

#### Bước 2: Chunking (nếu cần)
- **Logic**: Nếu text > 2KB
- **Technology**: Custom chunking algorithm
- **Process**:
  - Chia text thành chunks 1.5KB
  - Giữ overlap 100 characters
  - Tìm boundary tốt (câu, từ)

#### Bước 3: Generate embeddings
- **Technology**: Azure OpenAI API
- **Model**: text-embedding-ada-002
- **Process**:
  - Gửi text chunks đến OpenAI
  - Nhận về vector embeddings
  - Batch processing để tối ưu API calls

#### Bước 4: Lưu trữ
- **Technology**: Azure Cosmos DB
- **Process**:
  - Lưu embeddings vào patent document
  - Lưu chunks riêng biệt (nếu có)
  - Sync với Azure AI Search index

### Công nghệ sử dụng
- **Azure OpenAI**: Embedding generation
- **Cosmos DB**: Vector storage
- **Custom Algorithm**: Text chunking
- **Batch Processing**: API optimization

---

## 6. Setup và Configuration

### Azure AI Search Setup
- **Index Configuration**: 
  - Text fields với analyzer
  - Vector field với HNSW algorithm
  - Semantic search capabilities
- **Scoring Profiles**: Custom scoring cho hybrid search
- **Analyzers**: Language-specific text analysis

### Cosmos DB Setup
- **Container**: Patent documents + chunks
- **Indexing Policy**: Full-text search cho chunks
- **Vector Policy**: Embedding storage và search
- **Partitioning**: Theo patent_office

### Azure OpenAI Setup
- **Deployment**: text-embedding-ada-002
- **API Configuration**: Rate limiting, retry logic
- **Cost Optimization**: Batch processing, caching

### Monitoring và Analytics
- **Search Performance**: Response time, throughput
- **Cost Tracking**: OpenAI API usage, Cosmos DB RU
- **Quality Metrics**: Search relevance, user satisfaction

---

## 7. Lợi ích của Chunking

### Hiệu suất
- **Faster Search**: Chunks nhỏ hơn, search nhanh hơn
- **Better Relevance**: Tìm kiếm chính xác hơn trong nội dung lớn
- **Scalability**: Xử lý được patents có nội dung rất lớn

### Chi phí
- **Reduced RU**: Cosmos DB operations tối ưu hơn
- **Lower API Costs**: OpenAI embedding cho chunks nhỏ hơn
- **Efficient Storage**: Không duplicate data

### User Experience
- **Faster Results**: Response time giảm đáng kể
- **Better Accuracy**: Kết quả search chính xác hơn
- **Comprehensive Coverage**: Tìm kiếm được trong toàn bộ nội dung

---

## 8. Kết luận

Hệ thống chunking được thiết kế để:
1. **Tự động hóa**: Không cần can thiệp thủ công
2. **Tối ưu hóa**: Hiệu suất cao, chi phí thấp
3. **Mở rộng**: Xử lý được patents có nội dung rất lớn
4. **Tương thích**: Hoạt động với tất cả search methods hiện có

Với kiến trúc này, hệ thống có thể xử lý hiệu quả các patent có nội dung lớn mà không ảnh hưởng đến trải nghiệm người dùng và chi phí vận hành.
