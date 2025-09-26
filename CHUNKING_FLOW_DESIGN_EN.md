# Chunking Flow Design Overview

## System Overview

The Patent Search system uses **Azure Cosmos DB** as the primary database and **Azure AI Search** for advanced search methods. When applying chunking, the system automatically splits large patents (>2KB) into smaller text segments to optimize search performance and reduce costs.

## Overall Architecture

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

## 1. Flow: Basic Search with BM25

### Purpose
Basic text search using BM25 algorithm on Azure AI Search.

### Implementation Steps

#### Step 1: Receive user query
- **Input**: Text query from user
- **Technology**: Web interface or API endpoint

#### Step 2: Search on Azure AI Search
- **Technology**: Azure AI Search with BM25 algorithm
- **Index**: Full-text index of patent fields (title, abstract, claims)
- **Process**: 
  - Parse query into keywords
  - Calculate relevance score for each patent
  - Sort results by score

#### Step 3: Check and supplement from chunks
- **Logic**: If results from search index are fewer than requested
- **Technology**: Cosmos DB query with CONTAINS function
- **Process**:
  - Search text in chunked segments
  - Group results by patent_id
  - Supplement to final results list

#### Step 4: Return results
- **Output**: List of patents with search score
- **Format**: Patent info + relevance score + source (search_index/chunks)

### Technologies Used
- **Azure AI Search**: BM25 algorithm, full-text indexing
- **Cosmos DB**: Text search in chunks
- **Azure Functions/Web App**: API handling

---

## 2. Flow: Vector Search with Embeddings

### Purpose
Semantic similarity search using vector embeddings.

### Implementation Steps

#### Step 1: Receive query and generate embedding
- **Input**: Text query from user
- **Technology**: Azure OpenAI (text-embedding-3-large)
- **Process**:
  - Send query text to Azure OpenAI
  - Receive vector embedding with 3072 dimensions
  - Prepare vector query for Azure AI Search

#### Step 2: Vector similarity search
- **Technology**: Azure AI Search with Vector Index
- **Index**: Vector field "combined_vector" with HNSW algorithm
- **Process**:
  - Calculate cosine distance between query vector and patent vectors
  - Sort by similarity score
  - Filter by threshold (default: 0.5)

#### Step 3: Combine with chunk search (if needed)
- **Logic**: Similar to Basic Search
- **Technology**: Cosmos DB vector search in chunks
- **Process**: Search similarity in chunk vectors

#### Step 4: Return results
- **Output**: List of patents with similarity score
- **Format**: Patent info + similarity score + matching chunks info

### Technologies Used
- **Azure OpenAI**: Text embedding generation (text-embedding-3-large)
- **Azure AI Search**: Vector index with HNSW algorithm
- **Cosmos DB**: Vector search in chunks
- **HNSW Algorithm**: Approximate nearest neighbor search

---

## 3. Flow: Hybrid Search

### Purpose
Combine text search (BM25) and vector search for optimal results.

### Implementation Steps

#### Step 1: Parallel execution
- **Text Search**: BM25 on Azure AI Search
- **Vector Search**: Embedding + similarity search
- **Technology**: Parallel execution to optimize time

#### Step 2: Combine results
- **Method**: Reciprocal Rank Fusion (RRF)
- **Formula**: RRF = 1/(k + rank_text) + 1/(k + rank_vector)
- **Process**:
  - Assign rank to each patent in both methods
  - Calculate RRF score
  - Sort by RRF score descending

#### Step 3: Supplement from chunks
- **Logic**: Similar to other methods
- **Technology**: Cosmos DB search in chunks
- **Process**: Combine with main results

#### Step 4: Return results
- **Output**: List of patents with combined score
- **Format**: Patent info + RRF score + individual scores

### Technologies Used
- **Azure AI Search**: BM25 + Vector search
- **Azure OpenAI**: Embedding generation
- **RRF Algorithm**: Rank fusion algorithm (k=60)
- **Parallel Processing**: Concurrent search execution

---

## 4. Flow: Semantic Search with Reranker

### Purpose
Semantic search with context understanding and result reranking.

### Implementation Steps

#### Step 1: Semantic understanding
- **Technology**: Azure AI Search Semantic Search
- **Process**:
  - Analyze query semantics
  - Understand user intent
  - Expand query with synonyms

#### Step 2: Multi-modal search
- **Text Search**: Semantic text search
- **Vector Search**: Embedding similarity
- **Technology**: Azure AI Search semantic capabilities

#### Step 3: Reranking
- **Method**: Semantic reranking
- **Process**:
  - Use language model to evaluate relevance
  - Rerank results based on semantic understanding
  - Prioritize semantically most relevant results

#### Step 4: Supplement from chunks
- **Logic**: Similar to other methods
- **Technology**: Cosmos DB semantic search in chunks

#### Step 5: Return results
- **Output**: List of patents with semantic score
- **Format**: Patent info + semantic score + reranked order

### Technologies Used
- **Azure AI Search**: Semantic search, reranking
- **Azure OpenAI**: Language understanding
- **Semantic Models**: Context-aware search
- **Reranking Engine**: AI-powered result ordering

---

## 5. Flow: Embedding Generation

### Purpose
Generate vector embeddings for patent content to support vector search.

### Implementation Steps

#### Step 1: Prepare text content
- **Input**: Patent fields (title, abstract, claims, description)
- **Process**:
  - Combine text fields
  - Truncate if exceeding 8000 characters (OpenAI limit)
  - Prepare batch processing

#### Step 2: Chunking (if needed)
- **Logic**: If text > 2KB
- **Technology**: Custom chunking algorithm
- **Process**:
  - Split text into 1.5KB chunks
  - Maintain 100 character overlap
  - Find good boundaries (sentences, words)

#### Step 3: Generate embeddings
- **Technology**: Azure OpenAI API
- **Model**: text-embedding-3-large
- **Process**:
  - Send text chunks to OpenAI
  - Receive vector embeddings (3072 dimensions)
  - Batch processing to optimize API calls

#### Step 4: Storage
- **Technology**: Azure Cosmos DB
- **Process**:
  - Store embeddings in patent document
  - Store chunks separately (if any)
  - Sync with Azure AI Search index

### Technologies Used
- **Azure OpenAI**: Embedding generation (text-embedding-3-large)
- **Cosmos DB**: Vector storage
- **Custom Algorithm**: Text chunking
- **Batch Processing**: API optimization

---

## 6. Setup and Configuration

### Azure AI Search Setup
- **Endpoint**: ircaisearch.search.windows.net
- **Index Name**: pkindex
- **Index Configuration**: 
  - Text fields with analyzer
  - Vector field with HNSW algorithm
  - Semantic search capabilities
- **Scoring Profiles**: Custom scoring for hybrid search
- **Analyzers**: Language-specific text analysis

### Cosmos DB Setup
- **Endpoint**: irctesting.documents.azure.com
- **Database**: ClaimChartDB
- **Container**: ClaimCharts
- **Container Configuration**: Patent documents + chunks
- **Indexing Policy**: Full-text search for chunks
- **Vector Policy**: Embedding storage and search
- **Partitioning**: By patent_office

### Azure OpenAI Setup
- **Endpoint**: irctesting.openai.azure.com
- **Deployment**: text-embedding-3-large
- **API Version**: 2024-02-01
- **Embedding Dimension**: 3072
- **API Configuration**: Rate limiting, retry logic
- **Cost Optimization**: Batch processing, caching

### Search Configuration
- **Similarity Threshold**: 0.5 (default)
- **Hybrid Search Weights**: Text (0.6), Vector (0.4)
- **RRF Parameter**: k = 60

### Monitoring and Analytics
- **Search Performance**: Response time, throughput
- **Cost Tracking**: OpenAI API usage, Cosmos DB RU
- **Quality Metrics**: Search relevance, user satisfaction

---

## 7. Benefits of Chunking

### Performance
- **Faster Search**: Smaller chunks, faster search
- **Better Relevance**: More accurate search in large content
- **Scalability**: Handle very large patents efficiently

### Cost
- **Reduced RU**: Optimized Cosmos DB operations
- **Lower API Costs**: OpenAI embedding for smaller chunks
- **Efficient Storage**: No data duplication

### User Experience
- **Faster Results**: Significantly reduced response time
- **Better Accuracy**: More accurate search results
- **Comprehensive Coverage**: Search across entire content

---

## 8. Conclusion

The chunking system is designed to:
1. **Automation**: No manual intervention required
2. **Optimization**: High performance, low cost
3. **Scalability**: Handle patents with very large content
4. **Compatibility**: Work with all existing search methods

With this architecture, the system can efficiently process large content patents without affecting user experience and operational costs.

---

## Technical Specifications

### Azure AI Search
- **Service**: ircaisearch.search.windows.net
- **Index**: pkindex
- **Capabilities**: Vector search, semantic search, BM25

### Cosmos DB
- **Service**: irctesting.documents.azure.com
- **Database**: ClaimChartDB
- **Container**: ClaimCharts
- **Features**: Vector search, full-text search

### Azure OpenAI
- **Service**: irctesting.openai.azure.com
- **Model**: text-embedding-3-large
- **Dimensions**: 3072
- **API Version**: 2024-02-01

### Search Parameters
- **Default Similarity**: 0.5
- **Hybrid Weights**: Text (60%), Vector (40%)
- **RRF K Value**: 60

**Note:** These **Search configuration settings** can all be customized/adjusted:
#### Search Configuration
- DEFAULT_SIMILARITY_THRESHOLD = 0.5
- HYBRID_SEARCH_TEXT_WEIGHT = 0.6
- HYBRID_SEARCH_VECTOR_WEIGHT = 0.4
- DEFAULT_RRF_K = 60
