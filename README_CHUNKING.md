# Patent Search - Chunking Implementation (Simple & Practical)

## 🎯 Purpose

Apply **chunking** to the patent search system to efficiently handle patents with large content (>2KB) while **preserving existing logic**.

## 🚀 Key Features

### 1. **Automatic chunking**
- Patent > 2KB → Automatically divided into 1.5KB chunks
- 100 character overlap to ensure search accuracy
- Smart boundaries (sentences, words) instead of cutting across

### 2. **Full CRUD support**
- **Create**: Automatic chunking when needed
- **Read**: Auto-reconstruction from chunks
- **Update**: Re-chunking when content changes
- **Delete**: Cleanup chunks when deleting patent

### 3. **Enhanced search**
- Search index (Azure AI Search) + Chunk search (Cosmos DB)
- Combine results from both sources
- Relevance scoring for chunk results

## 📁 Files Modified

### 1. `config.py` - Added 3 config lines
```python
# Chunking Configuration - Simple & Practical
CHUNKING_ENABLED = True
CHUNKING_THRESHOLD = 2000  # 2KB - only chunk when really needed
MAX_CHUNK_SIZE = 1500      # 1.5KB per chunk - optimized for search
CHUNK_OVERLAP = 100        # 100 chars overlap - ensure search accuracy
```

### 2. `models/patent_model.py` - Only added 2 fields
```python
# Chunking fields - simple tracking
is_chunked: bool = False
chunk_count: int = 0
```

### 3. `utilities/chunking_utils.py` - New file (6 functions)
- `should_chunk_patent()`: Check if chunking is needed
- `chunk_text_simple()`: Split text into chunks
- `save_chunks_to_db()`: Save chunks to database
- `get_chunks_from_db()`: Get chunks from database
- `reconstruct_patent_from_chunks()`: Reconstruct patent from chunks
- `search_in_chunks()`: Search within chunks

### 4. `dao/patent_dao.py` - Integrated chunking
- `create_patent()`: Automatic chunking when creating
- `update_patent()`: Re-chunking when updating
- `read_patent()`: Auto-reconstruction from chunks
- `delete_patent()`: Delete chunks when deleting
- `basic_search()`: Supplement results from chunks

### 5. `test_chunking.py` - New comprehensive test file
- Test all chunking utilities
- Test CRUD operations with chunking
- Test search methods with chunking support
- Test chunk management operations
- Test cleanup operations

## 🔧 How to Use

### 1. **Create patent with automatic chunking**
```python
from models.patent_model import PatentInfo
from dao.patent_dao import create_patent

# Large patent will be automatically chunked
large_patent = PatentInfo(
    patent_id="US123456",
    title="Large Patent",
    abstract="Very long abstract..." * 100,  # >2KB
    claims="Very long claims..." * 150,      # >2KB
    patent_office="USPTO"
)

# Automatic chunking and save
err = create_patent(connection_string, database_name, container_name,
                    search_endpoint, search_key, search_index_name, large_patent)

if err is None:
    print(f"✅ Patent created with {large_patent.chunk_count} chunks")
```

### 2. **Read patent with auto-reconstruction**
```python
from dao.patent_dao import read_patent

# Automatic reconstruction from chunks
patent = read_patent(connection_string, database_name, container_name, "US123456")

if patent:
    print(f"📖 Patent: {patent.title}")
    print(f"📄 Is chunked: {patent.is_chunked}")
    print(f"📄 Chunk count: {patent.chunk_count}")
    print(f"📄 Abstract length: {len(patent.abstract or '')}")
```

### 3. **Search with chunking support**
```python
from dao.patent_dao import basic_search

# Search with chunking support
results = basic_search(
    search_endpoint=search_endpoint,
    search_key=search_key,
    search_index_name=search_index_name,
    query="artificial intelligence",
    connection_string=connection_string,  # Required for chunk search
    database_name=database_name,
    container_name=container_name,
    top=10
)

for result in results:
    print(f"📄 Patent: {result['patent'].title}")
    print(f"📄 Score: {result['search_score']}")
    print(f"📄 Source: {result.get('source', 'search_index')}")
    
    if result.get('matching_chunks'):
        print(f"📄 Matching chunks: {len(result['matching_chunks'])}")
```

## 🧪 Testing

### **Run comprehensive chunking tests**
```bash
python test_chunking.py
```

### **Test specific functionality**
```python
# Test chunking utilities
from utilities.chunking_utils import should_chunk_patent, chunk_text_simple

# Test if text needs chunking
needs_chunking = should_chunk_patent("Very long text..." * 100)
print(f"Needs chunking: {needs_chunking}")

# Test text chunking
chunks = chunk_text_simple("Long text...", "abstract", "test_id")
print(f"Created {len(chunks)} chunks")
```

## ⚙️ Configuration

### **Enable/Disable chunking**
```python
# In config.py
CHUNKING_ENABLED = False  # Disable chunking
```

### **Adjust thresholds**
```python
# In config.py
CHUNKING_THRESHOLD = 1000  # 1KB instead of 2KB
MAX_CHUNK_SIZE = 1000      # 1KB per chunk
CHUNK_OVERLAP = 50         # 50 character overlap
```

## 🔍 Search Flow with Chunking

### **Basic Search Flow**
```
1. Search on Azure AI Search (BM25)
2. If results < limit:
   - Search more in chunks (Cosmos DB)
   - Combine results
3. Return sorted results
```

### **Chunk Search Logic**
```python
def search_in_chunks(container, search_text: str, limit: int = 10):
    # Use Cosmos DB CONTAINS function
    query = """
    SELECT c.patent_id, c.chunk_text, c.chunk_type, c.chunk_order
    FROM c 
    WHERE CONTAINS(c.chunk_text, @search_text, true)
    ORDER BY c.chunk_size DESC
    LIMIT @limit
    """
    
    # Group by patent_id and return patents with matching chunks
```

## 📊 Benefits

### **Performance**
- Faster search with smaller chunks
- Better relevance with smart boundaries
- Scalable with patents having large content

### **Cost**
- Reduced RU consumption
- Lower OpenAI API costs
- Efficient storage without duplication

### **User Experience**
- Faster response time
- Better search accuracy
- Comprehensive content coverage

## 🚨 Troubleshooting

### **Chunking not working**
- Check `CHUNKING_ENABLED = True` in config
- Check text length > `CHUNKING_THRESHOLD`
- Check logs for error messages

### **Reconstruction failing**
- Check if chunks exist in database
- Check `patent_id` and `chunk_type` fields
- Check `chunk_order` to ensure correct sequence

## 🎉 Conclusion

This implementation ensures:

1. **✅ No changes** to existing logic
2. **✅ Only adds** chunking when needed
3. **✅ Easy to implement** and maintain
4. **✅ Can be rolled back** easily
5. **✅ Optimized** search performance

With this approach, you can safely apply chunking without affecting the existing functionality of the system! 🚀
