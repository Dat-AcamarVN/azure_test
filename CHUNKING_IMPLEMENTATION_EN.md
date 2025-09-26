# Chunking Implementation Guide - Simple Approach

## Overview

This guide presents how to apply chunking to existing source code with **minimal changes**, using **simple functions** instead of OOP classes.

## 1. Simple Strategy

### 1.1 How it works
```
Large Patent (>2KB) → Split into chunks → Save chunks + original patent
                                    ↓
Small Patent (<2KB) → Save directly → No changes
```

### 1.2 Data structure
```json
// Original Patent (no changes)
{
  "id": "patent_id",
  "patent_id": "patent_id",
  "title": "Patent Title",
  "abstract": "Short abstract...",
  "is_chunked": true,        // New field
  "chunk_count": 3           // New field
}

// Chunks (stored separately)
{
  "id": "chunk_uuid",
  "patent_id": "patent_id",
  "chunk_type": "abstract",
  "chunk_order": 0,
  "chunk_text": "Text content...",
  "chunk_size": 1500
}
```

## 2. Implementation with Minimal Changes

### 2.1 Add new functions (don't change existing code)

```python
# utilities/chunking_utils.py - New file
import uuid
import logging

logger = logging.getLogger(__name__)

def should_chunk_patent(patent_text: str) -> bool:
    """Check if patent needs chunking"""
    return len(patent_text) > 2000  # 2KB threshold

def chunk_text_simple(text: str, chunk_type: str, patent_id: str) -> list:
    """Split text into simple chunks"""
    if not text or len(text) <= 1500:
        return []
    
    chunks = []
    chunk_size = 1500
    overlap = 100
    
    start = 0
    order = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Find good boundary (sentence or word)
        if end < len(text):
            end = find_good_boundary(text, start, end)
        
        chunk_text = text[start:end]
        
        chunk = {
            "id": str(uuid.uuid4()),
            "patent_id": patent_id,
            "chunk_type": chunk_type,
            "chunk_order": order,
            "chunk_text": chunk_text,
            "chunk_size": len(chunk_text)
        }
        
        chunks.append(chunk)
        order += 1
        start = max(start + 1, end - overlap)
        
        if start >= len(text):
            break
    
    return chunks

def find_good_boundary(text: str, start: int, max_end: int) -> int:
    """Find good boundary to split chunk"""
    # Find nearest punctuation
    for i in range(max_end, max(start, max_end - 200), -1):
        if i < len(text) and text[i] in '.!?;':
            return i + 1
    
    return max_end

def save_chunks_to_db(container, chunks: list, partition_key: str) -> bool:
    """Save chunks to database"""
    try:
        if not chunks:
            return True
        
        # Use batch to optimize RU
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # Create batch operations
            batch = container.create_transactional_batch(partition_key)
            for chunk in batch_chunks:
                batch.create_item(chunk)
            
            # Execute batch
            batch_result = batch.execute()
            if not batch_result.is_successful:
                logger.error(f"Failed to save chunk batch {i//batch_size}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving chunks: {e}")
        return False

def get_chunks_from_db(container, patent_id: str) -> list:
    """Get chunks from database"""
    try:
        query = "SELECT * FROM c WHERE c.patent_id = @patent_id AND c.chunk_type != null ORDER BY c.chunk_order"
        parameters = [{"name": "@patent_id", "value": patent_id}]
        
        chunks = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error getting chunks: {e}")
        return []

def reconstruct_patent_from_chunks(patent: dict, chunks: list) -> dict:
    """Reconstruct patent from chunks"""
    try:
        if not chunks:
            return patent
        
        # Group chunks by type
        chunks_by_type = {}
        for chunk in chunks:
            chunk_type = chunk["chunk_type"]
            if chunk_type not in chunks_by_type:
                chunks_by_type[chunk_type] = []
            chunks_by_type[chunk_type].append(chunk)
        
        # Reconstruct each field
        for chunk_type, type_chunks in chunks_by_type.items():
            # Sort by order
            type_chunks.sort(key=lambda x: x["chunk_order"])
            
            # Combine text
            combined_text = " ".join([chunk["chunk_text"] for chunk in type_chunks])
            
            # Set to patent field
            if chunk_type == "abstract":
                patent["abstract"] = combined_text
            elif chunk_type == "claims":
                patent["claims"] = combined_text
            elif chunk_type == "description":
                patent["description"] = combined_text
        
        return patent
        
    except Exception as e:
        logger.error(f"Error reconstructing patent: {e}")
        return patent
```

### 2.2 Update PatentInfo (only add 2 fields)

```python
# models/patent_model.py - Only add 2 new fields
@dataclass
class PatentInfo:
    # ... existing fields (NO CHANGES) ...
    
    # Only add 2 new fields
    is_chunked: bool = False
    chunk_count: int = 0
    
    # ... existing methods (NO CHANGES) ...
```

### 2.3 Update Patent DAO (only add chunking logic)

```python
# dao/patent_dao.py - Only add import and chunking logic
from utilities.chunking_utils import (
    should_chunk_patent, 
    chunk_text_simple, 
    save_chunks_to_db,
    get_chunks_from_db,
    reconstruct_patent_from_chunks
)

def create_patent(connection_string: str, database_name: str, container_name: str,
                  search_endpoint: str, search_key: str, search_index_name: str,
                  patent: PatentInfo) -> Optional[Exception]:
    """Create patent with automatic chunking"""
    try:
        container = _get_container(connection_string, database_name, container_name)
        search_client = _get_search_client(search_endpoint, search_key, search_index_name)

        # Validate patent (NO CHANGES)
        if not patent.patent_id:
            raise ValueError("Patent ID is required")

        # Update timestamp (NO CHANGES)
        patent.update_timestamps()

        # === NEW CHUNKING LOGIC ===
        combined_text = _prepare_combined_text(patent)
        if should_chunk_patent(combined_text):
            logger.info(f"🔄 Applying chunking to patent {patent.patent_id}")
            
            # Create chunks
            all_chunks = []
            
            # Chunk abstract
            if patent.abstract:
                abstract_chunks = chunk_text_simple(patent.abstract, "abstract", patent.patent_id)
                all_chunks.extend(abstract_chunks)
            
            # Chunk claims
            if patent.claims:
                claims_chunks = chunk_text_simple(patent.claims, "claims", patent.patent_id)
                all_chunks.extend(claims_chunks)
            
            # Chunk description
            if patent.description:
                desc_chunks = chunk_text_simple(patent.description, "description", patent.patent_id)
                all_chunks.extend(desc_chunks)
            
            # Save chunks
            if all_chunks:
                if save_chunks_to_db(container, all_chunks, patent.patent_office):
                    patent.is_chunked = True
                    patent.chunk_count = len(all_chunks)
                    logger.info(f"✅ Patent chunked into {len(all_chunks)} chunks")
                else:
                    logger.warning("⚠️ Failed to save chunks, continuing without chunking")
        
        # === EXISTING LOGIC (NO CHANGES) ===
        combined_text = _prepare_combined_text(patent)
        patent.combined_vector = generate_embeddings([combined_text])[0]

        if len(patent.combined_vector) != config.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch: expected {config.EMBEDDING_DIMENSION}, got {len(patent.combined_vector)}"
            )
        
        # Prepare document (NO CHANGES)
        doc = _prepare_document_for_upload(patent)

        # Insert to Cosmos DB (NO CHANGES)
        container.create_item(body=doc)

        # Sync to Azure AI Search (NO CHANGES)
        try:
            search_client.upload_documents([doc])
            logger.info(f"✅ Synced to Azure AI Search: {patent.patent_id}")
        except Exception as search_error:
            logger.warning(f"⚠️ Failed to sync to Azure AI Search: {search_error}")

        logger.info(f"✅ Created Patent: {patent.patent_id}")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error creating Patent: {e}")
        return e

def read_patent(connection_string: str, database_name: str, container_name: str, 
                patent_id: str, reconstruct_chunks: bool = True) -> Optional[PatentInfo]:
    """Read patent with auto-reconstruction from chunks"""
    try:
        container = _get_container(connection_string, database_name, container_name)

        # === EXISTING LOGIC (NO CHANGES) ===
        query = "SELECT * FROM c WHERE c.patent_id = @patent_id"
        parameters = [{"name": "@patent_id", "value": patent_id}]

        items = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))

        if items:
            patent_dict = items[0]
            
            # === NEW CHUNKING LOGIC ===
            if reconstruct_chunks and patent_dict.get("is_chunked"):
                logger.info(f"🔄 Reconstructing patent {patent_id} from chunks")
                
                # Get chunks
                chunks = get_chunks_from_db(container, patent_id)
                
                # Reconstruct
                if chunks:
                    patent_dict = reconstruct_patent_from_chunks(patent_dict, chunks)
                    logger.info(f"✅ Patent {patent_id} reconstructed from {len(chunks)} chunks")
            
            # === EXISTING LOGIC (NO CHANGES) ===
            return PatentInfo.from_dict(patent_dict)
        else:
            logger.warning(f"❌ Patent not found: {patent_id}")
            return None

    except Exception as e:
        logger.error(f"❌ Error reading Patent: {e}")
        return None
```

## 3. Search Operations with Chunks

### 3.1 Text Search in Chunks

```python
def search_in_chunks(connection_string: str, database_name: str, container_name: str,
                     search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search text in chunks"""
    try:
        container = _get_container(connection_string, database_name, container_name)
        
        # Search in chunks
        query = """
        SELECT c.patent_id, c.chunk_text, c.chunk_type, c.chunk_order
        FROM c 
        WHERE CONTAINS(c.chunk_text, @search_text, true)
        ORDER BY c.chunk_size DESC
        LIMIT @limit
        """
        
        parameters = [
            {"name": "@search_text", "value": search_text},
            {"name": "@limit", "value": limit}
        ]
        
        chunks = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        
        # Group by patent_id
        results = []
        patent_chunks = {}
        
        for chunk in chunks:
            patent_id = chunk["patent_id"]
            if patent_id not in patent_chunks:
                patent_chunks[patent_id] = []
            patent_chunks[patent_id].append(chunk)
        
        # Convert to results
        for patent_id, chunk_list in patent_chunks.items():
            # Get patent info
            patent = read_patent(connection_string, database_name, container_name, 
                               patent_id, reconstruct_chunks=False)
            
            if patent:
                results.append({
                    "patent": patent,
                    "matching_chunks": chunk_list,
                    "total_matches": len(chunk_list)
                })
        
        return results[:limit]
        
    except Exception as e:
        logger.error(f"❌ Error searching in chunks: {e}")
        return []
```

### 3.2 Update existing search functions

```python
def basic_search(search_endpoint: str, search_key: str, search_index_name: str,
                 query: str, filters: List[SearchInfo] = [],
                 skip: int = 0, top: int = 10) -> List[Dict[str, Any]]:
    """Basic search with chunking support"""
    try:
        # === EXISTING LOGIC (NO CHANGES) ===
        search_client = _get_search_client(search_endpoint, search_key, search_index_name)
        filter_str = _build_filter_conditions(filters)
        
        results = search_client.search(
            search_text=query,
            filter=filter_str,
            top=top,
            skip=skip,
            include_total_count=True,
            select=["patent_id", "title", "abstract", "claims", "patent_office", "created_at", "updated_at"]
        )
        
        # === NEW CHUNKING LOGIC ===
        # If results are few, search more in chunks
        if len(results) < top:
            remaining = top - len(results)
            chunk_results = search_in_chunks(connection_string, database_name, container_name, 
                                          query, remaining)
            
            # Convert chunk results to same format
            for chunk_result in chunk_results:
                patent = chunk_result["patent"]
                results_list.append({
                    'patent': patent, 
                    'search_score': 0.8,  # Lower score for chunk results
                    'source': 'chunks'
                })
        
        # === EXISTING LOGIC (NO CHANGES) ===
        results_list = []
        for r in results:
            search_score = r.get('@search.score', 0.0)
            clean_result = {k: v for k, v in r.items() if not k.startswith('@search.')}
            patent = PatentInfo.from_dict(clean_result)
            results_list.append({'patent': patent, 'search_score': search_score, 'source': 'search_index'})
        
        return results_list
        
    except Exception as e:
        logger.error(f"❌ Error in basic search: {e}")
        return []
```

## 4. Simple Configuration

### 4.1 Add config for chunking

```python
# config.py - Only add 4 lines
# ... existing config ...

# Chunking Configuration (new)
CHUNKING_ENABLED = True
CHUNKING_THRESHOLD = 2000  # 2KB
MAX_CHUNK_SIZE = 1500      # 1.5KB per chunk
CHUNK_OVERLAP = 100        # 100 chars overlap
```

### 4.2 Use config

```python
# utilities/chunking_utils.py
def should_chunk_patent(patent_text: str) -> bool:
    """Check if patent needs chunking"""
    if not config.CHUNKING_ENABLED:
        return False
    return len(patent_text) > config.CHUNKING_THRESHOLD

def chunk_text_simple(text: str, chunk_type: str, patent_id: str) -> list:
    """Split text into simple chunks"""
    if not text or len(text) <= config.MAX_CHUNK_SIZE:
        return []
    
    # ... existing logic with config values ...
```

## 5. Simple Testing

### 5.1 Test functions

```python
# test_chunking.py
def test_chunking_functions():
    """Test chunking functions"""
    
    # Test should_chunk_patent
    assert not should_chunk_patent("Short text")
    assert should_chunk_patent("A" * 2500)
    
    # Test chunk_text_simple
    long_text = "A" * 5000
    chunks = chunk_text_simple(long_text, "abstract", "test_id")
    assert len(chunks) > 1
    assert all(chunk["chunk_size"] <= 1500 for chunk in chunks)

def test_integration():
    """Test integration with existing code"""
    
    # Create large patent
    large_patent = PatentInfo(
        patent_id="test_chunking",
        title="Test",
        abstract="A" * 3000  # 3KB
    )
    
    # Test create (should trigger chunking)
    err = create_patent(conn_str, database_name, container_name,
                        search_endpoint, search_key, search_index_name, large_patent)
    assert err is None
    
    # Test read (should reconstruct)
    retrieved = read_patent(conn_str, database_name, container_name, "test_chunking")
    assert retrieved.abstract == large_patent.abstract
    assert retrieved.is_chunked == True
```

## 6. Migration Strategy

### 6.1 Gradual Rollout

```python
# Only apply to new patents
if config.CHUNKING_ENABLED:
    # Apply chunking logic
    pass

# Or apply to patents with content > threshold
if len(combined_text) > config.CHUNKING_THRESHOLD:
    # Apply chunking
    pass
```

### 6.2 Simple Rollback

```python
def disable_chunking_for_patent(patent_id: str):
    """Disable chunking for 1 patent"""
    try:
        # 1. Reconstruct patent
        patent = read_patent(conn_str, database_name, container_name, patent_id)
        
        # 2. Delete chunks
        container = _get_container(conn_str, database_name, container_name)
        query = "SELECT c.id FROM c WHERE c.patent_id = @patent_id AND c.chunk_type != null"
        parameters = [{"name": "@patent_id", "value": patent_id}]
        
        chunks = list(container.query_items(query=query, parameters=parameters))
        
        for chunk in chunks:
            container.delete_item(item=chunk["id"], partition_key=patent.patent_office)
        
        # 3. Update patent
        patent.is_chunked = False
        patent.chunk_count = 0
        
        update_patent(conn_str, database_name, container_name,
                     search_endpoint, search_key, search_index_name, patent)
        
        logger.info(f"✅ Disabled chunking for patent {patent_id}")
        
    except Exception as e:
        logger.error(f"❌ Error disabling chunking: {e}")
```

## 7. Summary of Changes

### 7.1 New files
- `utilities/chunking_utils.py` - Contains all chunking logic

### 7.2 Changed files
- `models/patent_model.py` - Only add 2 fields: `is_chunked`, `chunk_count`
- `dao/patent_dao.py` - Only add chunking logic to 2 functions: `create_patent`, `read_patent`
- `config.py` - Only add 4 lines of config

### 7.3 Files NOT changed
- All other search functions
- All other CRUD functions
- All other models
- All other utilities

### 7.4 Benefits
- **Minimal code changes** - Only change 3 files
- **Backward compatible** - Works with old patents
- **Simple approach** - No OOP, just functions
- **Easy to test** - Simple logic, easy to test
- **Easy to rollback** - Can disable easily

## 8. Conclusion

This approach ensures:
1. **No changes** to existing logic
2. **Only add** chunking when needed
3. **Easy to implement** and test
4. **Easy to maintain** and debug
5. **Can rollback** if needed

With this approach, you can safely apply chunking without affecting the existing functionality of the system.
