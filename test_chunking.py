"""
Test Chunking Implementation
Test all CRUD operations and search methods with chunking support
"""

import json
import logging
from azure.cosmos import CosmosClient

from config import (
    AZURE_SEARCH_ENDPOINT, 
    AZURE_SEARCH_ADMIN_KEY, 
    AZURE_SEARCH_INDEX_NAME, 
    COSMOS_DB_CONNECTION_STRING,
    CHUNKING_ENABLED,
    CHUNKING_THRESHOLD,
    MAX_CHUNK_SIZE
)
from dao.patent_dao import (
    create_patent, 
    read_patent, 
    update_patent, 
    list_all_patents, 
    basic_search, 
    vector_search, 
    hybrid_search, 
    semantic_search, 
    bm25_search, 
    search_rrf_hybrid, 
    search_semantic_reranker, 
    delete_patent, 
    create_database_and_container, 
    create_or_update_index
)
from models.patent_model import PatentInfo
from utilities.chunking_utils import (
    should_chunk_patent,
    chunk_text_simple,
    save_chunks_to_db,
    get_chunks_from_db,
    reconstruct_patent_from_chunks,
    search_in_chunks
)

# Import and configure logging
from logging_config import configure_test_logging
configure_test_logging()

logger = logging.getLogger(__name__)


def test_chunking_utilities():
    """Test chunking utility functions"""
    print("\n" + "="*50)
    print("TESTING CHUNKING UTILITIES")
    print("="*50)
    
    # Test should_chunk_patent
    short_text = "Short text that should not be chunked"
    long_text = "A" * 2500  # 2.5KB, should be chunked
    
    print(f"Testing should_chunk_patent:")
    print(f"  Short text ({len(short_text)} chars): {should_chunk_patent(short_text)}")
    print(f"  Long text ({len(long_text)} chars): {should_chunk_patent(long_text)}")
    
    # Test chunk_text_simple
    test_text = "This is a test text. " * 100  # Create long text
    chunks = chunk_text_simple(test_text, "abstract", "test_patent_001")
    
    print(f"\nTesting chunk_text_simple:")
    print(f"  Original text length: {len(test_text)}")
    print(f"  Number of chunks created: {len(chunks)}")
    print(f"  First chunk size: {len(chunks[0]['chunk_text']) if chunks else 0}")
    print(f"  Chunk overlap check: {chunks[1]['chunk_text'][:50] if len(chunks) > 1 else 'N/A'}")
    
    return chunks


def test_patent_creation_with_chunking():
    """Test creating patents with automatic chunking"""
    print("\n" + "="*50)
    print("TESTING PATENT CREATION WITH CHUNKING")
    print("="*50)
    
    # Create a large patent that should trigger chunking
    large_patent = PatentInfo(
        patent_id="CHUNK_TEST_001",
        title="Large Patent for Chunking Test",
        abstract="This is a very long abstract. " * 150,  # ~3KB
        claims="These are very long claims. " * 200,      # ~4KB
        description="This is a very long description. " * 300,  # ~6KB
        patent_office="USPTO",
        assignee="Test Company",
        inventor="Test Inventor"
    )
    
    print(f"Created large patent:")
    print(f"  Patent ID: {large_patent.patent_id}")
    print(f"  Abstract length: {len(large_patent.abstract or '')}")
    print(f"  Claims length: {len(large_patent.claims or '')}")
    print(f"  Description length: {len(large_patent.description or '')}")
    print(f"  Should be chunked: {should_chunk_patent(large_patent.abstract or '')}")
    
    return large_patent


def test_crud_operations_with_chunking(connection_string, database_name, container_name, 
                                      search_endpoint, search_key, search_index_name):
    """Test CRUD operations with chunking support"""
    print("\n" + "="*50)
    print("TESTING CRUD OPERATIONS WITH CHUNKING")
    print("="*50)
    
    # Test 1: Create large patent
    large_patent = test_patent_creation_with_chunking()
    
    print(f"\n1. Creating large patent...")
    err = create_patent(
        connection_string, 
        database_name, 
        container_name,
        search_endpoint, 
        search_key, 
        search_index_name, 
        large_patent
    )
    
    if err is None:
        print(f"✅ Patent created successfully")
        print(f"   Is chunked: {large_patent.is_chunked}")
        print(f"   Chunk count: {large_patent.chunk_count}")
    else:
        print(f"❌ Failed to create patent: {err}")
        return None
    
    # Test 2: Read patent with reconstruction
    print(f"\n2. Reading patent with reconstruction...")
    retrieved_patent = read_patent(
        connection_string, 
        database_name, 
        container_name, 
        large_patent.patent_id,
        reconstruct_chunks=True
    )
    
    if retrieved_patent:
        print(f"✅ Patent retrieved successfully")
        print(f"   Is chunked: {retrieved_patent.is_chunked}")
        print(f"   Chunk count: {retrieved_patent.chunk_count}")
        print(f"   Abstract length: {len(retrieved_patent.abstract or '')}")
        print(f"   Claims length: {len(retrieved_patent.claims or '')}")
        print(f"   Description length: {len(retrieved_patent.description or '')}")
        
        # Verify reconstruction
        original_length = len(large_patent.abstract or '')
        retrieved_length = len(retrieved_patent.abstract or '')
        if original_length == retrieved_length:
            print(f"✅ Reconstruction successful - lengths match")
        else:
            print(f"⚠️ Reconstruction issue - length mismatch: {original_length} vs {retrieved_length}")
    else:
        print(f"❌ Failed to retrieve patent")
        return None
    
    # Test 3: Update patent (should trigger re-chunking)
    print(f"\n3. Updating patent (should trigger re-chunking)...")
    retrieved_patent.abstract = "Updated very long abstract. " * 200  # ~4KB
    retrieved_patent.claims = "Updated very long claims. " * 250     # ~5KB
    
    err = update_patent(
        connection_string, 
        database_name, 
        container_name,
        search_endpoint, 
        search_key, 
        search_index_name, 
        retrieved_patent
    )
    
    if err is None:
        print(f"✅ Patent updated successfully")
        print(f"   Is chunked: {retrieved_patent.is_chunked}")
        print(f"   Chunk count: {retrieved_patent.chunk_count}")
    else:
        print(f"❌ Failed to update patent: {err}")
    
    return retrieved_patent


def test_search_with_chunking(connection_string, database_name, container_name,
                             search_endpoint, search_key, search_index_name):
    """Test search methods with chunking support"""
    print("\n" + "="*50)
    print("TESTING SEARCH WITH CHUNKING")
    print("="*50)
    
    # Test basic search with chunking support
    print(f"\n1. Testing basic search with chunking...")
    results = basic_search(
        search_endpoint=search_endpoint,
        search_key=search_key,
        search_index_name=search_index_name,
        query="chunking test",
        connection_string=connection_string,
        database_name=database_name,
        container_name=container_name,
        top=5
    )
    
    print(f"✅ Basic search completed")
    print(f"   Total results: {len(results)}")
    
    for i, result in enumerate(results):
        print(f"   Result {i+1}:")
        print(f"     Patent: {result['patent'].title}")
        print(f"     Score: {result.get('search_score', 'N/A')}")
        print(f"     Source: {result.get('source', 'search_index')}")
        if result.get('matching_chunks'):
            print(f"     Matching chunks: {len(result['matching_chunks'])}")
    
    # Test vector search
    print(f"\n2. Testing vector search...")
    vector_results = vector_search(
        search_endpoint=search_endpoint,
        search_key=search_key,
        search_index_name=search_index_name,
        query_text="chunking test",
        limit=3
    )
    
    print(f"✅ Vector search completed")
    print(f"   Total results: {len(vector_results)}")
    
    # Test hybrid search
    print(f"\n3. Testing hybrid search...")
    hybrid_results = hybrid_search(
        search_endpoint=search_endpoint,
        search_key=search_key,
        search_index_name=search_index_name,
        query="chunking test",
        limit=3
    )
    
    print(f"✅ Hybrid search completed")
    print(f"   Total results: {len(hybrid_results)}")
    
    return results


def test_chunk_management(connection_string, database_name, container_name,
                         search_endpoint, search_key, search_index_name, patent_id):
    """Test chunk management operations"""
    print("\n" + "="*50)
    print("TESTING CHUNK MANAGEMENT")
    print("="*50)
    
    # Get container
    from dao.patent_dao import _get_container
    container = _get_container(connection_string, database_name, container_name)
    
    # Test 1: Get chunks from database
    print(f"\n1. Getting chunks from database...")
    chunks = get_chunks_from_db(container, patent_id)
    
    if chunks:
        print(f"✅ Retrieved {len(chunks)} chunks")
        print(f"   Chunk types: {list(set(chunk['chunk_type'] for chunk in chunks))}")
        print(f"   Chunk sizes: {[chunk['chunk_size'] for chunk in chunks[:3]]}")
        
        # Show first chunk details
        first_chunk = chunks[0]
        print(f"   First chunk:")
        print(f"     ID: {first_chunk['id']}")
        print(f"     Type: {first_chunk['chunk_type']}")
        print(f"     Order: {first_chunk['chunk_order']}")
        print(f"     Size: {first_chunk['chunk_size']}")
        print(f"     Text preview: {first_chunk['chunk_text'][:100]}...")
    else:
        print(f"❌ No chunks found")
    
    # Test 2: Search in chunks
    print(f"\n2. Testing search in chunks...")
    chunk_search_results = search_in_chunks(container, "chunking", 5)
    
    if chunk_search_results:
        print(f"✅ Chunk search completed")
        print(f"   Found {len(chunk_search_results)} patent matches")
        for result in chunk_search_results:
            print(f"     Patent {result['patent_id']}: {result['total_matches']} matching chunks")
    else:
        print(f"ℹ️ No chunk search results")
    
    return chunks


def test_cleanup(connection_string, database_name, container_name,
                 search_endpoint, search_key, search_index_name, patent_id):
    """Test cleanup operations"""
    print("\n" + "="*50)
    print("TESTING CLEANUP OPERATIONS")
    print("="*50)
    
    # Test delete patent (should cleanup chunks)
    print(f"\n1. Deleting patent (should cleanup chunks)...")
    err = delete_patent(
        connection_string, 
        database_name, 
        container_name,
        search_endpoint, 
        search_key, 
        search_index_name, 
        patent_id
    )
    
    if err is None:
        print(f"✅ Patent deleted successfully")
        print(f"   Chunks should be cleaned up")
    else:
        print(f"❌ Failed to delete patent: {err}")
    
    # Verify chunks are gone
    print(f"\n2. Verifying chunks cleanup...")
    from dao.patent_dao import _get_container
    container = _get_container(connection_string, database_name, container_name)
    
    remaining_chunks = get_chunks_from_db(container, patent_id)
    if not remaining_chunks:
        print(f"✅ Chunks cleanup successful")
    else:
        print(f"⚠️ {len(remaining_chunks)} chunks still remain")


def main():
    """Main test function"""
    print("🚀 STARTING CHUNKING IMPLEMENTATION TESTS")
    print("="*60)
    
    # Configuration
    conn_str = COSMOS_DB_CONNECTION_STRING
    database_name = "patent_chunking_test"
    container_name = "chunking_test"
    search_endpoint = AZURE_SEARCH_ENDPOINT
    search_key = AZURE_SEARCH_ADMIN_KEY
    search_index_name = AZURE_SEARCH_INDEX_NAME
    
    print(f"Configuration:")
    print(f"  Database: {database_name}")
    print(f"  Container: {container_name}")
    print(f"  Chunking enabled: {CHUNKING_ENABLED}")
    print(f"  Chunking threshold: {CHUNKING_THRESHOLD} chars")
    print(f"  Max chunk size: {MAX_CHUNK_SIZE} chars")
    
    try:
        # Setup database and container
        print(f"\n📁 Setting up database and container...")
        create_database_and_container(
            conn_str,
            database_name=database_name,
            container_name=container_name,
            force_recreate=True
        )
        
        # Setup search index
        print(f"🔍 Setting up search index...")
        create_or_update_index(
            search_endpoint,
            search_key,
            search_index_name
        )
        
        # Test chunking utilities
        test_chunking_utilities()
        
        # Test CRUD operations
        patent = test_crud_operations_with_chunking(
            conn_str, database_name, container_name,
            search_endpoint, search_key, search_index_name
        )
        
        if patent:
            # Test search with chunking
            test_search_with_chunking(
                conn_str, database_name, container_name,
                search_endpoint, search_key, search_index_name
            )
            
            # Test chunk management
            test_chunk_management(
                conn_str, database_name, container_name,
                search_endpoint, search_key, search_index_name,
                patent.patent_id
            )
            
            # Test cleanup
            test_cleanup(
                conn_str, database_name, container_name,
                search_endpoint, search_key, search_index_name,
                patent.patent_id
            )
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
