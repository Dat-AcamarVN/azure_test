"""
Comprehensive Chunking Test
Test all CRUD operations and search methods with chunking support
Based on test.py structure but with longer text fields
"""

import json
import logging
from azure.cosmos import CosmosClient

import config
from config import (
    AZURE_SEARCH_ENDPOINT, 
    AZURE_SEARCH_ADMIN_KEY, 
    AZURE_SEARCH_INDEX_NAME, 
    COSMOS_DB_CONNECTION_STRING
)
from dao.patent_dao import setup_change_feed_processor
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

# Import chunking utilities
from utilities.chunking_utils_v2 import (
    save_chunks_to_db,
    get_chunks_from_db, search_in_chunks
)

# Import and configure logging
from logging_config import configure_test_logging
configure_test_logging()

logger = logging.getLogger(__name__)


def create_large_test_patents():
    """Create test patents with long claims and description to trigger chunking"""
    
    # Create normal abstract (not chunked)
    normal_abstract = (
        "This invention relates to a novel system and method for processing large amounts of data "
        "using advanced machine learning algorithms. The system provides real-time analysis capabilities "
        "with high accuracy and efficiency. It includes multiple interconnected components that work together "
        "to achieve optimal performance and reliability."
    )  # ~200 chars - normal size
    
    # Create very long claims (will be chunked)
    long_claims = (
        "The invention relates to a novel system and method for processing large amounts of data. " * 300 +
        "According to one aspect, the system comprises multiple interconnected components that work together " * 250 +
        "to achieve optimal performance and reliability. The method includes several steps for data validation, " * 200 +
        "transformation, and analysis. Each step is carefully designed to handle edge cases and error conditions. " * 150 +
        "The system also includes advanced features such as real-time monitoring, automated scaling, " * 100
    )  # ~30,000 chars - will be chunked
    
    # Create very long description (will be chunked)
    long_description = (
        "Detailed Description of the Invention: This invention represents a significant advancement " * 500 +
        "in the field of data processing and analysis. The system architecture is designed with scalability " * 400 +
        "in mind, allowing it to handle workloads ranging from small datasets to enterprise-level operations. " * 300 +
        "The core components include a data ingestion layer, processing engine, storage system, and API gateway. " * 250 +
        "Each component is built using modern technologies and follows industry best practices for security, " * 200 +
        "performance, and maintainability. The data ingestion layer supports multiple input formats including " * 150 +
        "JSON, XML, CSV, and binary data. It includes validation rules and error handling mechanisms. " * 100 +
        "The processing engine uses a distributed architecture with worker nodes that can be scaled horizontally. " * 200 +
        "It supports both batch and stream processing modes, making it suitable for various use cases. " * 150
    )  # ~60,000 chars - will be chunked
    
    # Create test patents
    test_patents = [
        PatentInfo(
            patent_id="CHUNK_TEST_001",
            title="Advanced Data Processing System with Machine Learning Capabilities",
            abstract=normal_abstract,
            claims=long_claims,
            description=long_description,
            patent_office="USPTO",
            assignee="Tech Innovations Corp",
            inventor="Dr. John Smith",
            filing_date="2024-01-15",
            priority_date="2024-01-15",
            publication_date="2024-07-15",
            country="US",
            status="Published",
            application_number="US20240012345",
            cpc="G06F 17/00",
            ipc="G06F 17/00"
        ),
        PatentInfo(
            patent_id="CHUNK_TEST_002",
            title="Blockchain-Based Intellectual Property Management Platform",
            abstract=normal_abstract + " Blockchain technology integration for secure IP management.",
            claims=long_claims + " Distributed ledger implementation for patent tracking. " * 200,
            description=long_description + " Smart contract functionality for automated IP transactions. " * 300,
            patent_office="EPO",
            assignee="Blockchain IP Solutions Ltd",
            inventor="Alice Johnson",
            filing_date="2024-02-20",
            priority_date="2024-02-20",
            publication_date="2024-08-20",
            country="EP",
            status="Published",
            application_number="EP20240098765",
            cpc="G06Q 50/18",
            ipc="G06Q 50/18"
        ),
        PatentInfo(
            patent_id="CHUNK_TEST_003",
            title="Quantum Computing Algorithm for Patent Analysis",
            abstract=normal_abstract + " Quantum computing principles for advanced patent analysis.",
            claims=long_claims + " Quantum algorithm implementation for patent similarity analysis. " * 250,
            description=long_description + " Quantum circuit design for parallel patent processing. " * 400,
            patent_office="JPO",
            assignee="Quantum Research Institute",
            inventor="Prof. Tanaka",
            filing_date="2024-03-10",
            priority_date="2024-03-10",
            publication_date="2024-09-10",
            country="JP",
            status="Published",
            application_number="JP20240054321",
            cpc="G06N 10/00",
            ipc="G06N 10/00"
        )
    ]
    
    return test_patents


def test_chunking_analysis(patents):
    """Analyze chunking behavior for test patents"""
    print("\n" + "="*60)
    print("CHUNKING ANALYSIS")
    print("="*60)
    
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    for i, patent in enumerate(patents, 1):
        print(f"\n📋 Patent {i}: {patent.title}")
        print(f"   ID: {patent.patent_id}")
        
        fields = [
            ("Abstract", patent.abstract, False),  # Not chunked
            ("Claims", patent.claims, True),       # Will be chunked
            ("Description", patent.description, True)  # Will be chunked
        ]
        
        total_expected_chunks = 0
        for field_name, field_text, will_be_chunked in fields:
            if field_text:
                token_count = len(tokenizer.encode(field_text))
                needs_chunking = will_be_chunked and token_count > MAX_CHUNK_SIZE
                
                if needs_chunking:
                    # Approximate chunks: since chunk_size=200 tokens
                    expected_chunks = (token_count + MAX_CHUNK_SIZE - 1) // MAX_CHUNK_SIZE  # Ceiling division
                    total_expected_chunks += expected_chunks
                    print(f"   {field_name}: {token_count:,} tokens → ~{expected_chunks} chunks (needs chunking)")
                else:
                    if will_be_chunked:
                        print(f"   {field_name}: {token_count:,} tokens → no chunking needed")
                    else:
                        print(f"   {field_name}: {token_count:,} tokens → not chunked (abstract)")
            else:
                print(f"   {field_name}: empty")
        
        print(f"   Total expected chunks: ~{total_expected_chunks}")


def test_search_index_verification(connection_string, database_name, container_name,
                                  search_endpoint, search_key, search_index_name):
    """Test if search index is working and contains data"""
    print("\n" + "="*60)
    print("SEARCH INDEX VERIFICATION")
    print("="*60)
    
    # Test simple wildcard search to see if index has data
    print(f"\n🔍 Testing search index with wildcard search...")
    
    try:
        from dao.patent_dao import _get_search_client
        search_client = _get_search_client(search_endpoint, search_key, search_index_name)
        
        # Test wildcard search
        wildcard_results = search_client.search(
            search_text="*",
            top=5,
            select=["patent_id", "title", "abstract", "patent_office"]
        )
        
        wildcard_list = list(wildcard_results)
        print(f"✅ Wildcard search completed")
        print(f"   Total results: {len(wildcard_list)}")
        
        if wildcard_list:
            print(f"   📊 Index contains data:")
            for i, result in enumerate(wildcard_list):
                print(f"     Result {i+1}:")
                print(f"       Patent ID: {result.get('patent_id', 'N/A')}")
                print(f"       Title: {result.get('title', 'N/A')}")
                print(f"       Office: {result.get('patent_office', 'N/A')}")
                print(f"       Vector: {result.get('combined_vector', 'N/A')}")

                if result.get('abstract'):
                    abstract_preview = result['abstract'][:100] + "..." if len(result['abstract']) > 100 else result['abstract']
                    print(f"       Abstract: {abstract_preview}")
        else:
            print(f"   ⚠️ Index appears to be empty")
            
        # Test specific field search
        print(f"\n🔍 Testing field-specific search...")
        field_results = search_client.search(
            search_text="data",
            top=3,
            select=["patent_id", "title", "abstract"]
        )
        
        field_list = list(field_results)
        print(f"   Search for 'data': {len(field_list)} results")
        
        if field_list:
            for i, result in enumerate(field_list):
                print(f"     Match {i+1}: {result.get('patent_id', 'N/A')} - {result.get('title', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error testing search index: {e}")
        logger.error(f"Search index test failed: {e}", exc_info=True)


def test_crud_operations_with_chunking(connection_string, database_name, container_name, 
                                      search_endpoint, search_key, search_index_name, patents):
    """Test CRUD operations with chunking support"""
    print("\n" + "="*60)
    print("TESTING CRUD OPERATIONS WITH CHUNKING")
    print("="*60)
    
    created_patents = []
    
    # Test 1: Create patents with chunking
    print(f"\n1. Creating {len(patents)} patents with chunking...")
    for i, patent in enumerate(patents, 1):
        print(f"   Creating patent {i}/{len(patents)}: {patent.patent_id}")
        
        err = create_patent(
            connection_string, 
            database_name, 
            container_name,
            search_endpoint, 
            search_key, 
            search_index_name, 
            patent
        )
        
        if err is None:
            print(f"   ✅ Patent {patent.patent_id} created successfully")
            print(f"      Chunk count: {patent.chunk_count}")
            created_patents.append(patent)
        else:
            print(f"   ❌ Failed to create patent {patent.patent_id}: {err}")
    
    if not created_patents:
        print("❌ No patents were created successfully")
        return None
    
    # Test 2: Read patents with reconstruction
    print(f"\n2. Reading patents with reconstruction...")
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    for patent in created_patents:
        print(f"   Reading patent: {patent.patent_id}")
        
        retrieved_patent = read_patent(
            connection_string, 
            database_name, 
            container_name, 
            patent.patent_id,
            reconstruct_chunks=True
        )
        
        if retrieved_patent:
            print(f"   ✅ Patent retrieved successfully")
            print(f"      Chunk count: {retrieved_patent.chunk_count}")

            # In toàn bộ dữ liệu của patent được retrieve để kiểm tra
            print(f"   📊 Full retrieved patent data:")
            print(f"      Patent ID: {retrieved_patent.patent_id}")
            print(f"      ID: {retrieved_patent.id if retrieved_patent.id else 'None'}")
            print(f"      Title: {retrieved_patent.title}")
            print(f"      Abstract: {retrieved_patent.abstract[:100] + '...' if retrieved_patent.abstract and len(retrieved_patent.abstract) > 100 else retrieved_patent.abstract}")
            print(f"      Claims: {retrieved_patent.claims[:100] + '...' if retrieved_patent.claims and len(retrieved_patent.claims) > 100 else retrieved_patent.claims}")
            print(f"      Description: {retrieved_patent.description[:100] + '...' if retrieved_patent.description and len(retrieved_patent.description) > 100 else retrieved_patent.description}")
            print(f"      Patent Office: {retrieved_patent.patent_office}")
            print(f"      Filing Date: {retrieved_patent.filing_date}")
            print(f"      Status: {retrieved_patent.status}")
            print(f"      Assignee: {retrieved_patent.assignee}")
            print(f"      Inventor: {retrieved_patent.inventor}")
            print(f"      Language: {retrieved_patent.language}")
            print(f"      Priority Date: {retrieved_patent.priority_date}")
            print(f"      Publication Date: {retrieved_patent.publication_date}")
            print(f"      Country: {retrieved_patent.country}")
            print(f"      Application Number: {retrieved_patent.application_number}")
            print(f"      CPC: {retrieved_patent.cpc}")
            print(f"      IPC: {retrieved_patent.ipc}")
            print(f"      User ID: {retrieved_patent.user_id}")
            print(f"      Combined Vector: {retrieved_patent.combined_vector[:5] if retrieved_patent.combined_vector else 'None'}...")
            print(f"      Chunk Count: {retrieved_patent.chunk_count}")
            print(f"      Created At: {retrieved_patent.created_at}")
            print(f"      Updated At: {retrieved_patent.updated_at}")
            if hasattr(retrieved_patent, 'extra_fields') and retrieved_patent.extra_fields:
                print(f"      Extra Fields: {retrieved_patent.extra_fields}")

            # Verify reconstruction
            for field in ['abstract', 'claims', 'description']:
                original_text = getattr(patent, field, '') or ''
                retrieved_text = getattr(retrieved_patent, field, '') or ''
                original_tokens = len(tokenizer.encode(original_text))
                retrieved_tokens = len(tokenizer.encode(retrieved_text))
                if original_tokens == retrieved_tokens and original_text == retrieved_text:
                    print(f"      ✅ {field.capitalize()} reconstruction successful: {original_tokens:,} tokens")
                else:
                    print(f"      ⚠️ {field.capitalize()} reconstruction issue: {original_tokens:,} tokens vs {retrieved_tokens:,} tokens (texts differ)")
        else:
            print(f"   ❌ Failed to retrieve patent {patent.patent_id}")
    
    # Test 3: Update patents (should trigger re-chunking)
    print(f"\n3. Updating patents (should trigger re-chunking)...")
    for patent in created_patents:
        print(f"   Updating patent: {patent.patent_id}")
        
        # Add more text to trigger re-chunking
        patent.abstract = patent.abstract + " Additional updated content. " * 50
        patent.claims = patent.claims + " Updated claims content. " * 100
        
        err = update_patent(
            connection_string, 
            database_name, 
            container_name,
            search_endpoint, 
            search_key, 
            search_index_name, 
            patent
        )
        
        if err is None:
            print(f"   ✅ Patent {patent.patent_id} updated successfully")
            print(f"      New chunk count: {patent.chunk_count}")
        else:
            print(f"   ❌ Failed to update patent {patent.patent_id}: {err}")
    
    return created_patents


def test_additional_crud_operations(connection_string, database_name, container_name,
                                  search_endpoint, search_key, search_index_name):
    """Test additional CRUD operations"""
    print("\n" + "="*60)
    print("TESTING ADDITIONAL CRUD OPERATIONS")
    print("="*60)
    
    # Test list_all_patents
    print(f"\n1. Testing list_all_patents...")
    all_patents = list_all_patents(connection_string, database_name, container_name, top=10)
    print(f"   ✅ Retrieved {len(all_patents)} patents")
    if all_patents:
        print(f"   📋 First patent: {all_patents[0].patent_id} - {all_patents[0].title}")
        print(f"   📊 Chunk counts: {[p.chunk_count for p in all_patents[:3]]}")

        # In toàn bộ dữ liệu của first patent để kiểm tra
        first_patent = all_patents[0]
        print(f"   📊 Full data of first patent:")
        print(f"      Patent ID: {first_patent.patent_id}")
        print(f"      ID: {first_patent.id if first_patent.id else 'None'}")
        print(f"      Title: {first_patent.title}")
        print(f"      Abstract: {first_patent.abstract[:100] + '...' if first_patent.abstract and len(first_patent.abstract) > 100 else first_patent.abstract}")
        print(f"      Claims: {first_patent.claims[:100] + '...' if first_patent.claims and len(first_patent.claims) > 100 else first_patent.claims}")
        print(f"      Description: {first_patent.description[:100] + '...' if first_patent.description and len(first_patent.description) > 100 else first_patent.description}")
        print(f"      Patent Office: {first_patent.patent_office}")
        print(f"      Filing Date: {first_patent.filing_date}")
        print(f"      Status: {first_patent.status}")
        print(f"      Assignee: {first_patent.assignee}")
        print(f"      Inventor: {first_patent.inventor}")
        print(f"      Language: {first_patent.language}")
        print(f"      Priority Date: {first_patent.priority_date}")
        print(f"      Publication Date: {first_patent.publication_date}")
        print(f"      Country: {first_patent.country}")
        print(f"      Application Number: {first_patent.application_number}")
        print(f"      CPC: {first_patent.cpc}")
        print(f"      IPC: {first_patent.ipc}")
        print(f"      User ID: {first_patent.user_id}")
        print(f"      Combined Vector: {first_patent.combined_vector[:5] if first_patent.combined_vector else 'None'}...")
        print(f"      Chunk Count: {first_patent.chunk_count}")
        print(f"      Created At: {first_patent.created_at}")
        print(f"      Updated At: {first_patent.updated_at}")
        if hasattr(first_patent, 'extra_fields') and first_patent.extra_fields:
            print(f"      Extra Fields: {first_patent.extra_fields}")

        # In tất cả patents
        print(f"   📊 All {len(all_patents)} patents:")
        for i, p in enumerate(all_patents, 1):
            print(f"\n   📋 === PATENT {i}/{len(all_patents)} ===")
            print(f"      Patent ID: {p.patent_id}")
            print(f"      ID: {p.id if p.id else 'None'}")
            print(f"      Title: {p.title}")
            print(f"      Abstract: {p.abstract[:100] + '...' if p.abstract and len(p.abstract) > 100 else p.abstract}")
            print(f"      Claims: {p.claims[:100] + '...' if p.claims and len(p.claims) > 100 else p.claims}")
            print(f"      Description: {p.description[:100] + '...' if p.description and len(p.description) > 100 else p.description}")
            print(f"      Patent Office: {p.patent_office}")
            print(f"      Filing Date: {p.filing_date}")
            print(f"      Status: {p.status}")
            print(f"      Assignee: {p.assignee}")
            print(f"      Inventor: {p.inventor}")
            print(f"      Language: {p.language}")
            print(f"      Priority Date: {p.priority_date}")
            print(f"      Publication Date: {p.publication_date}")
            print(f"      Country: {p.country}")
            print(f"      Application Number: {p.application_number}")
            print(f"      CPC: {p.cpc}")
            print(f"      IPC: {p.ipc}")
            print(f"      User ID: {p.user_id}")
            print(f"      Combined Vector: {p.combined_vector[:5] if p.combined_vector else 'None'}...")
            print(f"      Chunk Count: {p.chunk_count}")
            print(f"      Created At: {p.created_at}")
            print(f"      Updated At: {p.updated_at}")
            if hasattr(p, 'extra_fields') and p.extra_fields:
                print(f"      Extra Fields: {p.extra_fields}")
            print(f"      === END PATENT {i} ===")
    
    # Test read_patent without reconstruction
    print(f"\n2. Testing read_patent without reconstruction...")
    if all_patents:
        patent_id = all_patents[0].patent_id
        patent_no_reconstruct = read_patent(
            connection_string, database_name, container_name, 
            patent_id, reconstruct_chunks=False
        )
        if patent_no_reconstruct:
            print(f"   ✅ Retrieved patent without reconstruction")
            print(f"   📊 Claims length: {len(patent_no_reconstruct.claims or '')}")
            print(f"   📊 Description length: {len(patent_no_reconstruct.description or '')}")

            # In toàn bộ dữ liệu của patent without reconstruction để kiểm tra
            print(f"   📊 Full patent data (without reconstruction):")
            print(f"      Patent ID: {patent_no_reconstruct.patent_id}")
            print(f"      ID: {patent_no_reconstruct.id if patent_no_reconstruct.id else 'None'}")
            print(f"      Title: {patent_no_reconstruct.title}")
            print(f"      Abstract: {patent_no_reconstruct.abstract[:100] + '...' if patent_no_reconstruct.abstract and len(patent_no_reconstruct.abstract) > 100 else patent_no_reconstruct.abstract}")
            print(f"      Claims: {patent_no_reconstruct.claims[:100] + '...' if patent_no_reconstruct.claims and len(patent_no_reconstruct.claims) > 100 else patent_no_reconstruct.claims}")
            print(f"      Description: {patent_no_reconstruct.description[:100] + '...' if patent_no_reconstruct.description and len(patent_no_reconstruct.description) > 100 else patent_no_reconstruct.description}")
            print(f"      Patent Office: {patent_no_reconstruct.patent_office}")
            print(f"      Filing Date: {patent_no_reconstruct.filing_date}")
            print(f"      Status: {patent_no_reconstruct.status}")
            print(f"      Assignee: {patent_no_reconstruct.assignee}")
            print(f"      Inventor: {patent_no_reconstruct.inventor}")
            print(f"      Language: {patent_no_reconstruct.language}")
            print(f"      Priority Date: {patent_no_reconstruct.priority_date}")
            print(f"      Publication Date: {patent_no_reconstruct.publication_date}")
            print(f"      Country: {patent_no_reconstruct.country}")
            print(f"      Application Number: {patent_no_reconstruct.application_number}")
            print(f"      CPC: {patent_no_reconstruct.cpc}")
            print(f"      IPC: {patent_no_reconstruct.ipc}")
            print(f"      User ID: {patent_no_reconstruct.user_id}")
            print(f"      Combined Vector: {patent_no_reconstruct.combined_vector[:5] if patent_no_reconstruct.combined_vector else 'None'}...")
            print(f"      Chunk Count: {patent_no_reconstruct.chunk_count}")
            print(f"      Created At: {patent_no_reconstruct.created_at}")
            print(f"      Updated At: {patent_no_reconstruct.updated_at}")
            if hasattr(patent_no_reconstruct, 'extra_fields') and patent_no_reconstruct.extra_fields:
                print(f"      Extra Fields: {patent_no_reconstruct.extra_fields}")
    
    return all_patents


def print_full_result_data(result, result_index, total_results):
    """Helper function to print full data of a search result"""
    print(f"\n   📊 === RESULT {result_index + 1}/{total_results} ===")
    print(f"      Patent ID: {result['patent'].patent_id}")
    print(f"      ID: {result['patent'].id if result['patent'].id else 'None'}")
    print(f"      Title: {result['patent'].title}")
    print(f"      Abstract: {result['patent'].abstract[:100] + '...' if result['patent'].abstract and len(result['patent'].abstract) > 100 else result['patent'].abstract}")
    print(f"      Claims: {result['patent'].claims[:100] + '...' if result['patent'].claims and len(result['patent'].claims) > 100 else result['patent'].claims}")
    print(f"      Description: {result['patent'].description[:100] + '...' if result['patent'].description and len(result['patent'].description) > 100 else result['patent'].description}")
    print(f"      Patent Office: {result['patent'].patent_office}")
    print(f"      Filing Date: {result['patent'].filing_date}")
    print(f"      Status: {result['patent'].status}")
    print(f"      Assignee: {result['patent'].assignee}")
    print(f"      Inventor: {result['patent'].inventor}")
    print(f"      Language: {result['patent'].language}")
    print(f"      Priority Date: {result['patent'].priority_date}")
    print(f"      Publication Date: {result['patent'].publication_date}")
    print(f"      Country: {result['patent'].country}")
    print(f"      Application Number: {result['patent'].application_number}")
    print(f"      CPC: {result['patent'].cpc}")
    print(f"      IPC: {result['patent'].ipc}")
    print(f"      User ID: {result['patent'].user_id}")
    print(f"      Chunk Count: {result['patent'].chunk_count}")
    print(f"      Combined Vector: {result['patent'].combined_vector[:5] if result['patent'].combined_vector else 'None'}...")

    # Print specific scores based on result type
    if 'search_score' in result:
        print(f"      Search Score: {result['search_score']}")
    if 'similarity' in result:
        print(f"      Similarity: {result['similarity']}")
    if 'scores' in result:
        scores = result['scores']
        if 'text_score' in scores:
            print(f"      Text Score: {scores['text_score']}")
        if 'vector_score' in scores:
            print(f"      Vector Score: {scores['vector_score']}")
        if 'combined_score' in scores:
            print(f"      Combined Score: {scores['combined_score']}")
    if 'semantic_score' in result:
        print(f"      Semantic Score: {result['semantic_score']}")
    if 'score' in result:
        print(f"      BM25 Score: {result['score']}")
    if 'rrf_score' in result:
        print(f"      RRF Score: {result['rrf_score']}")

    print(f"      Source: {result.get('source', 'N/A')}")

    if 'matching_chunks' in result:
        print(f"      Matching Chunks Count: {len(result['matching_chunks'])}")
        if result['matching_chunks']:
            print(f"      First Chunk Text: {result['matching_chunks'][0]['text'][:100] + '...' if len(result['matching_chunks'][0]['text']) > 100 else result['matching_chunks'][0]['text']}")
            print(f"      First Chunk Field: {result['matching_chunks'][0]['field']}")
            print(f"      First Chunk Index: {result['matching_chunks'][0]['chunk_index']}")
            print(f"      First Chunk Size: {result['matching_chunks'][0]['chunk_size']}")

    print(f"      Created At: {result['patent'].created_at}")
    print(f"      Updated At: {result['patent'].updated_at}")
    if hasattr(result['patent'], 'extra_fields') and result['patent'].extra_fields:
        print(f"      Extra Fields: {result['patent'].extra_fields}")
    print(f"      === END RESULT {result_index + 1} ===")

def test_all_search_methods(connection_string, database_name, container_name,
                           search_endpoint, search_key, search_index_name):
    """Test all search methods in patent_dao_v2"""
    print("\n" + "="*60)
    print("TESTING ALL SEARCH METHODS")
    print("="*60)
    
    # Test 1: Basic search with chunking support
    print(f"\n1. Testing basic search with chunking...")
    search_queries = ["data processing", "blockchain", "quantum"]
    
    for query in search_queries:
        print(f"\n🔍 Basic search for: '{query}'")
        results = basic_search(
            search_endpoint=search_endpoint,
            search_key=search_key,
            search_index_name=search_index_name,
            query=query,
            connection_string=connection_string,
            database_name=database_name,
            container_name=container_name,
            top=5
        )
        print(f"   ✅ Results: {len(results)}")
        if results:
            # In tất cả kết quả
            for i, result in enumerate(results):
                print(f"   📋 Result {i+1}: {result['patent'].patent_id} - {result['patent'].title}")
                print_full_result_data(result, i, len(results))
        else:
            print("   ⚠️ No results found")
    
    # Test 2: Vector search
    print(f"\n2. Testing vector search...")
    vector_queries = ["machine learning", "distributed system", "patent analysis"]
    
    for query in vector_queries:
        print(f"\n🔍 Vector search for: '{query}'")
        vector_results = vector_search(
            search_endpoint=search_endpoint,
            search_key=search_key,
            search_index_name=search_index_name,
            query_text=query,
            limit=5
        )
        print(f"   ✅ Results: {len(vector_results)}")
        if vector_results:
            # In tất cả kết quả
            for i, result in enumerate(vector_results):
                print(f"   📋 Result {i+1}: {result['patent'].patent_id} - {result['patent'].title}")
                print_full_result_data(result, i, len(vector_results))
        else:
            print("   ⚠️ No results found")
    
    # Test 3: Hybrid search
    print(f"\n3. Testing hybrid search...")
    hybrid_queries = ["data processing", "blockchain technology", "quantum computing"]
    
    for query in hybrid_queries:
        print(f"\n🔍 Hybrid search for: '{query}'")
        hybrid_results = hybrid_search(
            search_endpoint=search_endpoint,
            search_key=search_key,
            search_index_name=search_index_name,
            query=query,
            limit=5
        )
        print(f"   ✅ Results: {len(hybrid_results)}")
        if hybrid_results:
            # In tất cả kết quả
            for i, result in enumerate(hybrid_results):
                print(f"   📋 Result {i+1}: {result['patent'].patent_id} - {result['patent'].title}")
                print_full_result_data(result, i, len(hybrid_results))
        else:
            print("   ⚠️ No results found")
    
    # Test 4: Semantic search
    print(f"\n4. Testing semantic search...")
    semantic_queries = ["artificial intelligence", "intellectual property", "algorithm design"]
    
    for query in semantic_queries:
        print(f"\n🔍 Semantic search for: '{query}'")
        semantic_results = semantic_search(
            search_endpoint=search_endpoint,
            search_key=search_key,
            search_index_name=search_index_name,
            query=query,
            limit=5
        )
        print(f"   ✅ Results: {len(semantic_results)}")
        if semantic_results:
            # In tất cả kết quả
            for i, result in enumerate(semantic_results):
                print(f"   📋 Result {i+1}: {result['patent'].patent_id} - {result['patent'].title}")
                print_full_result_data(result, i, len(semantic_results))
        else:
            print("   ⚠️ No results found")
    
    # Test 5: BM25 search
    print(f"\n5. Testing BM25 search...")
    bm25_queries = ["system", "technology", "algorithm"]
    
    for query in bm25_queries:
        print(f"\n🔍 BM25 search for: '{query}'")
        bm25_results = bm25_search(
            search_endpoint=search_endpoint,
            search_key=search_key,
            search_index_name=search_index_name,
            query=query,
            limit=5
        )
        print(f"   ✅ Results: {len(bm25_results)}")
        if bm25_results:
            # In tất cả kết quả
            for i, result in enumerate(bm25_results):
                print(f"   📋 Result {i+1}: {result['patent'].patent_id} - {result['patent'].title}")
                print_full_result_data(result, i, len(bm25_results))
        else:
            print("   ⚠️ No results found")
    
    # Test 6: RRF Hybrid search
    print(f"\n6. Testing RRF Hybrid search...")
    rrf_queries = ["data processing", "blockchain", "quantum"]
    
    for query in rrf_queries:
        print(f"\n🔍 RRF Hybrid search for: '{query}'")
        rrf_results = search_rrf_hybrid(
            search_endpoint=search_endpoint,
            search_key=search_key,
            search_index_name=search_index_name,
            query=query,
            limit=5
        )
        print(f"   ✅ Results: {len(rrf_results)}")
        if rrf_results:
            # In tất cả kết quả
            for i, result in enumerate(rrf_results):
                print(f"   📋 Result {i+1}: {result['patent'].patent_id} - {result['patent'].title}")
                print_full_result_data(result, i, len(rrf_results))
        else:
            print("   ⚠️ No results found")
    
    # Test 7: Semantic Reranker search
    print(f"\n7. Testing Semantic Reranker search...")
    reranker_queries = ["machine learning", "distributed system", "patent analysis"]
    
    for query in reranker_queries:
        print(f"\n🔍 Semantic Reranker search for: '{query}'")
        reranker_results = search_semantic_reranker(
            search_endpoint=search_endpoint,
            search_key=search_key,
            search_index_name=search_index_name,
            query=query,
            limit=5
        )
        print(f"   ✅ Results: {len(reranker_results)}")
        if reranker_results:
            # In tất cả kết quả
            for i, result in enumerate(reranker_results):
                print(f"   📋 Result {i+1}: {result['patent'].patent_id} - {result['patent'].title}")
                print_full_result_data(result, i, len(reranker_results))
        else:
            print("   ⚠️ No results found")
    
    return results




def test_cleanup(connection_string, database_name, container_name,
                 search_endpoint, search_key, search_index_name, patents):
    """Test cleanup operations"""
    print("\n" + "="*60)
    print("TESTING CLEANUP OPERATIONS")
    print("="*60)
    
    total_chunks_before = 0
    
    # Count chunks before deletion
    from dao.patent_dao import _get_container
    container = _get_container(connection_string, database_name, container_name)
    
    for patent in patents:
        chunks = get_chunks_from_db(container, patent.patent_id)
        total_chunks_before += len(chunks)
    
    print(f"📊 Total chunks before cleanup: {total_chunks_before}")
    
    # Delete patents (should cleanup chunks)
    for i, patent in enumerate(patents, 1):
        print(f"\n🗑️ Deleting patent {i}/{len(patents)}: {patent.patent_id}")
        
        err = delete_patent(
            connection_string, 
            database_name, 
            container_name,
            search_endpoint, 
            search_key, 
            search_index_name, 
            patent.patent_id
        )
        
        if err is None:
            print(f"   ✅ Patent deleted successfully")
        else:
            print(f"   ❌ Failed to delete patent: {err}")
    
    # Verify chunks are gone
    print(f"\n🔍 Verifying chunks cleanup...")
    total_chunks_after = 0
    
    for patent in patents:
        remaining_chunks = get_chunks_from_db(container, patent.patent_id)
        total_chunks_after += len(remaining_chunks)
    
    if total_chunks_after == 0:
        print(f"✅ All chunks cleaned up successfully")
    else:
        print(f"⚠️ {total_chunks_after} chunks still remain")


def main():
    """Main test function"""
    print("🚀 STARTING COMPREHENSIVE CHUNKING TESTS")
    print("="*70)
    
    # Configuration
    conn_str = COSMOS_DB_CONNECTION_STRING
    database_name = "patent_chunking_comprehensive_test"
    container_name = "comprehensive_chunking_test"
    search_endpoint = AZURE_SEARCH_ENDPOINT
    search_key = AZURE_SEARCH_ADMIN_KEY
    search_index_name = AZURE_SEARCH_INDEX_NAME
    
    print(f"Configuration:")
    print(f"  Database: {database_name}")
    print(f"  Container: {container_name}")
    print(f"  Chunking enabled: {CHUNKING_ENABLED}")
    print(f"  Chunking threshold: {MAX_CHUNK_SIZE} tokens") # Changed from MAX_TOKENS_THRESHOLD to MAX_CHUNK_SIZE
    
    try:
        # Create test patents with very long text
        print(f"\n📝 Creating test patents with long text fields...")
        test_patents = create_large_test_patents()
        
        # Analyze chunking behavior
        test_chunking_analysis(test_patents)
        
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

        # Setup change feed processor (skip for now)
        # setup_change_feed_processor(conn_str, database_name, container_name, search_endpoint, search_key, search_index_name)

        # Test search index verification
        test_search_index_verification(
            conn_str, database_name, container_name,
            search_endpoint, search_key, search_index_name
        )
        
        # Test CRUD operations
        created_patents = test_crud_operations_with_chunking(
            conn_str, database_name, container_name,
            search_endpoint, search_key, search_index_name,
            test_patents
        )
        
        if created_patents:
            # Test additional CRUD operations
            test_additional_crud_operations(
                conn_str, database_name, container_name,
                search_endpoint, search_key, search_index_name
            )
            
            # Test all search methods
            test_all_search_methods(
                conn_str, database_name, container_name,
                search_endpoint, search_key, search_index_name
            )
            
            # Test cleanup
            test_cleanup(
                conn_str, database_name, container_name,
                search_endpoint, search_key, search_index_name,
                created_patents
            )
        
        print("\n" + "="*70)
        print("🎉 ALL COMPREHENSIVE TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)


def main2():
    """Quick test function - only create and delete operations"""
    print("🚀 STARTING QUICK CREATE/DELETE TEST")
    print("="*50)

    # Configuration
    conn_str = COSMOS_DB_CONNECTION_STRING
    database_name = "patent_chunking_quick_test"
    container_name = "quick_chunking_test"
    search_endpoint = AZURE_SEARCH_ENDPOINT
    search_key = AZURE_SEARCH_ADMIN_KEY
    search_index_name = AZURE_SEARCH_INDEX_NAME

    print(f"Configuration:")
    print(f"  Database: {database_name}")
    print(f"  Container: {container_name}")

    try:
        # Create test patent with long text
        print(f"\n📝 Creating test patent...")
        test_patents = create_large_test_patents()[:1]  # Only first patent
        patent = test_patents[0]

        # Setup database and container
        print(f"📁 Setting up database and container...")
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

        # Test create
        print(f"\n🆕 Creating patent: {patent.patent_id}")
        err = create_patent(
            conn_str, database_name, container_name,
            search_endpoint, search_key, search_index_name,
            patent
        )

        if err is None:
            print(f"   ✅ Patent created successfully")
            print(f"      Chunk count: {patent.chunk_count}")

            # Test delete immediately
            print(f"\n🗑️ Deleting patent: {patent.patent_id}")
            delete_err = delete_patent(
                conn_str, database_name, container_name,
                search_endpoint, search_key, search_index_name,
                patent.patent_id
            )

            if delete_err is None:
                print(f"   ✅ Patent deleted successfully")
            else:
                print(f"   ❌ Failed to delete patent: {delete_err}")

        else:
            print(f"   ❌ Failed to create patent: {err}")

        print("\n" + "="*50)
        print("🎉 QUICK TEST COMPLETED!")
        print("="*50)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.error(f"Quick test failed: {e}", exc_info=True)


if __name__ == "__main__":
    # Uncomment to run full test
    main()

    # Run quick test
    # main2()