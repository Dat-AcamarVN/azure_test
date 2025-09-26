import logging
import time
from typing import List, Dict, Any
from datetime import datetime
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField, SearchFieldDataType, SearchField, SearchIndex,
    VectorSearch, VectorSearchAlgorithmConfiguration, HnswAlgorithmConfiguration,
    HnswParameters, VectorSearchProfile, SemanticConfiguration, SemanticPrioritizedFields, SemanticSearch
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError

from models.patent_model import PatentInfo, SearchInfo, FilterInfoWithPageInput
from dao.patent_dao_v2 import (
    create_patent, read_patent, update_patent, delete_patent, get_all_patents,
    search_with_page, bm25_search, hybrid_search, semantic_search, vector_search,
    create_database_and_container, delete_container, generate_embeddings, _get_search_client, _get_container,
    _prepare_document_for_search, create_search_index_with_semantic
)
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
try:
    from logging_config import configure_test_logging

    configure_test_logging()
except ImportError:
    pass

# Configuration
CONNECTION_STRING = config.COSMOS_DB_CONNECTION_STRING
DATABASE_NAME = "PatentDB_test"
CONTAINER_NAME = "Patents_test"
SEARCH_ENDPOINT = config.AZURE_SEARCH_ENDPOINT
SEARCH_KEY = config.AZURE_SEARCH_ADMIN_KEY
SEARCH_INDEX = config.AZURE_SEARCH_INDEX_NAME
VECTOR_DIMENSIONS = 3072  # Adjust based on your embedding model

# Sample data
SAMPLE_PATENTS = [
    {
        "patent_id": "US12365714B2",
        "title": "Sample Patent 1",
        "abstract": "This is a sample patent abstract for testing.",
        "claims": "Claim 1: A method for testing patent storage. " * 200,
        "description": "Detailed description of the sample patent. " * 300,
        "priority_date": "2023-01-01",
        "assignee": "Test Corp",
        "inventor": "John Doe",
        "language": "en",
        "patent_office": "USPTO",
        "country": "US",
        "status": "Granted",
        "application_number": "US12345678",
        "cpc": "G06F17/30",
        "ipc": "G06F17/00"
    },
    {
        "patent_id": "US98765432B2",
        "title": "Sample Patent 2",
        "abstract": "Another sample patent for search testing.",
        "claims": "Claim 1: A novel system for data processing. " * 150,
        "description": "Detailed description of the second patent. " * 200,
        "priority_date": "2023-02-01",
        "assignee": "Tech Inc",
        "inventor": "Jane Smith",
        "language": "en",
        "patent_office": "USPTO",
        "country": "US",
        "status": "Pending",
        "application_number": "US98765432",
        "cpc": "H04L29/08",
        "ipc": "H04L29/00"
    }
]





def upload_patents_to_search_index(connection_string: str, database_name: str, container_name: str,
                                   endpoint: str, key: str, index_name: str, patent_ids: List[str] = None) -> bool:
    """Upload patents from Cosmos DB to Azure Search index."""
    try:
        container = _get_container(connection_string, database_name, container_name)
        search_client = _get_search_client(endpoint, key, index_name)
        patents = []
        if patent_ids:
            for pid in patent_ids:
                patent = read_patent(connection_string, database_name, container_name, pid)
                if patent:
                    patents.append(patent)
        else:
            patents = get_all_patents(connection_string, database_name, container_name)
        docs = []
        for patent in patents:
            doc = _prepare_document_for_search(patent)
            doc['combined_vector'] = \
            generate_embeddings([f"{patent.title or ''} {patent.abstract or ''} {patent.claims or ''}"[:8000]])[0]
            docs.append(doc)
        if docs:
            search_client.upload_documents(docs)
            logger.info(f"✅ Uploaded {len(docs)} documents to Search index")
        return True
    except Exception as e:
        logger.error(f"❌ Error uploading to search index: {e}")
        return False


def test_create_search_index():
    """Test creating search index with semantic configuration"""
    logger.info("=== Testing create_search_index ===")
    success = create_search_index_with_semantic(SEARCH_ENDPOINT, SEARCH_KEY, SEARCH_INDEX)
    logger.info(f"Search index creation: {'✅ Success' if success else '❌ Failed'}")
    time.sleep(2)  # Wait for index to be ready


def test_create_database_and_container():
    """Test creating database and container"""
    logger.info("=== Testing create_database_and_container ===")
    success = create_database_and_container(
        CONNECTION_STRING, DATABASE_NAME, CONTAINER_NAME, force_recreate=True
    )
    logger.info(f"Result: {'✅ Success' if success else '❌ Failed'}")
    time.sleep(2)  # Wait for database/container to be ready


def test_create_patent():
    """Test creating patents with chunking and embeddings"""
    logger.info("=== Testing create_patent ===")
    for patent_data in SAMPLE_PATENTS:
        patent = PatentInfo.from_dict(patent_data)
        success = create_patent(CONNECTION_STRING, DATABASE_NAME, CONTAINER_NAME, patent)
        logger.info(f"Created patent {patent.patent_id}: {'✅ Success' if success else '❌ Failed'}")

    time.sleep(2)  # Wait for data to be indexed


def test_read_patent():
    """Test reading a patent with reconstructed chunks"""
    logger.info("=== Testing read_patent ===")
    patent_id = SAMPLE_PATENTS[0]["patent_id"]
    patent = read_patent(CONNECTION_STRING, DATABASE_NAME, CONTAINER_NAME, patent_id)
    if patent:
        logger.info(f"✅ Retrieved patent {patent_id} with id={patent.id}")
        logger.info(f"Title: {patent.title}")
        logger.info(f"Claims length: {len(patent.claims or '')}")
        logger.info(f"Description length: {len(patent.description or '')}")
        assert patent.claims is not None, "Claims should be reconstructed"
        assert patent.description is not None, "Description should be reconstructed"
    else:
        logger.error(f"❌ Failed to read patent {patent_id}")
        assert False, f"Patent {patent_id} not found"
    time.sleep(1)


def test_update_patent():
    """Test updating a patent"""
    logger.info("=== Testing update_patent ===")
    patent_id = SAMPLE_PATENTS[0]["patent_id"]
    patent = read_patent(CONNECTION_STRING, DATABASE_NAME, CONTAINER_NAME, patent_id)
    if patent:
        patent.title = "Updated Sample Patent 1"
        patent.claims = "Updated claim text. " * 100
        patent.description = "Updated description text. " * 150
        success = update_patent(CONNECTION_STRING, DATABASE_NAME, CONTAINER_NAME, patent)
        logger.info(f"Updated patent {patent_id}: {'✅ Success' if success else '❌ Failed'}")
        if success:
            updated_patent = read_patent(CONNECTION_STRING, DATABASE_NAME, CONTAINER_NAME, patent_id)

        updated_patent = read_patent(CONNECTION_STRING, DATABASE_NAME, CONTAINER_NAME, patent_id)
        if updated_patent:
            logger.info(f"Updated title: {updated_patent.title}")
            logger.info(f"Claims length: {len(updated_patent.claims or '')}")
            logger.info(f"Description length: {len(updated_patent.description or '')}")
            assert updated_patent.title == "Updated Sample Patent 1", "Title should be updated"
        else:
            assert False, f"Updated patent {patent_id} not found"
    else:
        logger.error(f"❌ Cannot update: Patent {patent_id} not found")
        assert False, f"Patent {patent_id} not found"
    time.sleep(2)


def test_get_all_patents():
    """Test retrieving all patents"""
    logger.info("=== Testing get_all_patents ===")
    patents = get_all_patents(CONNECTION_STRING, DATABASE_NAME, CONTAINER_NAME)
    logger.info(f"Retrieved {len(patents)} patents")
    for patent in patents:
        logger.info(f"Patent {patent.patent_id}: Title={patent.title}, Claims length={len(patent.claims or '')}")
        assert patent.claims is not None, f"Claims for {patent.patent_id} should be reconstructed"
        assert patent.description is not None, f"Description for {patent.patent_id} should be reconstructed"
    assert len(patents) >= len(SAMPLE_PATENTS), "Should retrieve at least the sample patents"
    time.sleep(1)


def test_bm25_search():
    """Test BM25 search"""
    logger.info("=== Testing bm25_search ===")
    search_text = "sample patent"
    filters = [SearchInfo(search_by="language", search_value="en")]
    results = bm25_search(SEARCH_ENDPOINT, SEARCH_KEY, SEARCH_INDEX, search_text, filters, limit=5, skip=0)
    logger.info(f"Found {len(results)} results")
    for result in results:
        patent = result['patent']
        logger.info(f"Patent {patent.patent_id}: BM25 Score={result['bm25_score']}")
    time.sleep(1)


def test_hybrid_search():
    """Test hybrid search"""
    logger.info("=== Testing hybrid_search ===")
    search_text = "sample patent"
    query_vector = generate_embeddings([search_text])[0]
    filters = [SearchInfo(search_by="language", search_value="en")]
    results = hybrid_search(
        SEARCH_ENDPOINT, SEARCH_KEY, SEARCH_INDEX, search_text, query_vector,
        filters, limit=5, skip=0
    )
    logger.info(f"Found {len(results)} results")
    for result in results:
        patent = result['patent']
        logger.info(f"Patent {patent.patent_id}: Hybrid Score={result['hybrid_score']}")
    time.sleep(1)


def test_semantic_search():
    """Test semantic search"""
    logger.info("=== Testing semantic_search ===")
    search_text = "sample patent"
    filters = [SearchInfo(search_by="language", search_value="en")]
    results = semantic_search(SEARCH_ENDPOINT, SEARCH_KEY, SEARCH_INDEX, search_text, filters, limit=5, skip=0)
    logger.info(f"Found {len(results)} results")
    for result in results:
        patent = result['patent']
        logger.info(f"Patent {patent.patent_id}: Semantic Score={result['semantic_score']}")
    time.sleep(1)


def test_vector_search():
    """Test vector search"""
    logger.info("=== Testing vector_search ===")
    search_text = "sample patent"
    filters = [SearchInfo(search_by="language", search_value="en")]
    results = vector_search(SEARCH_ENDPOINT, SEARCH_KEY, SEARCH_INDEX, search_text, filters, limit=5, skip=0)
    logger.info(f"Found {len(results)} results")
    for result in results:
        patent = result['patent']
        logger.info(f"Patent {patent.patent_id}: Vector Score={result['vector_score']}")
    time.sleep(1)


def test_search_with_page():
    """Test search_with_page with different search types"""
    logger.info("=== Testing search_with_page ===")
    filter_input = FilterInfoWithPageInput(
        page_number=1,
        page_size=5,
        sort_by="created_at",
        sort_order="desc",
        search_infos=[
            SearchInfo(search_by="query", search_value="sample patent"),
            SearchInfo(search_by="language", search_value="en")
        ],
        search_type="hybrid"
    )

    # Test hybrid search
    result = search_with_page(
        CONNECTION_STRING, DATABASE_NAME, CONTAINER_NAME,
        SEARCH_ENDPOINT, SEARCH_KEY, SEARCH_INDEX, filter_input
    )
    logger.info(f"Hybrid search: Found {len(result['results'])} results, Total={result['total_count']}")
    for res in result['results']:
        patent = res['patent']
        logger.info(f"Patent {patent.patent_id}: Rerank Score={res.get('rerank_score', 'N/A')}")

    # Test semantic search
    filter_input.search_type = "semantic"
    result = search_with_page(
        CONNECTION_STRING, DATABASE_NAME, CONTAINER_NAME,
        SEARCH_ENDPOINT, SEARCH_KEY, SEARCH_INDEX, filter_input
    )
    logger.info(f"Semantic search: Found {len(result['results'])} results, Total={result['total_count']}")
    for res in result['results']:
        patent = res['patent']
        logger.info(f"Patent {patent.patent_id}: Rerank Score={res.get('rerank_score', 'N/A')}")

    # Test BM25 search
    filter_input.search_type = "bm25"
    result = search_with_page(
        CONNECTION_STRING, DATABASE_NAME, CONTAINER_NAME,
        SEARCH_ENDPOINT, SEARCH_KEY, SEARCH_INDEX, filter_input
    )
    logger.info(f"BM25 search: Found {len(result['results'])} results, Total={result['total_count']}")
    for res in result['results']:
        patent = res['patent']
        logger.info(f"Patent {patent.patent_id}: Rerank Score={res.get('rerank_score', 'N/A')}")

    # Test vector search
    filter_input.search_type = "vector"
    result = search_with_page(
        CONNECTION_STRING, DATABASE_NAME, CONTAINER_NAME,
        SEARCH_ENDPOINT, SEARCH_KEY, SEARCH_INDEX, filter_input
    )
    logger.info(f"Vector search: Found {len(result['results'])} results, Total={result['total_count']}")
    for res in result['results']:
        patent = res['patent']
        logger.info(f"Patent {patent.patent_id}: Rerank Score={res.get('rerank_score', 'N/A')}")
    time.sleep(1)


def test_delete_patent():
    """Test deleting a patent"""
    logger.info("=== Testing delete_patent ===")
    patent_id = SAMPLE_PATENTS[0]["patent_id"]
    success = delete_patent(CONNECTION_STRING, DATABASE_NAME, CONTAINER_NAME, patent_id)
    logger.info(f"Deleted patent {patent_id}: {'✅ Success' if success else '❌ Failed'}")
    assert success, f"Failed to delete patent {patent_id}"
    time.sleep(1)


def test_delete_container():
    """Test deleting container"""
    logger.info("=== Testing delete_container ===")
    success = delete_container(CONNECTION_STRING, DATABASE_NAME, CONTAINER_NAME)
    logger.info(f"Deleted container {CONTAINER_NAME}: {'✅ Success' if success else '❌ Failed'}")
    assert success, f"Failed to delete container {CONTAINER_NAME}"
    time.sleep(1)


def run_all_tests():
    """Run all tests sequentially without manual input"""
    logger.info("🚀 Starting Automated Tests for patent_dao_v2.py")

    # Step 0: Create search index
    test_create_search_index()

    # Step 1: Create database and container
    test_create_database_and_container()

    # Step 2: Test create_patent
    test_create_patent()

    # Step 3: Test read_patent
    test_read_patent()

    # Step 4: Test update_patent
    test_update_patent()

    # Step 5: Test get_all_patents
    test_get_all_patents()

    # Step 6: Test searches
    test_bm25_search()
    test_hybrid_search()
    test_semantic_search()
    test_vector_search()
    test_search_with_page()

    # Step 7: Test delete_patent
    test_delete_patent()

    # Step 8: Test delete_container
    test_delete_container()

    logger.info("🏁 All tests completed")


if __name__ == "__main__":
    run_all_tests()
