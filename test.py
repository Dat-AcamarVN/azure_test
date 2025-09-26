import json
import logging

from config import AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY, AZURE_SEARCH_INDEX_NAME, COSMOS_DB_CONNECTION_STRING
from old_patent_dao import create_patent, list_all_patents, basic_search, vector_search, \
    hybrid_search, semantic_search, bm25_search, search_rrf_hybrid, search_semantic_reranker, \
    create_database_and_container, create_or_update_index
from models.patent_model import PatentInfo

# Import and configure logging
from logging_config import configure_test_logging
configure_test_logging()

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # ================== CONFIG ==================
    # Sử dụng connection string từ config.py thay vì hardcode
    conn_str = COSMOS_DB_CONNECTION_STRING
    database_name = "patent_info_test"
    container_name = "testing"
    search_endpoint = AZURE_SEARCH_ENDPOINT
    search_key = AZURE_SEARCH_ADMIN_KEY
    search_index_name = AZURE_SEARCH_INDEX_NAME
    blob_name="sample_patents.json"
    
    # ================== LOAD TEST DATA ==================
    with open(blob_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    patent = PatentInfo.from_dict(data[0])  # lấy 1 record để test

    # Convert toàn bộ list JSON thành list PatentInfo
    patent_list = [PatentInfo.from_dict(d) for d in data]

    # ================== TEST CRUD ==================

    create_database_and_container(
        conn_str,
        database_name=database_name,
        container_name=container_name,
        force_recreate=True  # xóa nếu đã có, tạo lại từ đầu
    )
    create_or_update_index(
        search_endpoint,
        search_key,
        search_index_name
    )

    print("\n=== CREATE PATENTS ===")
    for patent in patent_list:
        err = create_patent(
            conn_str,
            database_name,
            container_name,
            search_endpoint,
            search_key,
            search_index_name,
            patent
        )
        if err:
            print(f"❌ Failed to insert patent {patent.patent_id}: {err}")
        else:
            print(f"✅ Inserted patent {patent.patent_id}")

    print("Result:", "OK" if err is None else f"FAILED: {err}")

    # print("\n=== READ PATENT ===")
    # p = read_patent(conn_str, database_name, container_name, patent.patent_id)
    # print("Result:", p.to_dict() if p else "NOT FOUND")
    #
    # print("\n=== UPDATE PATENT ===")
    # patent.title = patent.title + " (updated)"
    # err = update_patent(conn_str, database_name, container_name,
    #                     search_endpoint, search_key, search_index_name,
    #                     patent)
    # print("Result:", "OK" if err is None else f"FAILED: {err}")
    #
    print("\n=== LIST ALL PATENTS ===")
    all_patents = list_all_patents(conn_str, database_name, container_name, top=5)
    for p in all_patents:
        print(p.to_dict())

    # ================== TEST SEARCH ==================
    print("\n=== BASIC SEARCH ===")
    results = basic_search(search_endpoint, search_key, search_index_name,
                           query="battery", filters=[], top=5)
    for r in results:
        print(r)

    print("\n=== VECTOR SEARCH ===")
    results = vector_search(search_endpoint, search_key, search_index_name,
                            query_text="Blockchain-Based Intellectual Property Management System", limit=5)
    for r in results:
        print(r)

    print("\n=== HYBRID SEARCH ===")
    results = hybrid_search(search_endpoint, search_key, search_index_name,
                            query="battery", limit=5)
    for r in results:
        print(r)

    print("\n=== SEMANTIC SEARCH ===")
    results = semantic_search(search_endpoint, search_key, search_index_name,
                              query="battery", limit=5)
    for r in results:
        print(r)

    print("\n=== BM25 SEARCH ===")
    results = bm25_search(search_endpoint, search_key, search_index_name,
                          query="battery", limit=5)
    for r in results:
        print(r)

    print("\n=== RRF HYBRID SEARCH ===")
    results = search_rrf_hybrid(search_endpoint, search_key, search_index_name,
                                query="battery", limit=5)
    for r in results:
        print(r)

    print("\n=== SEMANTIC RERANKER SEARCH ===")
    results = search_semantic_reranker(search_endpoint, search_key, search_index_name,
                                       query="battery", limit=5)
    for r in results:
        print(r)

    # ================== DELETE ==================
    # print("\n=== DELETE PATENT ===")
    # err = delete_patent(conn_str, database_name, container_name,
    #                     search_endpoint, search_key, search_index_name,
    #                     patent.patent_id)
    # print("Result:", "OK" if err is None else f"FAILED: {err}")
