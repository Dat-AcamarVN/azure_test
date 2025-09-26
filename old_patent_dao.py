import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes._generated.models import VectorSearch, VectorSearchAlgorithmConfiguration, \
    HnswAlgorithmConfiguration, HnswParameters, VectorSearchProfile
from azure.search.documents.indexes.models import SimpleField, SearchFieldDataType, SearchField, SearchIndex
from scipy.spatial.distance import cosine
from tenacity import retry, stop_after_attempt, wait_exponential

from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure.cosmos.container import ContainerProxy
from azure.cosmos.database import DatabaseProxy
from azure.cosmos.cosmos_client import ConnectionPolicy
from openai import AzureOpenAI
from openai import APIError as OpenAIRateLimitError

from azure.search.documents import SearchClient

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError

from models.patent_model import PatentInfo, SearchInfo
import config
from azure.cosmos.documents import RetryOptions

# Import chunking utilities
from utilities.chunking_utils import (
    should_chunk_patent, 
    chunk_text_simple, 
    save_chunks_to_db,
    get_chunks_from_db,
    reconstruct_patent_from_chunks
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_TEXT_LENGTH = 8000  # Max chars for OpenAI input
MAX_BATCH_SIZE = 16  # Max texts per OpenAI batch


def _get_container(connection_string: str, database_name: str, container_name: str):
    """Get Cosmos DB container with retry logic"""

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=1, max=10),
           reraise=True)
    def get_container_internal():
        # Tạo ConnectionPolicy và RetryOptions đúng cách cho SDK v4
        policy = ConnectionPolicy()

        # Cách 1: dùng tham số vị trí
        policy.RetryOptions = RetryOptions(3, 10)  # (max_retry_attempts_on_throttled_requests, max_retry_wait_time_in_seconds)

        # Cách 2 (tương đương): set thuộc tính sau khi khởi tạo
        # ro = RetryOptions()
        # ro.max_retry_attempts_on_throttled_requests = 3
        # ro.max_retry_wait_time_in_seconds = 10
        # policy.RetryOptions = ro

        region = getattr(config, "COSMOS_DB_REGION", None)
        if region:
            policy.PreferredLocations = [region]

        client = CosmosClient.from_connection_string(
            connection_string,
            consistency_level="Session",
            connection_policy=policy,
        )

        database = client.get_database_client(database_name)
        container = database.get_container_client(container_name)

        # Ping nhẹ để xác thực kết nối
        container.read()
        return container

    try:
        return get_container_internal()
    except exceptions.CosmosResourceNotFoundError:
        logger.error(f"❌ Database '{database_name}' hoặc container '{container_name}' không tồn tại.")
        raise
    except Exception as e:
        logger.error(f"❌ Error getting container: {e}")
        raise


def _get_search_client(endpoint: str, key: str, index_name: str) -> SearchClient:
    """Get Azure AI Search client"""
    try:
        credential = AzureKeyCredential(key)
        return SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
    except Exception as e:
        logger.error(f"❌ Error getting search client: {e}")
        raise


# def create_database_and_container(connection_string: str, database_name: str, container_name: str,
#                                   force_recreate: bool = False) -> bool:
#     """Create Cosmos DB database and container with vector indexing"""
#     try:
#         client = CosmosClient.from_connection_string(connection_string)
#
#         if force_recreate:
#             try:
#                 client.delete_database(database_name)
#                 logger.info(f"🗑️ Deleted database '{database_name}'")
#             except exceptions.CosmosResourceNotFoundError:
#                 logger.info(f"ℹ️ Database '{database_name}' not found")
#
#         # Create database
#         try:
#             database = client.create_database(database_name)
#             logger.info(f"✅ Created database '{database_name}'")
#         except exceptions.CosmosResourceExistsError:
#             database = client.get_database_client(database_name)
#             logger.info(f"ℹ️ Database '{database_name}' already exists")
#
#         # Define normal indexing policy
#         indexing_policy = {
#             "indexingMode": "consistent",
#             "automatic": True,
#             "includedPaths": [{"path": "/*"}],
#             "excludedPaths": [
#                 {"path": "/_etag/?"},
#                 {"path": "/_ts/?"}
#             ]
#         }
#
#         # Define vector embedding policy
#         vector_embedding_policy = {
#             "vectorEmbeddings": [
#                 {
#                     "path": "/combined_vector",
#                     "dataType": "float32",
#                     "dimensions": config.EMBEDDING_DIMENSION,
#                     "distanceFunction": "cosine"
#                 }
#             ]
#         }
#
#         # Create container (⚠️ bỏ offer_throughput vì bạn dùng serverless)
#         try:
#             container = database.create_container(
#                 id=container_name,
#                 partition_key=PartitionKey(path="/patent_office"),
#                 indexing_policy=indexing_policy,
#                 vector_embedding_policy=vector_embedding_policy
#             )
#             logger.info(f"✅ Created container '{container_name}' with vector search")
#         except exceptions.CosmosResourceExistsError:
#             container = database.get_container_client(container_name)
#             logger.info(f"ℹ️ Container '{container_name}' already exists")
#
#         return True
#     except Exception as e:
#         logger.error(f"❌ Error creating database/container: {e}")
#         return False
#

def create_database_and_container(connection_string: str, database_name: str, container_name: str,
                                  force_recreate: bool = False) -> bool:
    """Create Cosmos DB database and container (no vector search)"""
    try:
        client = CosmosClient.from_connection_string(connection_string)

        if force_recreate:
            try:
                client.delete_database(database_name)
                logger.info(f"🗑️ Deleted database '{database_name}'")
            except exceptions.CosmosResourceNotFoundError:
                logger.info(f"ℹ️ Database '{database_name}' not found")

        # Create database
        try:
            database = client.create_database(database_name)
            logger.info(f"✅ Created database '{database_name}'")
        except exceptions.CosmosResourceExistsError:
            database = client.get_database_client(database_name)
            logger.info(f"ℹ️ Database '{database_name}' already exists")

        # Define normal indexing policy (không vector)
        indexing_policy = {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [
                {"path": "/_etag/?"},
                {"path": "/_ts/?"}
            ]
        }

        # Create container
        try:
            container = database.create_container(
                id=container_name,
                partition_key=PartitionKey(path="/patent_office"),
                indexing_policy=indexing_policy
            )
            logger.info(f"✅ Created container '{container_name}'")
        except exceptions.CosmosResourceExistsError:
            container = database.get_container_client(container_name)
            logger.info(f"ℹ️ Container '{container_name}' already exists")

        return True
    except Exception as e:
        logger.error(f"❌ Error creating database/container: {e}")
        return False


def create_or_update_index(search_endpoint: str, search_key: str, index_name: str):
    """Create or update Azure AI Search index for PatentInfo"""
    try:
        credential = AzureKeyCredential(search_key)
        index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)

        # Xoá index cũ nếu có
        try:
            index_client.delete_index(index_name)
            logger.info(f"🗑️ Deleted existing index '{index_name}'")
        except Exception as delete_error:
            logger.info(f"ℹ️ No existing index to delete: {delete_error}")

        # Define the fields for the index.
        # The vector field "combined_vector" is now defined correctly with its properties.
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="patent_id", type=SearchFieldDataType.String, filterable=True, sortable=True,
                        facetable=True),
            SearchField(name="title", type=SearchFieldDataType.String, searchable=True, analyzer_name="en.lucene"),
            SearchField(name="abstract", type=SearchFieldDataType.String, searchable=True, analyzer_name="en.lucene"),
            SearchField(name="claims", type=SearchFieldDataType.String, searchable=True, analyzer_name="en.lucene"),
            SearchField(name="description", type=SearchFieldDataType.String, searchable=True,
                        analyzer_name="en.lucene"),
            SimpleField(name="assignee", type=SearchFieldDataType.String, filterable=True, facetable=True,
                        sortable=True),
            SimpleField(name="filing_date", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SimpleField(name="inventor", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="language", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="patent_office", type=SearchFieldDataType.String, filterable=True, facetable=True,
                        sortable=True),
            SimpleField(name="priority_date", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SimpleField(name="publication_date", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SimpleField(name="country", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="status", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="application_number", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SimpleField(name="cpc", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="ipc", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="created_at", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SimpleField(name="updated_at", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SimpleField(name="user_id", type=SearchFieldDataType.String, filterable=True, facetable=True),

            # ✅ Fixed: The vector field definition is now correct.
            # It uses SearchFieldDataType.Collection(SearchFieldDataType.Single)
            # and sets the vector search properties directly.
            SearchField(
                name="combined_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=int(config.EMBEDDING_DIMENSION),
                vector_search_profile_name="myHnswProfile",
            ),
        ]

        # ✅ Cấu hình vector search
        # This part of the code was already correct.
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

        # Create the SearchIndex object
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search
        )

        # Create or update the index in Azure AI Search
        index_client.create_or_update_index(index)
        logger.info(f"✅ Search index '{index_name}' created/updated successfully")

    except Exception as e:
        logger.error(f"❌ Error creating/updating search index: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'json'):
            try:
                error_details = e.response.json()
                logger.error(f"❌ Error details: {error_details}")
            except:
                pass
        raise

# ===================== CRUD OPERATIONS =====================

def create_patent(connection_string: str, database_name: str, container_name: str,
                  search_endpoint: str, search_key: str, search_index_name: str,
                  patent: PatentInfo) -> Optional[Exception]:
    """Create new PatentInfo with validation, chunking, and sync to Azure AI Search"""
    try:
        container = _get_container(connection_string, database_name, container_name)
        search_client = _get_search_client(search_endpoint, search_key, search_index_name)

        # Validate patent
        if not patent.patent_id:
            raise ValueError("Patent ID is required")

        # Update timestamp
        patent.update_timestamps()

        # === CHUNKING LOGIC - Đơn giản và thiết thực ===
        combined_text = _prepare_combined_text(patent)
        if should_chunk_patent(combined_text):
            logger.info(f"🔄 Applying chunking to patent {patent.patent_id}")
            
            # Tạo chunks cho các field lớn
            all_chunks = []
            
            if patent.abstract:
                abstract_chunks = chunk_text_simple(patent.abstract, "abstract", patent.patent_id)
                all_chunks.extend(abstract_chunks)
            
            if patent.claims:
                claims_chunks = chunk_text_simple(patent.claims, "claims", patent.patent_id)
                all_chunks.extend(claims_chunks)
            
            if patent.description:
                desc_chunks = chunk_text_simple(patent.description, "description", patent.patent_id)
                all_chunks.extend(desc_chunks)
            
            # Lưu chunks nếu có
            if all_chunks:
                if save_chunks_to_db(container, all_chunks, patent.patent_office or "unknown"):
                    patent.is_chunked = True
                    patent.chunk_count = len(all_chunks)
                    logger.info(f"✅ Patent chunked into {len(all_chunks)} chunks")
                else:
                    logger.warning("⚠️ Failed to save chunks, continuing without chunking")
        
        # Regenerate embeddings
        combined_text = _prepare_combined_text(patent)
        patent.combined_vector = generate_embeddings([combined_text])[0]

        if len(patent.combined_vector) != config.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch: expected {config.EMBEDDING_DIMENSION}, got {len(patent.combined_vector)}"
            )
        # Prepare document
        doc = _prepare_document_for_upload(patent)

        # Insert to Cosmos DB
        container.create_item(body=doc)

        # Sync to Azure AI Search - Fixed: use synchronous call
        try:
            search_client.upload_documents([doc])
            logger.info(f"✅ Synced to Azure AI Search: {patent.patent_id}")
        except Exception as search_error:
            logger.warning(f"⚠️ Failed to sync to Azure AI Search: {search_error}")

        logger.info(f"✅ Created Patent: {patent.patent_id}")
        return None  # success
    except Exception as e:
        logger.error(f"❌ Error creating Patent: {e}")
        return e


def read_patent(connection_string: str, database_name: str, container_name: str, patent_id: str, reconstruct_chunks: bool = True) -> Optional[
    PatentInfo]:
    """Read PatentInfo by ID with optional chunk reconstruction"""
    try:
        container = _get_container(connection_string, database_name, container_name)

        query = "SELECT * FROM c WHERE c.patent_id = @patent_id"
        parameters = [{"name": "@patent_id", "value": patent_id}]

        items = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))

        if items:
            patent_dict = items[0]
            
            # === CHUNKING LOGIC - Auto-reconstruction ===
            if reconstruct_chunks and patent_dict.get("is_chunked"):
                logger.info(f"🔄 Reconstructing patent {patent_id} from chunks")
                
                # Lấy chunks
                chunks = get_chunks_from_db(container, patent_id)
                
                # Reconstruct
                if chunks:
                    patent_dict = reconstruct_patent_from_chunks(patent_dict, chunks)
                    logger.info(f"✅ Patent {patent_id} reconstructed from {len(chunks)} chunks")
            
            # PatentInfo.from_dict now automatically filters out unknown fields
            return PatentInfo.from_dict(patent_dict)
        else:
            logger.warning(f"❌ Patent not found: {patent_id}")
            return None

    except Exception as e:
        logger.error(f"❌ Error reading Patent: {e}")
        return None


def update_patent(connection_string: str, database_name: str, container_name: str,
                  search_endpoint: str, search_key: str, search_index_name: str,
                  patent: PatentInfo) -> Optional[Exception]:
    """Update existing PatentInfo with chunking and sync to Azure AI Search"""
    try:
        container = _get_container(connection_string, database_name, container_name)
        search_client = _get_search_client(search_endpoint, search_key, search_index_name)

        if not patent.patent_id:
            raise ValueError("Patent ID is required")

        # Update timestamp
        patent.update_timestamps()

        # === CHUNKING LOGIC - Re-chunking khi cần ===
        combined_text = _prepare_combined_text(patent)
        if should_chunk_patent(combined_text):
            logger.info(f"🔄 Applying chunking to updated patent {patent.patent_id}")
            
            # Xóa chunks cũ nếu có
            if patent.is_chunked:
                old_chunks = get_chunks_from_db(container, patent.patent_id)
                if old_chunks:
                    for chunk in old_chunks:
                        try:
                            container.delete_item(item=chunk["id"], partition_key=patent.patent_office or "unknown")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to delete old chunk {chunk['id']}: {e}")
            
            # Tạo chunks mới
            all_chunks = []
            
            if patent.abstract:
                abstract_chunks = chunk_text_simple(patent.abstract, "abstract", patent.patent_id)
                all_chunks.extend(abstract_chunks)
            
            if patent.claims:
                claims_chunks = chunk_text_simple(patent.claims, "claims", patent.patent_id)
                all_chunks.extend(claims_chunks)
            
            if patent.description:
                desc_chunks = chunk_text_simple(patent.description, "description", patent.patent_id)
                all_chunks.extend(desc_chunks)
            
            # Lưu chunks mới
            if all_chunks:
                if save_chunks_to_db(container, all_chunks, patent.patent_office or "unknown"):
                    patent.is_chunked = True
                    patent.chunk_count = len(all_chunks)
                    logger.info(f"✅ Updated patent chunked into {len(all_chunks)} chunks")
                else:
                    logger.warning("⚠️ Failed to save new chunks, continuing without chunking")
        else:
            # Nếu không cần chunking, xóa chunks cũ
            if patent.is_chunked:
                old_chunks = get_chunks_from_db(container, patent.patent_id)
                if old_chunks:
                    for chunk in old_chunks:
                        try:
                            container.delete_item(item=chunk["id"], partition_key=patent.patent_office or "unknown")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to delete old chunk {chunk['id']}: {e}")
                
                patent.is_chunked = False
                patent.chunk_count = 0

        # Regenerate embeddings
        combined_text = _prepare_combined_text(patent)
        patent.combined_vector = generate_embeddings([combined_text])[0]

        if len(patent.combined_vector) != config.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch: expected {config.EMBEDDING_DIMENSION}, got {len(patent.combined_vector)}"
            )

        doc = _prepare_document_for_upload(patent)

        # Upsert in Cosmos DB
        container.upsert_item(body=doc)

        # Sync to Azure AI Search - Fixed: use synchronous call
        try:
            search_client.merge_or_upload_documents([doc])
            logger.info(f"✅ Synced to Azure AI Search: {patent.patent_id}")
        except Exception as search_error:
            logger.warning(f"⚠️ Failed to sync to Azure AI Search: {search_error}")

        logger.info(f"✅ Updated Patent: {patent.patent_id}")
        return None
    except Exception as e:
        logger.error(f"❌ Error updating Patent: {e}")
        return e

def delete_patent(connection_string: str, database_name: str, container_name: str,
                  search_endpoint: str, search_key: str, search_index_name: str,
                  patent_id: str) -> Optional[Exception]:
    """Delete PatentInfo by ID with chunk cleanup and sync to Azure AI Search"""
    try:
        container = _get_container(connection_string, database_name, container_name)
        search_client = _get_search_client(search_endpoint, search_key, search_index_name)

        existing = read_patent(connection_string, database_name, container_name, patent_id, reconstruct_chunks=False)
        if not existing:
            raise ValueError(f"Patent not found: {patent_id}")

        # === CHUNKING LOGIC - Cleanup chunks ===
        # Xóa chunks nếu có
        if existing.is_chunked:
            logger.info(f"🗑️ Deleting chunks for patent {patent_id}")
            chunks = get_chunks_from_db(container, patent_id)
            if chunks:
                for chunk in chunks:
                    try:
                        container.delete_item(item=chunk["id"], partition_key=existing.patent_office or "unknown")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to delete chunk {chunk['id']}: {e}")
                logger.info(f"✅ Deleted {len(chunks)} chunks for patent {patent_id}")

        # Delete from Cosmos DB
        container.delete_item(item=patent_id, partition_key=existing.patent_office)

        # Delete from Azure AI Search - Fixed: use synchronous call
        try:
            search_client.delete_documents([{"id": patent_id}])
            logger.info(f"✅ Deleted from Azure AI Search: {patent_id}")
        except Exception as search_error:
            logger.warning(f"⚠️ Failed to delete from Azure AI Search: {search_error}")

        logger.info(f"✅ Deleted Patent: {patent_id}")
        return None
    except Exception as e:
        logger.error(f"❌ Error deleting Patent: {e}")
        return e


def list_all_patents(connection_string: str, database_name: str, container_name: str, top: int = 100) -> List[
    PatentInfo]:
    """List all PatentInfo records"""
    try:
        container = _get_container(connection_string, database_name, container_name)

        query = f"SELECT TOP {top} * FROM c ORDER BY c.created_at DESC"
        items = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))

        patents = []
        for item in items:
            # PatentInfo.from_dict now automatically filters out unknown fields
            patents.append(PatentInfo.from_dict(item))

        return patents

    except Exception as e:
        logger.error(f"❌ Error listing Patents: {e}")
        return []


# ===================== SEARCH OPERATIONS =====================

def basic_search(search_endpoint: str, search_key: str, search_index_name: str,
                 query: str, filters: List[SearchInfo] = [],
                 skip: int = 0, top: int = 10) -> List[Dict[str, Any]]:
    """Basic text search using Azure AI Search (BM25)"""
    try:
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
        results_list = []
        for r in results:
            # Extract search metadata before converting to PatentInfo
            search_score = r.get('@search.score', 0.0)
            # Remove Azure Search metadata fields
            clean_result = {k: v for k, v in r.items() if not k.startswith('@search.')}
            patent = PatentInfo.from_dict(clean_result)
            results_list.append({'patent': patent, 'search_score': search_score})
        return results_list
    except AzureError as ae:
        logger.error(f"❌ Azure error in basic search: {ae}")
        return []
    except Exception as e:
        logger.error(f"❌ Error in basic search: {e}")
        return []


def vector_search(search_endpoint: str, search_key: str, search_index_name: str,
                  query_text: str, vector_field: str = "combined_vector",
                  similarity_threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD, limit: int = 10) -> List[
    Dict[str, Any]]:
    """Vector similarity search using Azure AI Search"""
    try:
        search_client = _get_search_client(search_endpoint, search_key, search_index_name)
        query_embedding = generate_embeddings([query_text])[0]
        vector_query = {
            "vector": query_embedding,
            "k_nearest_neighbors": limit,
            "fields": vector_field,
            "kind": "vector"
        }
        results = search_client.search(
            search_text="*",
            vector_queries=[vector_query],
            top=limit,
            select=["patent_id", "title", "abstract", "claims", "patent_office", "created_at", "updated_at"]
        )

        filtered_results = [r for r in results if r.get('@search.score', 0.0) >= similarity_threshold]
        results_list = []
        for r in filtered_results:
            # Extract search metadata before converting to PatentInfo
            search_score = r.get('@search.score', 0.0)
            # Remove Azure Search metadata fields
            clean_result = {k: v for k, v in r.items() if not k.startswith('@search.')}
            patent = PatentInfo.from_dict(clean_result)
            results_list.append({'patent': patent, 'similarity': search_score})
        return results_list
    except AzureError as ae:
        logger.error(f"❌ Azure error in vector search: {ae}")
        return []
    except Exception as e:
        logger.error(f"❌ Error in vector search: {e}")
        return []


def hybrid_search(search_endpoint: str, search_key: str, search_index_name: str,
                  query: str, vector_field: str = "combined_vector",
                  similarity_threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD, limit: int = 10,
                  text_weight: float = config.HYBRID_SEARCH_TEXT_WEIGHT,
                  vector_weight: float = config.HYBRID_SEARCH_VECTOR_WEIGHT) -> List[Dict[str, Any]]:
    """Hybrid search combining text and vector using Azure AI Search"""
    try:
        search_client = _get_search_client(search_endpoint, search_key, search_index_name)
        query_embedding = generate_embeddings([query])[0]
        vector_query = {
            "vector": query_embedding,
            "k_nearest_neighbors": limit * 2,
            "fields": vector_field,
            "kind": "vector"
        }
        results = search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=limit * 2,
            select=["patent_id", "title", "abstract", "claims", "patent_office", "created_at", "updated_at"],
            query_type="simple"
        )
        final_results = []
        for r in results:
            text_score = r.get('@search.score', 0.0)
            vector_score = r.get('@search.score', 0.0)  # Azure AI Search combines internally
            combined_score = text_score * text_weight + vector_score * vector_weight
            if combined_score >= similarity_threshold:
                # Remove Azure Search metadata fields
                clean_result = {k: v for k, v in r.items() if not k.startswith('@search.')}
                patent = PatentInfo.from_dict(clean_result)
                final_results.append({
                    'patent': patent,
                    'scores': {'text_score': text_score, 'vector_score': vector_score, 'combined_score': combined_score}
                })
        return final_results[:limit]
    except AzureError as ae:
        logger.error(f"❌ Azure error in hybrid search: {ae}")
        return []
    except Exception as e:
        logger.error(f"❌ Error in hybrid search: {e}")
        return []


def semantic_search(search_endpoint: str, search_key: str, search_index_name: str,
                    query: str, vector_field: str = "combined_vector", limit: int = 10) -> List[Dict[str, Any]]:
    """Semantic search using Azure AI Search (vector + text)"""
    try:
        search_client = _get_search_client(search_endpoint, search_key, search_index_name)
        query_embedding = generate_embeddings([query])[0]
        vector_query = {
            "vector": query_embedding,
            "k_nearest_neighbors": limit * 2,
            "fields": vector_field,
            "kind": "vector"
        }
        results = search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=limit,
            select=["patent_id", "title", "abstract", "claims", "patent_office", "created_at", "updated_at"]
        )
        results_list = []
        for r in results:
            # Extract search metadata before converting to PatentInfo
            search_score = r.get('@search.score', 0.0)
            # Remove Azure Search metadata fields
            clean_result = {k: v for k, v in r.items() if not k.startswith('@search.')}
            patent = PatentInfo.from_dict(clean_result)
            results_list.append({'patent': patent, 'semantic_score': search_score})
        return results_list
    except AzureError as ae:
        logger.error(f"❌ Azure error in semantic search: {ae}")
        return []
    except Exception as e:
        logger.error(f"❌ Error in semantic search: {e}")
        return []


def bm25_search(search_endpoint: str, search_key: str, search_index_name: str,
                query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """BM25 search using Azure AI Search"""
    try:
        search_client = _get_search_client(search_endpoint, search_key, search_index_name)
        results = search_client.search(
            search_text=query,
            top=limit,
            select=["patent_id", "title", "abstract", "claims", "patent_office", "created_at", "updated_at"]
        )
        results_list = []
        for r in results:
            # Extract search metadata before converting to PatentInfo
            search_score = r.get('@search.score', 0.0)
            # Remove Azure Search metadata fields
            clean_result = {k: v for k, v in r.items() if not k.startswith('@search.')}
            patent = PatentInfo.from_dict(clean_result)
            results_list.append({'patent': patent, 'score': search_score})
        return results_list
    except AzureError as ae:
        logger.error(f"❌ Azure error in BM25 search: {ae}")
        return []
    except Exception as e:
        logger.error(f"❌ Error in BM25 search: {e}")
        return []


def search_rrf_hybrid(search_endpoint: str, search_key: str, search_index_name: str,
                      query: str, vector_field: str = "combined_vector",
                      similarity_threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD, limit: int = 10,
                      k: int = config.DEFAULT_RRF_K) -> List[Dict[str, Any]]:
    """Hybrid search using Reciprocal Rank Fusion (RRF)"""
    try:
        # Get text results (BM25)
        text_results = list(basic_search(search_endpoint, search_key, search_index_name, query, top=limit * 2))
        text_rank = {res['patent'].patent_id: rank + 1 for rank, res in enumerate(text_results)}

        # Get vector results
        vector_results = vector_search(search_endpoint, search_key, search_index_name,
                                       query_text=query, vector_field=vector_field,
                                       similarity_threshold=similarity_threshold, limit=limit * 2)
        vector_rank = {res['patent'].patent_id: rank + 1 for rank, res in enumerate(vector_results)}

        # Combine using RRF
        all_ids = set(text_rank.keys()) | set(vector_rank.keys())
        rrf_scores = {}
        for doc_id in all_ids:
            r_text = text_rank.get(doc_id, len(all_ids) + 1)
            r_vector = vector_rank.get(doc_id, len(all_ids) + 1)
            rrf_scores[doc_id] = 1 / (k + r_text) + 1 / (k + r_vector)

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:limit]

        # Fetch patents
        id_to_patent = {res['patent'].patent_id: res['patent'] for res in text_results}
        id_to_patent.update({res['patent'].patent_id: res['patent'] for res in vector_results})

        final_results = []
        for doc_id in sorted_ids:
            patent = id_to_patent.get(doc_id)
            if patent:
                final_results.append({'patent': patent, 'rrf_score': rrf_scores[doc_id]})

        return final_results

    except AzureError as ae:
        logger.error(f"❌ Azure error in RRF hybrid search: {ae}")
        return []
    except Exception as e:
        logger.error(f"❌ Error in RRF hybrid search: {e}")
        return []


def search_semantic_reranker(search_endpoint: str, search_key: str, search_index_name: str,
                             query: str, vector_field: str = "combined_vector", limit: int = 10) -> List[
    Dict[str, Any]]:
    """Semantic reranker using Azure AI Search (vector + text)"""
    try:
        search_client = _get_search_client(search_endpoint, search_key, search_index_name)
        query_embedding = generate_embeddings([query])[0]
        vector_query = {
            "vector": query_embedding,
            "k_nearest_neighbors": limit * 2,
            "fields": vector_field,
            "kind": "vector"
        }
        results = search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=limit,
            select=["patent_id", "title", "abstract", "claims", "patent_office", "created_at", "updated_at"]
        )
        final_results = []
        for r in results:
            # Extract search metadata before converting to PatentInfo
            search_score = r.get('@search.score', 0.0)
            # Remove Azure Search metadata fields
            clean_result = {k: v for k, v in r.items() if not k.startswith('@search.')}
            patent = PatentInfo.from_dict(clean_result)
            final_results.append({
                'patent': patent,
                'semantic_score': search_score
            })
        return final_results[:limit]
    except AzureError as ae:
        logger.error(f"❌ Azure error in semantic reranker search: {ae}")
        return []
    except Exception as e:
        logger.error(f"❌ Error in semantic reranker search: {e}")
        return []


# ===================== HELPER METHODS =====================

def _prepare_document_for_upload(patent: PatentInfo) -> Dict[str, Any]:
    """Prepare document for Cosmos DB and Azure AI Search"""
    doc = patent.to_dict()
    doc['id'] = patent.patent_id
    return doc


def _prepare_combined_text(patent: PatentInfo) -> str:
    """Prepare combined text for embeddings with truncation"""
    combined_text = ""
    fields = [patent.title, patent.abstract, patent.claims]
    for field in fields:
        if field and field.strip():
            if combined_text:
                combined_text += " "
            combined_text += field.strip()
    # Truncate to avoid OpenAI token limits
    return combined_text[:MAX_TEXT_LENGTH]


def _build_filter_conditions(filters: List[SearchInfo]) -> str:
    """Build OData filter string for Azure AI Search"""
    conditions = []
    for f in filters:
        if f.search_value:
            # OData eq cho string
            conditions.append(f"{f.search_by} eq '{f.search_value}'")
    return ' and '.join(conditions) if conditions else None


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Azure OpenAI with batch splitting"""
    if not texts:
        return []

    # Split into batches to respect OpenAI limits
    batches = [texts[i:i + MAX_BATCH_SIZE] for i in range(0, len(texts), MAX_BATCH_SIZE)]
    all_embeddings = []

    client = AzureOpenAI(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION
    )

    for batch in batches:
        logger.info(f"🔄 Generating embeddings for {len(batch)} texts...")
        try:
            response = client.embeddings.create(
                input=[text[:MAX_TEXT_LENGTH] for text in batch],  # Truncate each text
                model=config.AZURE_OPENAI_DEPLOYMENT
            )
            all_embeddings.extend([item.embedding for item in response.data])
        except OpenAIRateLimitError as e:
            logger.warning(f"Rate limit hit: {e}. Retrying...")
            raise
        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {e}")
            all_embeddings.extend([[] for _ in batch])

    return all_embeddings