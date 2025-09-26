import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes._generated.models import VectorSearch, VectorSearchAlgorithmConfiguration, \
    HnswAlgorithmConfiguration, HnswParameters, VectorSearchProfile, SemanticConfiguration, SemanticPrioritizedFields, \
    SemanticField, SemanticSearch
from azure.search.documents.indexes.models import SimpleField, SearchFieldDataType, SearchField, SearchIndex, \
    SearchableField
from azure.search.documents.models import VectorQuery
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from openai import AzureOpenAI
from openai import APIError as OpenAIRateLimitError
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import CrossEncoder
import copy

from models.patent_model import PatentInfo, SearchInfo, FilterInfoWithPageInput
import config
from utilities.chunking_utils_v2 import (
    should_chunk_field,
    create_field_chunks,
    save_chunks_to_db,
    get_chunks_from_db,
    reconstruct_field_from_chunks,
    delete_chunks_for_patent,
    search_in_chunks
)

# Get logger from logging config
try:
    from logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Constants
MAX_TEXT_LENGTH = 8000  # Max chars for OpenAI input
MAX_BATCH_SIZE = 16  # Max texts per OpenAI batch
VECTOR_DIMENSIONS = 3072  # Azure OpenAI embedding dimensions

# Re-ranker model
reranker = CrossEncoder('BAAI/bge-reranker-large')

def _get_container(connection_string: str, database_name: str, container_name: str):
    """Get Cosmos DB container"""
    try:
        client = CosmosClient.from_connection_string(connection_string, consistency_level="Session")
        database = client.get_database_client(database_name)
        container = database.get_container_client(container_name)
        container.read()  # Verify container exists
        return container
    except exceptions.CosmosResourceNotFoundError:
        logger.error(f"❌ Database '{database_name}' or container '{container_name}' does not exist.")
        raise
    except Exception as e:
        logger.error(f"❌ Error getting container: {e}")
        raise

def _get_search_client(endpoint: str, key: str, index_name: str) -> 'SearchClient':
    """Get Azure AI Search client"""
    try:
        credential = AzureKeyCredential(key)
        return SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
    except Exception as e:
        logger.error(f"❌ Error getting search client: {e}")
        raise

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using Azure OpenAI with batch splitting.
    """
    if not texts:
        return []
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
                input=[text[:MAX_TEXT_LENGTH] for text in batch],
                model=config.AZURE_OPENAI_DEPLOYMENT
            )
            embeddings = [item.embedding for item in response.data]
            for emb in embeddings:
                if len(emb) != VECTOR_DIMENSIONS:
                    logger.error(f"❌ Embedding dimension mismatch: expected {VECTOR_DIMENSIONS}, got {len(emb)}")
                    all_embeddings.append([])
                else:
                    all_embeddings.append(emb)
        except OpenAIRateLimitError as e:
            logger.error(f"❌ Rate limit hit: {e}")
            all_embeddings.extend([[] for _ in batch])
        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {e}")
            all_embeddings.extend([[] for _ in batch])
    return all_embeddings

def _prepare_combined_text(patent: PatentInfo) -> str:
    """
    Prepare combined text for embeddings with truncation.
    """
    combined_text = ""
    fields = [patent.title, patent.abstract, patent.claims]
    for field in fields:
        if field and field.strip():
            if combined_text:
                combined_text += " "
            combined_text += field.strip()
    return combined_text[:MAX_TEXT_LENGTH]

def generate_patent_embedding(patent: PatentInfo) -> List[float]:
    """
    Generate embedding for patent from title + abstract + claims.
    Check size before calling API; truncate tail if too long.

    Args:
        patent (PatentInfo): PatentInfo object to generate embedding.

    Returns:
        List[float]: Embedding vector of 3072 dimensions.
    """
    try:
        combined_text = _prepare_combined_text(patent)
        max_length = MAX_TEXT_LENGTH
        if len(combined_text) > max_length:
            logger.warning(f"⚠️ Combined text exceeds max length ({len(combined_text)} > {max_length}), truncating tail")
            combined_text = combined_text[:max_length]
        embedding = generate_embeddings([combined_text])[0]
        if not embedding or len(embedding) != VECTOR_DIMENSIONS:
            logger.error(f"❌ Invalid embedding for patent {patent.patent_id}: empty or wrong dimension")
            return [0.0] * VECTOR_DIMENSIONS
        return embedding
    except Exception as e:
        logger.error(f"❌ Error generating patent embedding for {patent.patent_id}: {e}")
        return [0.0] * VECTOR_DIMENSIONS

def _prepare_document_for_upload(patent: PatentInfo) -> Dict[str, Any]:
    """
    Prepare document for Cosmos DB and Azure AI Search.
    """
    doc = patent.to_dict()
    doc['id'] = patent.id  # Use UUID for document ID
    doc['partition_key'] = patent.partition_key or (
        patent.priority_date[:7] if patent.priority_date and len(patent.priority_date) >= 7 else "unknown"
    )
    return doc

def _prepare_document_for_search(patent: PatentInfo) -> Dict[str, Any]:
    """
    Prepare document specifically for Azure AI Search (essential and filterable fields + title, abstract).
    """
    doc = patent.to_dict()
    doc['id'] = patent.id  # Use UUID for document ID
    allowed_fields = {
        'id', 'patent_id', 'partition_key', 'combined_vector',
        'title', 'abstract',  # Added for full-text search
        'filing_date', 'assignee', 'cpc', 'ipc'  # Filterable fields
    }
    filtered_doc = {key: value for key, value in doc.items() if key in allowed_fields}
    return filtered_doc

def create_search_index_with_semantic(endpoint: str, key: str, index_name: str) -> bool:
    """
    Create Azure AI Search index with vector and semantic capabilities (including title, abstract).
    """
    try:
        index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        # Check if index exists and delete it to avoid conflicts
        try:
            index_client.get_index(index_name)
            index_client.delete_index(index_name)
            logger.info(f"🗑️ Deleted existing index '{index_name}' to avoid conflicts")
            time.sleep(2)  # Wait for deletion to propagate
        except Exception:
            logger.info(f"ℹ️ Index '{index_name}' does not exist, proceeding to create new index")

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="patent_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="partition_key", type=SearchFieldDataType.String, filterable=True),
            # Searchable fields
            SearchableField(name="title", type=SearchFieldDataType.String, searchable=True, filterable=True),
            SearchableField(name="abstract", type=SearchFieldDataType.String, searchable=True),
            # Filterable fields
            SimpleField(name="filing_date", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SimpleField(name="assignee", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="cpc", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="ipc", type=SearchFieldDataType.String, filterable=True),
            # Vector field
            SearchField(
                name="combined_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=VECTOR_DIMENSIONS,
                vector_search_profile_name="myHnswProfile"
            )
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw",
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric="cosine"
                    )
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw"
                )
            ]
        )

        semantic_config = SemanticConfiguration(
            name="default",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="abstract")]
            )
        )

        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=SemanticSearch(configurations=[semantic_config])
        )
        index_client.create_index(index)
        logger.info(f"✅ Created search index '{index_name}' with essential, filterable, and searchable fields")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating search index: {e}")
        return False

def create_patent(connection_string: str, database_name: str, container_name: str, patent: PatentInfo) -> bool:
    """
    Create a patent and its chunks in Cosmos DB with embedding.

    Args:
        connection_string (str): Cosmos DB connection string.
        database_name (str): Database name.
        container_name (str): Container name.
        patent (PatentInfo): PatentInfo object to insert.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        container = _get_container(connection_string, database_name, container_name)
        patent.combined_vector = generate_patent_embedding(patent)
        partition_key = patent.partition_key
        all_chunks = []
        if should_chunk_field(patent.claims):
            claims_chunks = create_field_chunks(patent.patent_id, patent.id, "claims", patent.claims, partition_key)
            all_chunks.extend(claims_chunks)
            patent.claims = None
        if should_chunk_field(patent.description):
            desc_chunks = create_field_chunks(patent.patent_id, patent.id, "description", patent.description, partition_key)
            all_chunks.extend(desc_chunks)
            patent.description = None
        doc = _prepare_document_for_upload(patent)
        container.upsert_item(body=doc)
        logger.info(f"✅ Created PatentInfo {patent.patent_id} with id {patent.id}")
        if all_chunks:
            success = save_chunks_to_db(container, all_chunks)
            if success:
                logger.info(f"✅ Inserted {len(all_chunks)} chunks for {patent.patent_id}")
            else:
                logger.warning(f"⚠️ Failed to insert some chunks for {patent.patent_id}")
        # Sync essential fields to Azure AI Search
        search_client = _get_search_client(config.AZURE_SEARCH_ENDPOINT, config.AZURE_SEARCH_ADMIN_KEY, config.AZURE_SEARCH_INDEX_NAME)
        search_doc = _prepare_document_for_search(patent)
        search_client.upload_documents([search_doc])
        logger.info(f"✅ Synced essential fields for {patent.patent_id} to Azure AI Search")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating patent {patent.patent_id}: {e}")
        return False


def read_patent(connection_string: str, database_name: str, container_name: str, patent_id: str) -> Optional[
    PatentInfo]:
    """
    Read a patent from Cosmos DB by patent_id, reconstructing claims and description from chunks.

    Args:
        connection_string (str): Cosmos DB connection string.
        database_name (str): Database name.
        container_name (str): Container name.
        patent_id (str): Patent ID to retrieve.

    Returns:
        Optional[PatentInfo]: PatentInfo object if found, None otherwise.
    """
    try:
        container = _get_container(connection_string, database_name, container_name)
        # First, get partition_key for the patent
        query = "SELECT c.partition_key FROM c WHERE c.patent_id = @patent_id AND NOT IS_DEFINED(c.field)"
        params = [{"name": "@patent_id", "value": patent_id}]
        items = list(container.query_items(query=query, parameters=params, enable_cross_partition_query=True))
        if not items:
            logger.info(f"ℹ️ Patent {patent_id} not found")
            return None
        partition_key = items[0]['partition_key']

        # Retrieve patent with partition_key
        query = "SELECT * FROM c WHERE c.patent_id = @patent_id AND NOT IS_DEFINED(c.field) AND c.partition_key = @partition_key"
        params = [
            {"name": "@patent_id", "value": patent_id},
            {"name": "@partition_key", "value": partition_key}
        ]
        items = list(container.query_items(query=query, parameters=params, enable_cross_partition_query=False))
        if not items:
            logger.info(f"ℹ️ Patent {patent_id} not found")
            return None

        patent = PatentInfo.from_dict(items[0])

        # Reconstruct claims from chunks
        claims_chunks = get_chunks_from_db(container, patent_id, field="claims", partition_key=partition_key)
        if claims_chunks:
            patent.claims = reconstruct_field_from_chunks(claims_chunks)

        # Reconstruct description from chunks
        desc_chunks = get_chunks_from_db(container, patent_id, field="description", partition_key=partition_key)
        if desc_chunks:
            patent.description = reconstruct_field_from_chunks(desc_chunks)

        logger.info(f"✅ Retrieved patent {patent_id} with reconstructed fields")
        return patent
    except Exception as e:
        logger.error(f"❌ Error retrieving patent {patent_id}: {e}")
        return None

def update_patent(connection_string: str, database_name: str, container_name: str, patent: PatentInfo) -> bool:
    """
    Update a patent and its chunks in Cosmos DB.

    Args:
        connection_string (str): Cosmos DB connection string.
        database_name (str): Database name.
        container_name (str): Container name.
        patent (PatentInfo): PatentInfo object to update.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        container = _get_container(connection_string, database_name, container_name)
        delete_chunks_for_patent(container, patent.patent_id)
        patent.combined_vector = generate_patent_embedding(patent)
        partition_key = patent.partition_key
        all_chunks = []
        if should_chunk_field(patent.claims):
            claims_chunks = create_field_chunks(patent.patent_id, patent.id, "claims", patent.claims, partition_key)
            all_chunks.extend(claims_chunks)
            patent.claims = None
        if should_chunk_field(patent.description):
            desc_chunks = create_field_chunks(patent.patent_id, patent.id, "description", patent.description, partition_key)
            all_chunks.extend(desc_chunks)
            patent.description = None
        patent.update_timestamps()
        doc = _prepare_document_for_upload(patent)
        container.upsert_item(body=doc)
        logger.info(f"✅ Updated PatentInfo {patent.patent_id} with id {patent.id}")
        if all_chunks:
            success = save_chunks_to_db(container, all_chunks)
            if success:
                logger.info(f"✅ Inserted {len(all_chunks)} chunks for {patent.patent_id}")
            else:
                logger.warning(f"⚠️ Failed to insert some chunks for {patent.patent_id}")
        # Sync essential fields to Azure AI Search
        search_client = _get_search_client(config.AZURE_SEARCH_ENDPOINT, config.AZURE_SEARCH_ADMIN_KEY, config.AZURE_SEARCH_INDEX_NAME)
        search_doc = _prepare_document_for_search(patent)
        search_client.merge_or_upload_documents([search_doc])
        logger.info(f"✅ Synced essential fields for {patent.patent_id} to Azure AI Search")
        return True
    except Exception as e:
        logger.error(f"❌ Error updating patent {patent.patent_id}: {e}")
        return False


def delete_patent(connection_string: str, database_name: str, container_name: str, patent_id: str) -> bool:
    """
    Delete a patent and its chunks from Cosmos DB.

    Args:
        connection_string (str): Cosmos DB connection string.
        database_name (str): Database name.
        container_name (str): Container name.
        patent_id (str): Patent ID to delete.

    Returns:
        bool: True if deleted or not found, False if error.
    """
    try:
        container = _get_container(connection_string, database_name, container_name)
        # First, get partition_key for the patent
        query = "SELECT c.id, c.partition_key FROM c WHERE c.patent_id = @patent_id AND NOT IS_DEFINED(c.field)"
        params = [{"name": "@patent_id", "value": patent_id}]
        items = list(container.query_items(query=query, parameters=params, enable_cross_partition_query=True))
        if not items:
            logger.info(f"ℹ️ Patent {patent_id} not found")
            return True

        # Delete chunks
        delete_chunks_for_patent(container, patent_id)

        # Delete patent with partition_key
        for item in items:
            container.delete_item(item=item['id'], partition_key=item['partition_key'])
        logger.info(f"🗑️ Deleted PatentInfo {patent_id}")

        # Delete from Azure AI Search
        search_client = _get_search_client(config.AZURE_SEARCH_ENDPOINT, config.AZURE_SEARCH_ADMIN_KEY,
                                           config.AZURE_SEARCH_INDEX_NAME)
        search_client.delete_documents([{"id": patent_id}])
        logger.info(f"🗑️ Deleted PatentInfo {patent_id} from Azure AI Search")
        return True
    except Exception as e:
        logger.error(f"❌ Error deleting patent {patent_id}: {e}")
        return False

def get_all_patents(connection_string: str, database_name: str, container_name: str) -> List[PatentInfo]:
    """
    Get all patents with reconstructed claims and description from chunks.
    """
    try:
        container = _get_container(connection_string, database_name, container_name)
        query = "SELECT * FROM c WHERE NOT IS_DEFINED(c.field)"
        patents_data = list(container.query_items(query=query, enable_cross_partition_query=True))
        patent_list = []
        for p_data in patents_data:
            patent = PatentInfo.from_dict(p_data)
            partition_key = patent.partition_key
            claims_query = "SELECT * FROM c WHERE c.patent_id = @patent_id AND c.field = 'claims' AND c.partition_key = @partition_key"
            params = [{"name": "@patent_id", "value": patent.patent_id}, {"name": "@partition_key", "value": partition_key}]
            claims_chunks = list(
                container.query_items(query=claims_query, parameters=params, enable_cross_partition_query=False)
            )
            if claims_chunks:
                patent.claims = reconstruct_field_from_chunks(claims_chunks)
            desc_query = "SELECT * FROM c WHERE c.patent_id = @patent_id AND c.field = 'description' AND c.partition_key = @partition_key"
            desc_chunks = list(
                container.query_items(query=desc_query, parameters=params, enable_cross_partition_query=False)
            )
            if desc_chunks:
                patent.description = reconstruct_field_from_chunks(desc_chunks)
            patent_list.append(patent)
        logger.info(f"🔍 Retrieved {len(patent_list)} patents with reconstructed fields")
        return patent_list
    except Exception as e:
        logger.error(f"❌ Error retrieving all patents: {e}")
        return []

def _build_filter_conditions(filters: List[SearchInfo]) -> str:
    """
    Build OData filter string for Azure AI Search.
    """
    conditions = []
    for f in filters:
        if f.search_value:
            if f.search_by in ['patent_id', 'title', 'assignee', 'filing_date', 'cpc', 'ipc']:
                conditions.append(f"{f.search_by} eq '{f.search_value}'")
    return ' and '.join(conditions) if conditions else None


def _re_rank_results(query_text: str, results: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Re-rank search results using BAAI/bge-reranker-large with title and abstract.

    Args:
        query_text (str): Search query text.
        results (List[Dict[str, Any]]): List of search results with patent_id and score.
        top_k (int): Number of top results to return after re-ranking.

    Returns:
        List[Dict[str, Any]]: Re-ranked results with updated scores.
    """
    try:
        if not results or not query_text:
            logger.info("No results or query text provided for re-ranking")
            return results

        search_client = _get_search_client(config.AZURE_SEARCH_ENDPOINT, config.AZURE_SEARCH_ADMIN_KEY,
                                           config.AZURE_SEARCH_INDEX_NAME)
        patent_ids = [result['patent_id'] for result in results if result.get('patent_id')]

        if not patent_ids:
            logger.info("No valid patent IDs found for re-ranking")
            return results

        # Use OData filter with parameterization
        filter_str = " or ".join([f"patent_id eq '{pid}'" for pid in patent_ids])
        search_results = list(search_client.search(
            search_text=None,
            select=["id", "patent_id", "title", "abstract"],
            filter=filter_str
        ))

        # Create mapping of patent_id to text for re-ranking
        id_to_text = {r['patent_id']: f"{r['title'] or ''} {r['abstract'] or ''}".strip() for r in search_results}

        # Prepare pairs for re-ranking
        pairs = [[query_text, id_to_text.get(result['patent_id'], '')] for result in results if
                 id_to_text.get(result['patent_id'])]
        if not pairs:
            logger.warning("No valid text pairs for re-ranking")
            return results

        # Run re-ranking
        scores = reranker.predict(pairs)
        ranked_results = []
        for i, score in enumerate(scores):
            if i < len(results):  # Ensure index alignment
                result = results[i].copy()  # Avoid modifying original
                result['rerank_score'] = float(score)
                ranked_results.append(result)
            else:
                logger.warning(f"Mismatch in scores length: {len(scores)} vs results: {len(results)}")

        # Sort by rerank_score and return top_k
        ranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        logger.info(f"Successfully re-ranked {len(ranked_results)} results")
        return ranked_results[:top_k]
    except AzureError as ae:
        logger.error(f"Azure Search error in re-ranking: {ae}")
        return results
    except Exception as e:
        logger.error(f"Unexpected error in re-ranking: {e}")
        return results


def _retrieve_full_patent(container, patent_id: str) -> Optional[PatentInfo]:
    """
    Retrieve full patent data from Cosmos DB by patent_id.
    """
    try:
        # First, get partition_key for the patent
        query = "SELECT c.partition_key FROM c WHERE c.patent_id = @patent_id AND NOT IS_DEFINED(c.field)"
        params = [{"name": "@patent_id", "value": patent_id}]
        items = list(container.query_items(query=query, parameters=params, enable_cross_partition_query=True))
        if not items:
            logger.info(f"ℹ️ Patent {patent_id} not found")
            return None
        partition_key = items[0]['partition_key']

        # Retrieve patent with partition_key
        query = "SELECT * FROM c WHERE c.patent_id = @patent_id AND NOT IS_DEFINED(c.field) AND c.partition_key = @partition_key"
        params = [
            {"name": "@patent_id", "value": patent_id},
            {"name": "@partition_key", "value": partition_key}
        ]
        items = list(container.query_items(query=query, parameters=params, enable_cross_partition_query=False))
        if not items:
            return None
        patent = PatentInfo.from_dict(items[0])

        # Reconstruct claims and description
        claims_chunks = get_chunks_from_db(container, patent_id, field="claims", partition_key=partition_key)
        if claims_chunks:
            patent.claims = reconstruct_field_from_chunks(claims_chunks)
        desc_chunks = get_chunks_from_db(container, patent_id, field="description", partition_key=partition_key)
        if desc_chunks:
            patent.description = reconstruct_field_from_chunks(desc_chunks)
        return patent
    except Exception as e:
        logger.error(f"❌ Error retrieving full patent {patent_id}: {e}")
        return None


def _batch_retrieve_full_patents(container, patent_ids: List[str]) -> List[PatentInfo]:
    """
    Batch retrieve full patents from Cosmos DB by list of patent_ids (optimized with IN clause).
    """
    try:
        if not patent_ids:
            return []
        # Get partition keys for all patent_ids
        query = "SELECT c.patent_id, c.partition_key FROM c WHERE c.patent_id IN (@patent_ids) AND NOT IS_DEFINED(c.field)"
        params = [{"name": "@patent_ids", "value": patent_ids}]
        items = list(container.query_items(query=query, parameters=params, enable_cross_partition_query=True))
        id_to_partition = {item['patent_id']: item['partition_key'] for item in items}

        patents = []
        for patent_id in patent_ids:
            partition_key = id_to_partition.get(patent_id)
            if not partition_key:
                logger.info(f"ℹ️ Patent {patent_id} not found")
                continue
            # Retrieve patent with partition_key
            query = "SELECT * FROM c WHERE c.patent_id = @patent_id AND NOT IS_DEFINED(c.field) AND c.partition_key = @partition_key"
            params = [
                {"name": "@patent_id", "value": patent_id},
                {"name": "@partition_key", "value": partition_key}
            ]
            items = list(container.query_items(query=query, parameters=params, enable_cross_partition_query=False))
            if items:
                patent = PatentInfo.from_dict(items[0])
                claims_chunks = get_chunks_from_db(container, patent_id, field="claims", partition_key=partition_key)
                if claims_chunks:
                    patent.claims = reconstruct_field_from_chunks(claims_chunks)
                desc_chunks = get_chunks_from_db(container, patent_id, field="description", partition_key=partition_key)
                if desc_chunks:
                    patent.description = reconstruct_field_from_chunks(desc_chunks)
                patents.append(patent)
        return patents
    except Exception as e:
        logger.error(f"❌ Error batch retrieving patents: {e}")
        return []

def bm25_search(endpoint: str, key: str, index_name: str, search_text: str,
                filters: List[SearchInfo], limit: int = 10, skip: int = 0) -> List[Dict[str, Any]]:
    """
    Perform BM25 keyword-based search using Azure AI Search on title and abstract.

    Args:
        endpoint (str): Azure Search endpoint.
        key (str): Azure Search key.
        index_name (str): Azure Search index name.
        search_text (str): Search query text.
        filters (List[SearchInfo]): List of filter conditions.
        limit (int): Number of results to return.
        skip (int): Number of results to skip.

    Returns:
        List[Dict[str, Any]]: List of results with patent_id and BM25 score.
    """
    try:
        search_client = _get_search_client(endpoint, key, index_name)
        filter_str = _build_filter_conditions(filters)
        results = search_client.search(
            search_text=search_text,
            search_fields=["title", "abstract"],
            filter=filter_str,
            top=limit,
            skip=skip,
            query_type="simple",
            select=["id", "patent_id", "partition_key"]
        )
        final_results = []
        for result in results:
            final_results.append({
                'patent_id': result['patent_id'],
                'bm25_score': result.get('@search.score', 0.0)
            })
        return final_results
    except AzureError as ae:
        logger.error(f"❌ Azure error in BM25 search: {ae}")
        return []
    except Exception as e:
        logger.error(f"❌ Error in BM25 search: {e}")
        return []

def hybrid_search(endpoint: str, key: str, index_name: str, search_text: str, query_vector: List[float],
                  filters: List[SearchInfo], limit: int = 10, skip: int = 0,
                  text_weight: float = config.HYBRID_SEARCH_TEXT_WEIGHT,
                  vector_weight: float = config.HYBRID_SEARCH_VECTOR_WEIGHT) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining BM25 (on title, abstract) and vector search.

    Args:
        endpoint (str): Azure Search endpoint.
        key (str): Azure Search key.
        index_name (str): Azure Search index name.
        search_text (str): Search query text.
        query_vector (List[float]): Query embedding vector.
        filters (List[SearchInfo]): List of filter conditions.
        limit (int): Number of results to return.
        skip (int): Number of results to skip.
        text_weight (float): Weight for text search.
        vector_weight (float): Weight for vector search.

    Returns:
        List[Dict[str, Any]]: List of results with patent_id and hybrid score.
    """
    try:
        search_client = _get_search_client(endpoint, key, index_name)
        filter_str = _build_filter_conditions(filters)
        if not query_vector or len(query_vector) != VECTOR_DIMENSIONS:
            logger.warning(f"⚠️ Invalid query vector, falling back to BM25 search")
            return bm25_search(endpoint, key, index_name, search_text, filters, limit, skip)
        vector_query = VectorQuery(
            vector=query_vector,
            k_nearest_neighbors=limit,
            fields="combined_vector",
            kind="vector"
        )
        results = search_client.search(
            search_text=search_text,
            search_fields=["title", "abstract"],
            vector_queries=[vector_query],
            filter=filter_str,
            top=limit,
            skip=skip,
            query_type="simple",
            select=["id", "patent_id", "partition_key"]
        )
        final_results = []
        for result in results:
            final_results.append({
                'patent_id': result['patent_id'],
                'hybrid_score': result.get('@search.score', 0.0)
            })
        return final_results
    except AzureError as ae:
        logger.error(f"❌ Azure error in hybrid search: {ae}")
        return []
    except Exception as e:
        logger.error(f"❌ Error in hybrid search: {e}")
        return []

def semantic_search(endpoint: str, key: str, index_name: str, search_text: str,
                    filters: List[SearchInfo], limit: int = 10, skip: int = 0) -> List[Dict[str, Any]]:
    """
    Perform semantic search using Azure AI Search on title and abstract.
    Fallback to BM25 if semantic configuration is not available.
    """
    try:
        search_client = _get_search_client(endpoint, key, index_name)
        filter_str = _build_filter_conditions(filters)
        try:
            results = search_client.search(
                search_text=search_text,
                search_fields=["title", "abstract"],
                filter=filter_str,
                top=limit,
                skip=skip,
                query_type="semantic",
                semantic_configuration_name="default",
                select=["id", "patent_id", "partition_key"]
            )
            final_results = []
            for result in results:
                final_results.append({
                    'patent_id': result['patent_id'],
                    'semantic_score': result.get('@search.reranker_score', result.get('@search.score', 0.0))
                })
            return final_results
        except AzureError as semantic_error:
            logger.warning(f"⚠️ Semantic search failed: {semantic_error}. Falling back to BM25.")
            bm25_results = bm25_search(endpoint, key, index_name, search_text, filters, limit, skip)
            return [
                {
                    'patent_id': result['patent_id'],
                    'semantic_score': result['bm25_score']
                } for result in bm25_results
            ]
    except AzureError as ae:
        logger.error(f"❌ Azure error in semantic search: {ae}")
        bm25_results = bm25_search(endpoint, key, index_name, search_text, filters, limit, skip)
        return [
            {
                'patent_id': result['patent_id'],
                'semantic_score': result['bm25_score']
            } for result in bm25_results
        ]
    except Exception as e:
        logger.error(f"❌ Error in semantic search: {e}")
        return []

def vector_search(endpoint: str, key: str, index_name: str, search_text: str,
                  filters: List[SearchInfo], limit: int = 10, skip: int = 0) -> List[Dict[str, Any]]:
    """
    Perform vector search using Azure AI Search.

    Args:
        endpoint (str): Azure Search endpoint.
        key (str): Azure Search key.
        index_name (str): Azure Search index name.
        search_text (str): Search query text to convert to vector.
        filters (List[SearchInfo]): List of filter conditions.
        limit (int): Number of results to return.
        skip (int): Number of results to skip.

    Returns:
        List[Dict[str, Any]]: List of results with patent_id and vector score.
    """
    try:
        search_client = _get_search_client(endpoint, key, index_name)
        filter_str = _build_filter_conditions(filters)
        query_vector = generate_embeddings([search_text])[0]
        if not query_vector or len(query_vector) != VECTOR_DIMENSIONS:
            logger.warning(f"⚠️ Invalid query vector for query '{search_text}', returning empty results")
            return []
        vector_query = VectorQuery(
            vector=query_vector,
            k_nearest_neighbors=limit,
            fields="combined_vector",
            kind="vector"
        )
        results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=filter_str,
            top=limit,
            skip=skip,
            select=["id", "patent_id", "partition_key"]
        )
        final_results = []
        result_list = list(results)
        if not result_list:
            logger.warning("⚠️ No results in vector search - check if index has data and vector is valid")
            return final_results
        for result in result_list:
            final_results.append({
                'patent_id': result['patent_id'],
                'vector_score': result.get('@search.score', 0.0)
            })
        return final_results
    except AzureError as ae:
        logger.error(f"❌ Azure error in vector search: {ae}")
        return []
    except Exception as e:
        logger.error(f"❌ Error in vector search: {e}")
        return []

def search_with_page(connection_string: str, database_name: str, container_name: str,
                     endpoint: str, key: str, index_name: str,
                     filter_input: FilterInfoWithPageInput) -> Dict[str, Any]:
    """
    Perform search with pagination, supporting hybrid, semantic, vector, or BM25 search, with re-ranking.
    """
    try:
        container = _get_container(connection_string, database_name, container_name)
        search_type = filter_input.search_type.lower()
        query_text = filter_input.extract_query()
        limit = filter_input.page_size
        skip = (filter_input.page_number - 1) * filter_input.page_size
        filters = filter_input.search_infos
        search_params = filter_input.get_search_parameters()

        # Generate query vector for hybrid or vector search
        query_vector = []
        if search_type in ["hybrid", "vector"]:
            if query_text.strip():
                query_vector = generate_embeddings([query_text])[0]
                if not query_vector or len(query_vector) != VECTOR_DIMENSIONS:
                    logger.warning(f"⚠️ Invalid query vector for query '{query_text}', falling back to BM25")
                    search_type = "bm25"
            else:
                logger.warning(f"⚠️ No query text for {search_type} search, falling back to BM25")
                search_type = "bm25"

        # Perform search based on search_type
        if search_type == "hybrid" and query_vector:
            results = hybrid_search(
                endpoint=endpoint,
                key=key,
                index_name=index_name,
                search_text=query_text,
                query_vector=query_vector,
                filters=filters,
                limit=limit * 2,
                skip=skip,
                text_weight=search_params.get("text_weight", config.HYBRID_SEARCH_TEXT_WEIGHT),
                vector_weight=search_params.get("vector_weight", config.HYBRID_SEARCH_VECTOR_WEIGHT)
            )
        elif search_type == "vector" and query_vector:
            results = vector_search(
                endpoint=endpoint,
                key=key,
                index_name=index_name,
                search_text=query_text,
                filters=filters,
                limit=limit * 2,
                skip=skip
            )
        elif search_type == "semantic":
            results = semantic_search(
                endpoint=endpoint,
                key=key,
                index_name=index_name,
                search_text=query_text,
                filters=filters,
                limit=limit * 2,
                skip=skip
            )
        else:  # Default to BM25
            results = bm25_search(
                endpoint=endpoint,
                key=key,
                index_name=index_name,
                search_text=query_text,
                filters=filters,
                limit=limit * 2,
                skip=skip
            )

        # Re-rank results
        results = _re_rank_results(query_text, results, top_k=limit)

        # Retrieve full patent data from Cosmos DB using batch
        patent_ids = [result['patent_id'] for result in results]
        patents = _batch_retrieve_full_patents(container, patent_ids)
        id_to_patent = {patent.patent_id: patent for patent in patents}

        final_results = []
        for result in results:
            patent = id_to_patent.get(result['patent_id'])
            if patent:
                final_results.append({
                    'patent': patent,
                    'score': result.get('rerank_score', result.get('score', 0.0))
                })

        # Get total count (approximate, using count query)
        search_client = _get_search_client(endpoint, key, index_name)
        filter_str = _build_filter_conditions(filters)
        count_results = search_client.search(
            search_text=query_text if search_type != "semantic" else None,
            search_fields=["title", "abstract"],
            filter=filter_str,
            query_type="simple" if search_type != "semantic" else "semantic",
            semantic_configuration_name="default" if search_type == "semantic" else None,
            include_total_count=True,
            select=["id"]
        )
        total_count = count_results.get_count() or 0

        return {
            "results": final_results,
            "total_count": total_count,
            "page_number": filter_input.page_number,
            "page_size": filter_input.page_size
        }
    except Exception as e:
        logger.error(f"❌ Error in search_with_page: {e}")
        return {
            "results": [],
            "total_count": 0,
            "page_number": filter_input.page_number,
            "page_size": filter_input.page_size
        }

def create_database_and_container(connection_string: str, database_name: str, container_name: str,
                                  force_recreate: bool = False) -> bool:
    """
    Create Cosmos DB database and container (no vector search).
    """
    try:
        client = CosmosClient.from_connection_string(connection_string)
        if force_recreate:
            try:
                client.delete_database(database_name)
                logger.info(f"🗑️ Deleted database '{database_name}'")
            except exceptions.CosmosResourceNotFoundError:
                logger.info(f"ℹ️ Database '{database_name}' not found")
        try:
            database = client.create_database(database_name)
            logger.info(f"✅ Created database '{database_name}'")
        except exceptions.CosmosResourceExistsError:
            database = client.get_database_client(database_name)
            logger.info(f"ℹ️ Database '{database_name}' already exists")
        indexing_policy = {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [
                {"path": "/_etag/?"},
                {"path": "/_ts/?"}
            ]
        }
        try:
            container = database.create_container(
                id=container_name,
                partition_key=PartitionKey(path="/partition_key"),
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

def delete_container(connection_string: str, database_name: str, container_name: str) -> bool:
    """
    Delete Cosmos DB container and all its documents.

    Args:
        connection_string (str): Cosmos DB connection string.
        database_name (str): Database name.
        container_name (str): Container name to delete.

    Returns:
        bool: True if deleted or not found, False if error.
    """
    try:
        client = CosmosClient.from_connection_string(connection_string)
        database = client.get_database_client(database_name)
        try:
            database.delete_container(container_name)
            logger.info(f"🗑️ Deleted container '{container_name}'")
            return True
        except exceptions.CosmosResourceNotFoundError:
            logger.info(f"ℹ️ Container '{container_name}' not found")
            return True
    except Exception as e:
        logger.error(f"❌ Error deleting container '{container_name}': {e}")
        return False