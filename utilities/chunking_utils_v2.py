import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from azure.cosmos import CosmosClient, exceptions
import tiktoken
import config
import time
import requests

from models.patent_model import PatentChunk

# Get logger
try:
    from logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Constants from config
CHUNK_SIZE = getattr(config, 'CHUNK_SIZE', 512)
MIN_CHUNK_SIZE_CHARS = getattr(config, 'MIN_CHUNK_SIZE_CHARS', 350)
MIN_CHUNK_LENGTH_TO_EMBED = getattr(config, 'MIN_CHUNK_LENGTH_TO_EMBED', 5)
MAX_NUM_CHUNKS = getattr(config, 'MAX_NUM_CHUNKS', 10000)
MAX_TOKENS_THRESHOLD = getattr(config, 'MAX_TOKENS_THRESHOLD', 1024)

def get_text_chunks(text: str, chunk_token_size: int = CHUNK_SIZE) -> List[str]:
    """
    Split text into chunks based on token size, respecting punctuation boundaries.

    Args:
        text (str): Input text to chunk.
        chunk_token_size (int): Target token size per chunk (default: config.CHUNK_SIZE).

    Returns:
        List[str]: List of chunked texts.
    """
    if not text or text.isspace():
        return []
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text, disallowed_special=())
        chunks = []
        num_chunks = 0
        while tokens and num_chunks < MAX_NUM_CHUNKS:
            chunk = tokens[:chunk_token_size]
            chunk_text = tokenizer.decode(chunk)
            last_punctuation = max(chunk_text.rfind("."), chunk_text.rfind("?"),
                                   chunk_text.rfind("!"), chunk_text.rfind("\n"))
            if last_punctuation != -1 and last_punctuation > MIN_CHUNK_SIZE_CHARS:
                chunk_text = chunk_text[:last_punctuation + 1]
            chunk_text_to_append = chunk_text.replace("\n", " ").strip()
            if len(chunk_text_to_append) > MIN_CHUNK_LENGTH_TO_EMBED:
                chunks.append(chunk_text_to_append)
            tokens = tokens[len(tokenizer.encode(chunk_text, disallowed_special=())):]
            num_chunks += 1
        if tokens:
            remaining_text = tokenizer.decode(tokens).replace("\n", " ").strip()
            if len(remaining_text) > MIN_CHUNK_LENGTH_TO_EMBED:
                chunks.append(remaining_text)
        return chunks
    except Exception as e:
        logger.error(f"❌ Error chunking text: {e}")
        return []

def should_chunk_field(text: str) -> bool:
    """
    Determine if a text field needs chunking based on token count.

    Args:
        text (str): Input text to check.

    Returns:
        bool: True if text exceeds MAX_TOKENS_THRESHOLD, False otherwise.
    """
    if not text:
        return False
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        return len(tokenizer.encode(text)) > MAX_TOKENS_THRESHOLD
    except Exception as e:
        logger.error(f"❌ Error checking chunk necessity: {e}")
        return False

def create_field_chunks(patent_id: str, id: str, field: str, text: str, partition_key: str) -> List[PatentChunk]:
    """
    Create chunk documents for a field with metadata, using id (UUID) for chunk_id.

    Args:
        patent_id (str): Patent business ID.
        id (str): Patent UUID id.
        field (str): Field name.
        text (str): Text to chunk.
        partition_key (str): Partition key.

    Returns:
        List[PatentChunk]: List of PatentChunk objects.
    """
    if not patent_id or not id or not field:
        raise ValueError("patent_id, id, and field are required")
    try:
        chunks = get_text_chunks(text)
        chunk_objs = []
        current_time = datetime.now().isoformat()
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{id}_{field}_{i:04d}"  # Use UUID id for chunk_id
            chunk = PatentChunk(
                id=chunk_id,
                patent_id=patent_id,
                text=chunk_text,
                field=field,
                partition_key=partition_key,
                created_at=current_time,
                updated_at=current_time,
                chunk_index=i,
                chunk_size=len(chunk_text)
            )
            chunk_objs.append(chunk)
        return chunk_objs
    except Exception as e:
        logger.error(f"❌ Error creating field chunks: {e}")
        return []

def save_chunks_to_db(container, chunks: List[PatentChunk]) -> bool:
    """
    Save chunks to Cosmos DB in parallel with retry logic.
    """
    def save_chunk(chunk, max_retries=3, initial_delay=1):
        for attempt in range(max_retries):
            try:
                container.upsert_item(body=chunk.to_dict())
                return True
            except (exceptions.CosmosHttpResponseError, requests.exceptions.ConnectionError) as e:
                if isinstance(e, exceptions.CosmosHttpResponseError) and e.status_code == 429:
                    delay = initial_delay * (2 ** attempt)
                    logger.warning(f"⚠️ Rate limit (429) for chunk {chunk.id}, retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Error saving chunk {chunk.id}: {e}")
                    if attempt < max_retries - 1:
                        delay = initial_delay * (2 ** attempt)
                        logger.warning(f"⚠️ Retrying chunk {chunk.id} in {delay}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        logger.error(f"❌ Failed to save chunk {chunk.id} after {max_retries} attempts")
                        return False
        return False

    try:
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(save_chunk, chunk) for chunk in chunks]
            results = [future.result() for future in futures]
        success = all(results)
        if success:
            logger.info(f"✅ Successfully saved {len(chunks)} chunks")
        else:
            logger.warning(f"⚠️ Failed to save some chunks")
        return success
    except Exception as e:
        logger.error(f"❌ Error saving chunks: {e}")
        return False

def get_chunks_from_db(container, patent_id: str, field: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve chunks for a patent from Cosmos DB.

    Args:
        container: Cosmos DB container client.
        patent_id (str): Patent business ID to retrieve chunks for.
        field (str, optional): Field to filter chunks.
        user_id (str, optional): User ID for authorization check.

    Returns:
        List[Dict[str, Any]]: List of chunk documents, sorted by chunk_index.
    """
    try:
        query = "SELECT * FROM c WHERE c.patent_id = @patent_id AND IS_DEFINED(c.field) AND c.field != ''"
        parameters = [{"name": "@patent_id", "value": patent_id}]
        if field:
            query += " AND c.field = @field"
            parameters.append({"name": "@field", "value": field})
        if user_id and hasattr(config, 'AUTHORIZED_USERS'):
            query += " AND EXISTS (SELECT VALUE p FROM p IN c WHERE p.user_id = @user_id)"
            parameters.append({"name": "@user_id", "value": user_id})
        chunks = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
        chunks.sort(key=lambda x: x.get('chunk_index', 0))
        return chunks
    except exceptions.CosmosHttpResponseError as e:
        logger.error(f"❌ Cosmos error getting chunks: {e}")
        return []
    except Exception as e:
        logger.error(f"❌ Error getting chunks: {e}")
        return []

def reconstruct_field_from_chunks(chunks: List[Dict[str, Any]]) -> str:
    """
    Reconstruct text from chunks in order.
    """
    try:
        sorted_chunks = sorted(chunks, key=lambda x: x.get('chunk_index', 0))
        return ' '.join(chunk.get('text', '') for chunk in sorted_chunks)
    except Exception as e:
        logger.error(f"❌ Error reconstructing field: {e}")
        return ""

def delete_chunks_for_patent(container, patent_id: str, user_id: Optional[str] = None):
    """
    Delete all chunks for a patent from Cosmos DB.

    Args:
        container: Cosmos DB container client.
        patent_id (str): Patent business ID to delete chunks for.
        user_id (str, optional): User ID for authorization check.
    """
    if not patent_id:
        raise ValueError("patent_id is required")
    try:
        chunks = get_chunks_from_db(container, patent_id, user_id=user_id)
        for chunk in chunks:
            container.delete_item(item=chunk['id'], partition_key=chunk['partition_key'])
        logger.info(f"🗑️ Deleted {len(chunks)} chunks for patent {patent_id}")
    except exceptions.CosmosHttpResponseError as e:
        logger.error(f"❌ Cosmos error deleting chunks: {e}")
    except Exception as e:
        logger.error(f"❌ Error deleting chunks: {e}")

def search_in_chunks(container, search_text: str, limit: int = 10, skip: int = 0,
                     similarity_threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD,
                     user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search text in chunks using CONTAINS with pagination and authorization.

    Args:
        container: Cosmos DB container client.
        search_text (str): Text to search for.
        limit (int): Number of results to return (default: 10).
        skip (int): Number of results to skip (default: 0).
        similarity_threshold (float): Minimum score for results (default: config.DEFAULT_SIMILARITY_THRESHOLD).
        user_id (str, optional): User ID for authorization check.

    Returns:
        List[Dict[str, Any]]: List of results with patent_id, matching_chunks, total_matches, and score.
    """
    try:
        query = """
        SELECT c.patent_id, c.text, c.field, c.chunk_index
        FROM c
        WHERE CONTAINS(c.text, @search_text, true) AND IS_DEFINED(c.field) AND c.field != ''
        """
        if user_id and hasattr(config, 'AUTHORIZED_USERS'):
            query += " AND EXISTS (SELECT VALUE p FROM p IN c WHERE p.user_id = @user_id)"
        query += " OFFSET @skip LIMIT @limit"
        parameters = [
            {"name": "@search_text", "value": search_text},
            {"name": "@skip", "value": skip},
            {"name": "@limit", "value": limit}
        ]
        if user_id and hasattr(config, 'AUTHORIZED_USERS'):
            parameters.append({"name": "@user_id", "value": user_id})
        chunks = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
        from collections import defaultdict
        patent_chunks = defaultdict(list)
        for chunk in chunks:
            patent_chunks[chunk["patent_id"]].append(chunk)
        results = []
        for patent_id, chunk_list in patent_chunks.items():
            results.append({
                "patent_id": patent_id,
                "matching_chunks": chunk_list,
                "total_matches": len(chunk_list),
                "score": 0.8
            })
        results.sort(key=lambda x: x["total_matches"], reverse=True)
        return [r for r in results[:limit] if r["score"] >= similarity_threshold]
    except exceptions.CosmosHttpResponseError as e:
        logger.error(f"❌ Cosmos error searching in chunks: {e}")
        return []
    except Exception as e:
        logger.error(f"❌ Error searching in chunks: {e}")
        return []
