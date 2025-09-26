"""
Simple & Practical Chunking Utilities
Easy to extend and maintain
"""

import logging
import uuid
from typing import List, Dict, Any
from datetime import datetime
import tiktoken

import config  # Assuming config has necessary values

logger = logging.getLogger(__name__)

# Constants from insert_large_data
CHUNK_SIZE = 200
MIN_CHUNK_SIZE_CHARS = 350
MIN_CHUNK_LENGTH_TO_EMBED = 5
MAX_NUM_CHUNKS = 10000
MAX_TOKENS_THRESHOLD = 500

def get_text_chunks(text: str, chunk_token_size: int = CHUNK_SIZE) -> List[str]:
    if not text or text.isspace():
        return []
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text, disallowed_special=())
    chunks = []
    num_chunks = 0
    while tokens and num_chunks < MAX_NUM_CHUNKS:
        chunk = tokens[:chunk_token_size]
        chunk_text = tokenizer.decode(chunk)
        if not chunk_text or chunk_text.isspace():
            tokens = tokens[len(chunk):]
            continue
        last_punctuation = max(chunk_text.rfind("."), chunk_text.rfind("?"), chunk_text.rfind("!"), chunk_text.rfind("\n"))
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

def should_chunk_field(text: str) -> bool:
    if not text:
        return False
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text)) > MAX_TOKENS_THRESHOLD

def create_field_chunks(patent_id: str, field: str, text: str, patent_office: str) -> List[Dict[str, Any]]:
    chunks = get_text_chunks(text)
    chunk_docs = []
    current_time = datetime.now().isoformat()
    for i, chunk_text in enumerate(chunks):
        chunk_id = f"{patent_id}_{field}_{uuid.uuid4().hex[:8]}_{i}"
        chunk_docs.append({
            "id": chunk_id,
            "patent_id": patent_id,
            "text": chunk_text,
            "type": field,
            "patent_office": patent_office or "unknown",
            "created_at": current_time,
            "updated_at": current_time
        })
    return chunk_docs

def save_chunks_to_db(container, chunks: List[Dict[str, Any]]) -> bool:
    try:
        for chunk in chunks:
            container.create_item(body=chunk)
        logger.info(f"✅ Saved {len(chunks)} chunks")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving chunks: {e}")
        return False

def get_chunks_from_db(container, patent_id: str, field: str = None) -> List[Dict[str, Any]]:
    try:
        query = "SELECT * FROM c WHERE c.patent_id = @patent_id"
        if field:
            query += " AND c.type = @field"
        parameters = [{"name": "@patent_id", "value": patent_id}]
        if field:
            parameters.append({"name": "@field", "value": field})
        chunks = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        # Sort by chunk index from id
        chunks.sort(key=lambda x: int(x['id'].split('_')[-1]))
        return chunks
    except Exception as e:
        logger.error(f"❌ Error getting chunks: {e}")
        return []

def reconstruct_field_from_chunks(chunks: List[Dict[str, Any]]) -> str:
    return ' '.join([chunk['text'] for chunk in chunks])

def delete_chunks_for_patent(container, patent_id: str):
    try:
        chunks = get_chunks_from_db(container, patent_id)
        for chunk in chunks:
            container.delete_item(item=chunk['id'], partition_key=chunk['patent_office'])
        logger.info(f"🗑️ Deleted {len(chunks)} chunks for patent {patent_id}")
    except Exception as e:
        logger.error(f"❌ Error deleting chunks: {e}")

# Add search_in_chunks
def search_in_chunks(container, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search text in chunks using CONTAINS"""
    try:
        query = """
        SELECT c.patent_id, c.text, c.type
        FROM c 
        WHERE CONTAINS(c.text, @search_text, true)
        OFFSET 0 LIMIT @limit
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
        from collections import defaultdict
        patent_chunks = defaultdict(list)
        for chunk in chunks:
            patent_chunks[chunk["patent_id"]].append(chunk)
        
        results = []
        for patent_id, chunk_list in patent_chunks.items():
            results.append({
                "patent_id": patent_id,
                "matching_chunks": chunk_list,
                "total_matches": len(chunk_list)
            })
        
        return results[:limit]
    except Exception as e:
        logger.error(f"❌ Error searching in chunks: {e}")
        return []
