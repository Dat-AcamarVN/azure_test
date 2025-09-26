"""
Test Fixed Chunking Logic
"""

from utilities.chunking_utils import chunk_text_simple
import config

def test_fixed_chunking():
    """Test the fixed chunking logic"""
    print("🧪 Testing Fixed Chunking Logic")
    print("=" * 40)
    
    # Test with the same text from the original test
    test_text = "This is a very long abstract. " * 150  # ~4,500 chars
    
    print(f"Input text length: {len(test_text)} chars")
    print(f"MAX_CHUNK_SIZE: {config.MAX_CHUNK_SIZE}")
    print(f"CHUNK_OVERLAP: {config.CHUNK_OVERLAP}")
    
    # Calculate expected chunks
    expected_chunks = (len(test_text) - config.CHUNK_OVERLAP) // (config.MAX_CHUNK_SIZE - config.CHUNK_OVERLAP) + 1
    print(f"Expected chunks (theoretical): {expected_chunks}")
    
    # Test chunking
    chunks = chunk_text_simple(test_text, "abstract", "test_patent_001")
    
    print(f"Actual chunks created: {len(chunks)}")
    print(f"Chunk sizes: {[chunk['chunk_size'] for chunk in chunks[:5]]}...")
    
    # Verify overlap
    if len(chunks) > 1:
        first_chunk_end = chunks[0]['chunk_text'][-50:]
        second_chunk_start = chunks[1]['chunk_text'][:50]
        print(f"First chunk end: ...{first_chunk_end}")
        print(f"Second chunk start: {second_chunk_start}")
        
        # Check if overlap is working
        overlap_text = chunks[0]['chunk_text'][-config.CHUNK_OVERLAP:]
        if overlap_text in chunks[1]['chunk_text']:
            print("✅ Overlap working correctly")
        else:
            print("❌ Overlap not working")
    
    # Test with different text sizes
    print("\n" + "=" * 40)
    print("Testing different text sizes:")
    
    test_cases = [
        ("Short text", "Short text that should not be chunked"),
        ("Medium text", "Medium length text. " * 50),  # ~1KB
        ("Long text", "Long text for chunking. " * 100),  # ~2.5KB
        ("Very long text", "Very long text for chunking. " * 200),  # ~5KB
    ]
    
    for name, text in test_cases:
        chunks = chunk_text_simple(text, "test", "test_patent")
        print(f"  {name} ({len(text)} chars): {len(chunks)} chunks")

if __name__ == "__main__":
    test_fixed_chunking() 