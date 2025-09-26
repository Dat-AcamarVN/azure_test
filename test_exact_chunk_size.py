"""
Test Exact Chunk Size - Verify each chunk has exactly 2000 characters
"""

from utilities.chunking_utils import chunk_text_simple
import config

def test_exact_chunk_size():
    """Test that chunks have exactly the specified size"""
    print("🧪 Testing Exact Chunk Size (2000 chars)")
    print("=" * 50)
    
    # Test with text that should create multiple chunks
    test_text = "This is a test sentence. " * 100  # ~2,600 chars
    
    print(f"Input text length: {len(test_text)} chars")
    print(f"MAX_CHUNK_SIZE: {config.MAX_CHUNK_SIZE}")
    print(f"CHUNK_OVERLAP: {config.CHUNK_OVERLAP}")
    
    # Test chunking
    chunks = chunk_text_simple(test_text, "test", "test_patent_001")
    
    print(f"\nCreated {len(chunks)} chunks")
    
    # Verify each chunk size
    all_correct = True
    for i, chunk in enumerate(chunks):
        chunk_size = chunk['chunk_size']
        expected_size = config.MAX_CHUNK_SIZE if i < len(chunks) - 1 else len(test_text) - sum(c['chunk_size'] for c in chunks[:-1])
        
        if i == len(chunks) - 1:
            # Last chunk can be smaller
            status = "✅" if chunk_size <= config.MAX_CHUNK_SIZE else "❌"
            print(f"  Chunk {i+1}: {chunk_size} chars (last chunk) {status}")
        else:
            # All other chunks should be exactly MAX_CHUNK_SIZE
            status = "✅" if chunk_size == config.MAX_CHUNK_SIZE else "❌"
            print(f"  Chunk {i+1}: {chunk_size} chars {status}")
            
            if chunk_size != config.MAX_CHUNK_SIZE:
                all_correct = False
        
        # Show chunk preview
        preview = chunk['chunk_text'][:50] + "..." if len(chunk['chunk_text']) > 50 else chunk['chunk_text']
        print(f"    Preview: {preview}")
    
    print(f"\nOverall result: {'✅ All chunks have correct size' if all_correct else '❌ Some chunks have wrong size'}")
    
    # Test with larger text
    print("\n" + "=" * 50)
    print("Testing with larger text:")
    
    large_text = "This is a very long text for testing chunking. " * 200  # ~10,000 chars
    large_chunks = chunk_text_simple(large_text, "large_test", "test_patent_002")
    
    print(f"Large text ({len(large_text)} chars) created {len(large_chunks)} chunks")
    
    # Calculate expected chunks
    expected_chunks = (len(large_text) - config.CHUNK_OVERLAP) // (config.MAX_CHUNK_SIZE - config.CHUNK_OVERLAP) + 1
    print(f"Expected chunks: {expected_chunks}")
    
    # Show chunk size distribution
    chunk_sizes = [chunk['chunk_size'] for chunk in large_chunks]
    print(f"Chunk sizes: {chunk_sizes[:10]}{'...' if len(chunk_sizes) > 10 else ''}")
    
    # Check if most chunks are exactly 2000
    exact_size_count = sum(1 for size in chunk_sizes if size == config.MAX_CHUNK_SIZE)
    print(f"Chunks with exact {config.MAX_CHUNK_SIZE} chars: {exact_size_count}/{len(chunk_sizes)}")

if __name__ == "__main__":
    test_exact_chunk_size() 