"""
Simple Search Test - Verify search index is working
"""

from config import (
    AZURE_SEARCH_ENDPOINT, 
    AZURE_SEARCH_ADMIN_KEY, 
    AZURE_SEARCH_INDEX_NAME
)
from dao.patent_dao import _get_search_client

def test_search_index_basic():
    """Test basic search index functionality"""
    print("🧪 Testing Basic Search Index")
    print("=" * 40)
    
    try:
        # Get search client
        search_client = _get_search_client(
            AZURE_SEARCH_ENDPOINT, 
            AZURE_SEARCH_ADMIN_KEY, 
            AZURE_SEARCH_INDEX_NAME
        )
        
        print(f"✅ Search client created successfully")
        print(f"   Endpoint: {AZURE_SEARCH_ENDPOINT}")
        print(f"   Index: {AZURE_SEARCH_INDEX_NAME}")
        
        # Test 1: Wildcard search
        print(f"\n1. Testing wildcard search...")
        wildcard_results = search_client.search(
            search_text="*",
            top=5,
            select=["patent_id", "title", "patent_office"]
        )
        
        wildcard_list = list(wildcard_results)
        print(f"   ✅ Wildcard search completed")
        print(f"   Total results: {len(wildcard_list)}")
        
        if wildcard_list:
            print(f"   📊 Index contains data:")
            for i, result in enumerate(wildcard_list):
                print(f"     Result {i+1}:")
                print(f"       Patent ID: {result.get('patent_id', 'N/A')}")
                print(f"       Title: {result.get('title', 'N/A')}")
                print(f"       Office: {result.get('patent_office', 'N/A')}")
        else:
            print(f"   ⚠️ Index appears to be empty")
        
        # Test 2: Simple text search
        print(f"\n2. Testing simple text search...")
        simple_results = search_client.search(
            search_text="the",
            top=3,
            select=["patent_id", "title"]
        )
        
        simple_list = list(simple_results)
        print(f"   ✅ Simple search completed")
        print(f"   Total results: {len(simple_list)}")
        
        if simple_list:
            for i, result in enumerate(simple_list):
                print(f"     Match {i+1}: {result.get('patent_id', 'N/A')} - {result.get('title', 'N/A')}")
        
        # Test 3: Field-specific search
        print(f"\n3. Testing field-specific search...")
        field_results = search_client.search(
            search_text="data",
            top=3,
            select=["patent_id", "title", "abstract"]
        )
        
        field_list = list(field_results)
        print(f"   ✅ Field search completed")
        print(f"   Total results: {len(field_list)}")
        
        if field_list:
            for i, result in enumerate(field_list):
                print(f"     Match {i+1}: {result.get('patent_id', 'N/A')} - {result.get('title', 'N/A')}")
                if result.get('abstract'):
                    abstract_preview = result['abstract'][:100] + "..." if len(result['abstract']) > 100 else result['abstract']
                    print(f"       Abstract: {abstract_preview}")
        
        print(f"\n🎉 Search index test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing search index: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_search_index_basic() 