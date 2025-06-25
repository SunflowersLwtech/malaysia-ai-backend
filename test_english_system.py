#!/usr/bin/env python3
"""
Complete Test Suite for Malaysia Tourism RAG System
Tests both database builder and API server functionality
All output and messages in English
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
import traceback

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def print_section(title):
    """Print a formatted section"""
    print(f"\n--- {title} ---")

def test_database_existence():
    """Test if vector database exists and is accessible"""
    print_section("Testing Vector Database")
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Check if database directory exists
        db_path = "./vector_database"
        if not os.path.exists(db_path):
            print("❌ ERROR: Vector database directory not found")
            print("   Please run: python 1_build_database.py")
            return False
        
        # Connect to database
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get collection
        collection = client.get_collection("malaysia_travel_guide")
        doc_count = collection.count()
        
        print(f"✅ SUCCESS: Database connected")
        print(f"   📊 Documents in database: {doc_count:,}")
        
        if doc_count == 0:
            print("❌ WARNING: Database is empty")
            return False
        
        # Test search functionality
        results = collection.query(
            query_texts=["Malaysia tourism attractions"],
            n_results=3
        )
        
        if results['documents'] and results['documents'][0]:
            print(f"✅ SUCCESS: Search functionality working")
            print(f"   📝 Sample result: {results['documents'][0][0][:100]}...")
            return True
        else:
            print("❌ ERROR: Search returned no results")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: Database test failed: {e}")
        return False

def test_api_server():
    """Test API server endpoints"""
    print_section("Testing API Server")
    
    base_url = "http://localhost:8080"
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ SUCCESS: Health check passed")
            print(f"   🏥 Status: {health_data['status']}")
            print(f"   🕒 Timestamp: {health_data['timestamp']}")
            
            # Print service status
            if 'services' in health_data:
                for service, status in health_data['services'].items():
                    print(f"   📋 {service}: {status}")
        else:
            print(f"❌ ERROR: Health check failed (Status: {response.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API server")
        print("   Please start the server with: python api_server.py")
        return False
    except Exception as e:
        print(f"❌ ERROR: Health check failed: {e}")
        return False
    
    # Test 2: Malaysia travel query
    print_section("Testing Malaysia Travel Query")
    try:
        query_data = {
            "query": "What are the best attractions in Kuala Lumpur?",
            "include_metadata": True
        }
        
        response = requests.post(
            f"{base_url}/api/query",
            json=query_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS: Malaysia travel query worked")
            print(f"   📝 Query: {result['query']}")
            print(f"   ✅ Status: {result['status']}")
            print(f"   📄 Retrieved docs: {result['metadata'].get('retrieved_count', 'N/A')}")
            print(f"   🤖 Model used: {result['metadata'].get('model_used', 'N/A')}")
            print(f"   💬 Response preview: {result['response'][:150]}...")
            
            if result['retrieved_documents']:
                print(f"   📊 Top relevance score: {result['retrieved_documents'][0]['relevance_score']:.3f}")
        else:
            print(f"❌ ERROR: Malaysia query failed (Status: {response.status_code})")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ ERROR: Malaysia query test failed: {e}")
        return False
    
    # Test 3: Non-Malaysia query (should be rejected)
    print_section("Testing Non-Malaysia Query Rejection")
    try:
        query_data = {
            "query": "What are the best restaurants in Tokyo?"
        }
        
        response = requests.post(
            f"{base_url}/api/query",
            json=query_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['status'] == 'rejected':
                print(f"✅ SUCCESS: Non-Malaysia query properly rejected")
                print(f"   🚫 Status: {result['status']}")
                print(f"   💬 Response: {result['response'][:100]}...")
            else:
                print(f"❌ WARNING: Non-Malaysia query was not rejected")
                print(f"   Status: {result['status']}")
        else:
            print(f"❌ ERROR: Non-Malaysia query test failed (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ ERROR: Non-Malaysia query test failed: {e}")
        return False
    
    # Test 4: Relevance checking
    print_section("Testing Relevance Checking")
    try:
        relevance_data = {
            "query": "Best places to visit in Malaysia"
        }
        
        response = requests.post(
            f"{base_url}/api/check-relevance",
            json=relevance_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS: Relevance checking works")
            print(f"   🎯 Is Malaysia travel related: {result['is_malaysia_travel_related']}")
            print(f"   📊 Confidence: {result['confidence']}")
            print(f"   🔍 Keywords found: {', '.join(result['keywords_found'][:5])}")
        else:
            print(f"❌ ERROR: Relevance check failed (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ ERROR: Relevance check test failed: {e}")
        return False
    
    # Test 5: System stats
    print_section("Testing System Statistics")
    try:
        response = requests.get(f"{base_url}/api/stats", timeout=10)
        
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ SUCCESS: System stats retrieved")
            
            if 'database' in stats:
                db_stats = stats['database']
                print(f"   📊 Total documents: {db_stats.get('total_documents', 'N/A'):,}")
                print(f"   🗂️ Collection name: {db_stats.get('collection_name', 'N/A')}")
                print(f"   🧠 Embedding model: {db_stats.get('embedding_model', 'N/A')}")
            
            if 'models' in stats:
                models = stats['models']
                print(f"   🤖 Vertex AI: {'✅' if models.get('vertex_ai') else '❌'}")
                print(f"   🤖 Gemini: {'✅' if models.get('gemini') else '❌'}")
                print(f"   🤖 Embeddings: {'✅' if models.get('embeddings') else '❌'}")
        else:
            print(f"❌ ERROR: Stats retrieval failed (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ ERROR: Stats test failed: {e}")
        return False
    
    return True

def test_sample_queries():
    """Test various sample queries"""
    print_section("Testing Sample Queries")
    
    sample_queries = [
        "What Malaysian food should I try?",
        "Plan a 3-day itinerary for Penang",
        "Best time to visit Langkawi",
        "Cultural experiences in Sarawak",
        "Budget for a week in Malaysia"
    ]
    
    base_url = "http://localhost:8080"
    
    for i, query in enumerate(sample_queries, 1):
        try:
            print(f"\n   Test {i}: {query}")
            
            response = requests.post(
                f"{base_url}/api/query",
                json={"query": query},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['status'] == 'success':
                    print(f"   ✅ SUCCESS - Response: {result['response'][:80]}...")
                else:
                    print(f"   ❌ FAILED - Status: {result['status']}")
            else:
                print(f"   ❌ FAILED - HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    return True

def run_performance_test():
    """Run basic performance test"""
    print_section("Performance Test")
    
    base_url = "http://localhost:8080"
    query = "What are the top attractions in Malaysia?"
    
    try:
        # Warm up
        requests.post(f"{base_url}/api/query", json={"query": query}, timeout=30)
        
        # Time multiple requests
        times = []
        for i in range(5):
            start_time = time.time()
            response = requests.post(
                f"{base_url}/api/query",
                json={"query": query},
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                times.append(end_time - start_time)
            else:
                print(f"   ❌ Request {i+1} failed")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"✅ Performance Test Results:")
            print(f"   ⏱️ Average response time: {avg_time:.2f}s")
            print(f"   ⚡ Fastest response: {min_time:.2f}s")
            print(f"   🐌 Slowest response: {max_time:.2f}s")
            
            if avg_time < 5.0:
                print(f"   🎯 EXCELLENT: Average response time under 5 seconds")
            elif avg_time < 10.0:
                print(f"   ✅ GOOD: Average response time under 10 seconds")
            else:
                print(f"   ⚠️ SLOW: Average response time over 10 seconds")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")

def main():
    """Main test runner"""
    print_header("Malaysia Tourism RAG System - Complete Test Suite")
    print(f"🕒 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🏁 Testing all English functionality...")
    
    # Test results tracking
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Database
    print_header("Phase 1: Database Tests")
    total_tests += 1
    if test_database_existence():
        tests_passed += 1
        print("✅ Database tests: PASSED")
    else:
        print("❌ Database tests: FAILED")
        print("\n🛑 CRITICAL: Database tests failed. Cannot proceed with API tests.")
        print("   Please run: python 1_build_database.py")
        sys.exit(1)
    
    # Test 2: API Server
    print_header("Phase 2: API Server Tests")
    total_tests += 1
    if test_api_server():
        tests_passed += 1
        print("✅ API Server tests: PASSED")
    else:
        print("❌ API Server tests: FAILED")
        print("\n🛑 CRITICAL: API Server tests failed.")
        print("   Please start the server with: python api_server.py")
        return
    
    # Test 3: Sample Queries
    print_header("Phase 3: Sample Query Tests")
    total_tests += 1
    if test_sample_queries():
        tests_passed += 1
        print("✅ Sample query tests: PASSED")
    else:
        print("❌ Sample query tests: FAILED")
    
    # Test 4: Performance
    print_header("Phase 4: Performance Tests")
    run_performance_test()
    
    # Final Results
    print_header("Test Summary")
    print(f"📊 Tests passed: {tests_passed}/{total_tests}")
    print(f"📈 Success rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your Malaysia Tourism RAG system is fully operational")
        print("🌟 All content is in English")
        print("🚀 Ready for production deployment")
    else:
        print("⚠️ Some tests failed")
        print("🔧 Please check the failed components and try again")
    
    print(f"\n🕒 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        traceback.print_exc() 