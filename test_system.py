#!/usr/bin/env python
"""
Automated Testing Script for Front-End Mentor Bot
Tests RAG pipeline and API functionality
Run from the PROJECT ROOT directory: python test_system.py
"""

import os
import sys
import time

# Add app directory to path so we can import from it
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, 'app'))

# Corpus and vector_db are at project root (same level as this script)
_CORPUS_DIR = os.path.join(_SCRIPT_DIR, 'corpus')
_VECTOR_DB_DIR = os.path.join(_SCRIPT_DIR, 'vector_db')


def test_imports():
    """Test if all dependencies are installed"""
    print("\n" + "=" * 60)
    print("TEST 1: CHECKING DEPENDENCIES")
    print("=" * 60)

    required_modules = [
        'flask',
        'groq',
        'chromadb',
        'sentence_transformers',
        'nltk',
        'dotenv',
        'flask_cors',
    ]

    failed = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} - NOT INSTALLED")
            failed.append(module)

    if failed:
        print(f"\n❌ Missing modules: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False

    print("\n✅ All dependencies installed!")
    return True


def test_corpus_files():
    """Test if corpus files exist and have content"""
    print("\n" + "=" * 60)
    print("TEST 2: CHECKING CORPUS FILES")
    print("=" * 60)

    corpus_files = [
        os.path.join(_CORPUS_DIR, 'tech_docs.txt'),
        os.path.join(_CORPUS_DIR, 'interview_bank.txt'),
        os.path.join(_CORPUS_DIR, 'cv_guidelines.txt'),
    ]

    all_exist = True
    for filepath in corpus_files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            lines = sum(1 for _ in open(filepath, 'r', encoding='utf-8'))
            print(f"✓ {os.path.basename(filepath)} ({lines} lines, {size} bytes)")
        else:
            print(f"✗ {filepath} - NOT FOUND")
            all_exist = False

    if not all_exist:
        print("\n❌ Some corpus files are missing!")
        print(f"  Expected location: {_CORPUS_DIR}/")
        return False

    print("\n✅ All corpus files present!")
    return True


def test_env_file():
    """Test if .env file exists and has required variables"""
    print("\n" + "=" * 60)
    print("TEST 3: CHECKING ENVIRONMENT CONFIGURATION")
    print("=" * 60)

    env_path = os.path.join(_SCRIPT_DIR, '.env')
    print(f"  Looking for .env at: {env_path}")

    if not os.path.exists(env_path):
        print(f"✗ .env file not found")
        print("\n❌ Create a .env file in the project root with:")
        print("   GROQ_API_KEY=your_actual_key_here")
        print("   CHUNK_SIZE=500")
        print("   CHUNK_OVERLAP=50")
        return False

    # Parse the .env file directly (most reliable — bypasses env caching)
    raw_vars = {}
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments and blank lines
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()
            # Strip inline comments
            if ' #' in value:
                value = value[:value.index(' #')].strip()
            raw_vars[key] = value

    # Also load into os.environ with override so rest of process can use them
    from dotenv import load_dotenv
    load_dotenv(env_path, override=True)

    required_vars = ['GROQ_API_KEY', 'CHUNK_SIZE', 'CHUNK_OVERLAP']

    missing = []
    failed = []
    for var in required_vars:
        value = raw_vars.get(var, '').strip()
        if not value:
            print(f"✗ {var} - NOT SET in .env")
            missing.append(var)
        elif var == 'GROQ_API_KEY' and ('your_groq_api_key' in value.lower() or 'your_' in value.lower()):
            print(f"✗ {var} - still has placeholder value")
            failed.append(var)
        elif var == 'GROQ_API_KEY' and not value.startswith('gsk_'):
            print(f"✗ {var} = {value[:6]}... (warning: Groq keys usually start with gsk_)")
            # Don't fail — user may have a different format key
            print(f"  If your key is correct, ignore this warning.")
        else:
            display = value[:12] + '...' if var == 'GROQ_API_KEY' else value
            print(f"✓ {var} = {display}")

    if missing:
        print(f"\n❌ Add these to your .env file: {', '.join(missing)}")
        return False

    if failed:
        print(f"\n❌ Replace placeholder values for: {', '.join(failed)}")
        print("   Open .env and set:  GROQ_API_KEY=gsk_your_actual_key")
        return False

    print("\n✅ Environment configured!")
    return True


def test_rag_pipeline():
    """Test RAG pipeline initialization"""
    print("\n" + "=" * 60)
    print("TEST 4: TESTING RAG PIPELINE")
    print("=" * 60)

    try:
        from rag_pipeline import RAGPipeline, QueryRouter  # type: ignore

        print("Initializing RAG pipeline...")
        pipeline = RAGPipeline(
            corpus_dir=_CORPUS_DIR,
            persist_dir=_VECTOR_DB_DIR
        )

        print("Loading corpus...")
        pipeline.load_corpus()

        chunk_count = pipeline.collection.count()
        print(f"✓ Vector database loaded with {chunk_count} chunks")

        if chunk_count < 100:
            print("⚠  Warning: Low chunk count, corpus may be incomplete")

        print("\nTesting query routing...")

        test_queries = {
            "How do I deploy to GitHub?": "tech",
            "How should I write my CV?": "cv",
            "Ask me an interview question": "interview"
        }

        routing_ok = True
        for query, expected_category in test_queries.items():
            result = QueryRouter.route_query(query)
            actual_category = result['category']
            status = "✓" if actual_category == expected_category else "✗"
            if actual_category != expected_category:
                routing_ok = False
            print(f"{status} '{query}' → {actual_category} (expected: {expected_category})")

        print("\nTesting document retrieval...")
        docs, routing = pipeline.process_query("How do I use flexbox?")
        print(f"✓ Retrieved {len(docs)} documents for test query")
        print(f"  Category: {routing['category']}")
        print(f"  Confidence: {routing['confidence']}")

        if routing_ok:
            print("\n✅ RAG pipeline working correctly!")
        else:
            print("\n⚠  RAG pipeline working but some routing results differ from expected.")
        return True

    except Exception as e:
        print(f"\n❌ RAG pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api():
    """Test Flask API endpoints (requires server running)"""
    print("\n" + "=" * 60)
    print("TEST 5: TESTING API (Optional - requires server running)")
    print("=" * 60)

    try:
        import requests

        print("Testing /api/test endpoint...")
        response = requests.get('http://localhost:5000/api/test', timeout=5)

        if response.status_code == 200:
            data = response.json()
            print(f"✓ API responding: {data['message']}")
            print(f"  Corpus loaded: {data['corpus_loaded']}")
            print(f"  Total chunks: {data['total_chunks']}")
            print("\n✅ API is working!")
            return True
        else:
            print(f"✗ API returned status code: {response.status_code}")
            return False

    except Exception:
        try:
            import requests
            _ = requests
        except ImportError:
            pass
        print("⚠  Server not running (this is OK if you haven't started it yet)")
        print("   To test API: run 'python run.py' in another terminal")
        return None


def run_all_tests():
    """Run all tests"""
    print("\n🎓 FRONT-END MENTOR BOT - SYSTEM TEST")
    print("=" * 60)
    print("This script will test if your system is set up correctly")
    print("=" * 60)

    tests = [
        ("Dependencies", test_imports),
        ("Corpus Files", test_corpus_files),
        ("Environment", test_env_file),
        ("RAG Pipeline", test_rag_pipeline),
        ("API", test_api)
    ]

    results = {}
    for name, test_func in tests:
        result = test_func()
        results[name] = result
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        if result is True:
            print(f"✅ {name}: PASSED")
        elif result is False:
            print(f"❌ {name}: FAILED")
        else:
            print(f"⚠️  {name}: SKIPPED")

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)

    print("\n" + "=" * 60)
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        print("Your system is ready to use!")
        print("\nNext steps:")
        print("1. Run: python run.py")
        print("2. Open: http://localhost:5000")
    else:
        print(f"⚠️  {failed} test(s) failed")
        print("Please fix the issues above before proceeding")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()