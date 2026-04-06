#!/usr/bin/env python
"""
diagnose.py - Run this from the project root to find the EXACT error.
Usage: python diagnose.py
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'app'))

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, '.env'), override=True)

GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
GROQ_MODEL   = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')

print("\n" + "=" * 60)
print("FRONT-END MENTOR BOT — DIAGNOSTICS")
print("=" * 60)

# ── 1. API key format ─────────────────────────────────────────────────────────
print("\n[1] Checking API key format...")
if not GROQ_API_KEY:
    print("    ❌ GROQ_API_KEY is empty! Check your .env file.")
    sys.exit(1)
elif GROQ_API_KEY.startswith('"') or GROQ_API_KEY.endswith('"'):
    print(f"    ❌ Key has quotes: {GROQ_API_KEY[:12]}...")
    print("       Remove the quotes from your .env file.")
    sys.exit(1)
elif not GROQ_API_KEY.startswith('gsk_'):
    print(f"    ⚠  Key doesn't start with gsk_: {GROQ_API_KEY[:8]}...")
else:
    print(f"    ✅ Key format looks correct: {GROQ_API_KEY[:8]}...")

# ── 2. Live Groq API test ─────────────────────────────────────────────────────
print("\n[2] Testing Groq API connection (live call)...")
try:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: WORKING"}],
        max_tokens=10
    )
    answer = resp.choices[0].message.content.strip()
    print(f"    ✅ Groq API works! Response: '{answer}'")
except Exception as e:
    print(f"    ❌ Groq API FAILED: {e}")
    print()
    print("    ► THIS IS YOUR PROBLEM.")
    print("    ► Your API key is INVALID or REVOKED.")
    print("    ► You must generate a NEW key at: https://console.groq.com")
    print("    ► Then update GROQ_API_KEY in your .env file.")
    sys.exit(1)

# ── 3. NLTK check ────────────────────────────────────────────────────────────
print("\n[3] Checking NLTK tokenizer...")
try:
    from nltk.tokenize import word_tokenize
    result = word_tokenize("Hello world this is a test")
    print(f"    ✅ NLTK works! Tokens: {result[:4]}...")
except Exception as e:
    print(f"    ❌ NLTK broken: {e}")
    print("    ► Run: python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab')\"")
    print("    ► Then delete: C:\\Users\\<you>\\AppData\\Roaming\\nltk_data\\tokenizers\\punkt")

# ── 4. ChromaDB / vector DB check ────────────────────────────────────────────
print("\n[4] Checking vector database...")
try:
    import chromadb
    from chromadb.config import Settings
    persist_dir = os.path.join(_PROJECT_ROOT, 'vector_db')
    client_db = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    col = client_db.get_or_create_collection("frontend_mentor_docs")
    count = col.count()
    print(f"    ✅ Vector DB accessible. Chunks stored: {count}")
    if count == 0:
        print("    ⚠  Collection is EMPTY — corpus was never loaded.")
        print("       Fix NLTK first, then delete the vector_db folder and restart.")
except Exception as e:
    print(f"    ❌ ChromaDB error: {e}")

# ── 5. Full end-to-end test ───────────────────────────────────────────────────
print("\n[5] End-to-end test (RAG + Groq)...")
try:
    from rag_pipeline import RAGPipeline  # type: ignore
    pipeline = RAGPipeline(
        corpus_dir=os.path.join(_PROJECT_ROOT, 'corpus'),
        persist_dir=os.path.join(_PROJECT_ROOT, 'vector_db')
    )
    pipeline.load_corpus()
    docs, routing = pipeline.process_query("What is HTML?")
    print(f"    ✅ RAG pipeline returned {len(docs)} docs (category={routing['category']})")

    from groq import Groq
    g = Groq(api_key=GROQ_API_KEY)
    context = "\n".join(d['content'][:200] for d in docs[:2])
    r = g.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": f"Answer using this context: {context}"},
            {"role": "user", "content": "What is HTML?"}
        ],
        max_tokens=100
    )
    print(f"    ✅ Full pipeline works! Preview: {r.choices[0].message.content[:80]}...")
except Exception as e:
    import traceback
    print(f"    ❌ End-to-end failed: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60 + "\n")
