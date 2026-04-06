"""
Front-End Mentor Bot - Main Flask Application
Web interface for RAG-powered chatbot using Groq API
"""

import os
import traceback
from typing import List, Dict
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from groq import Groq
from rag_pipeline import RAGPipeline, QueryRouter

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 1000))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50))

# Resolve paths relative to this file so they always point to project root
# app.py lives inside  project_root/app/  so we go one level up
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
CORPUS_DIR = os.path.join(_PROJECT_ROOT, "corpus")
PERSIST_DIR = os.path.join(_PROJECT_ROOT, "vector_db")

# ── Validate API key ──────────────────────────────────────────────────────────
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Please add it to your .env file.\n"
        "Get a free key at: https://console.groq.com"
    )

print(f"[STARTUP] Groq API key loaded: {GROQ_API_KEY[:8]}...")
print(f"[STARTUP] Model: {GROQ_MODEL}")
print(f"[STARTUP] Corpus dir: {CORPUS_DIR}")
print(f"[STARTUP] Vector DB dir: {PERSIST_DIR}")

# ── Verify Groq API key works before starting ─────────────────────────────────
try:
    _test_client = Groq(api_key=GROQ_API_KEY)
    _test_response = _test_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=5
    )
    print(f"[STARTUP] ✅ Groq API key verified successfully!")
except Exception as _e:
    print(f"\n{'='*60}")
    print(f"[STARTUP] ❌ GROQ API KEY VERIFICATION FAILED!")
    print(f"[STARTUP] Error: {_e}")
    print(f"[STARTUP] Your API key is invalid or has been revoked.")
    print(f"[STARTUP] Go to https://console.groq.com and generate a NEW key.")
    print(f"{'='*60}\n")
    raise RuntimeError(f"Groq API key is invalid: {_e}")

# ── Initialize Groq client ────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)

# ── Initialize RAG pipeline ───────────────────────────────────────────────────
print("Initializing RAG Pipeline...")
rag_pipeline = RAGPipeline(
    corpus_dir=CORPUS_DIR,
    persist_dir=PERSIST_DIR,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
)

# Load corpus on startup
print("Loading corpus...")
rag_pipeline.load_corpus()
print(f"[STARTUP] ✅ Corpus loaded. Total chunks: {rag_pipeline.collection.count()}")

# Conversation storage (in production, use Redis or database)
conversations = {}


def build_system_prompt(context: str, routing_info: Dict) -> str:
    """Build system prompt based on context and routing info."""
    is_evaluation = routing_info.get('is_evaluation', False)
    category = routing_info.get('category', 'tech')

    base_prompt = f"""You are a Front-End Development Mentor helping junior developers prepare for internships.

CONTEXT FROM KNOWLEDGE BASE:
{context}

ROUTING ANALYSIS:
- Primary Category: {category}
- Query Type: {'Interview Answer Evaluation' if is_evaluation else 'Information Request'}

"""

    if is_evaluation:
        instructions = """INSTRUCTIONS - EVALUATION MODE:
This is an EVALUATION task. The user is answering an interview question or wants feedback.
1. Retrieve the correct/ideal answer from the context above
2. Compare it to the user's answer
3. Provide constructive feedback highlighting what they got right and what could be improved
4. Do NOT just give them the complete answer - guide them to improve
5. Be encouraging but honest
6. Point out specific missing details or misconceptions
"""
    else:
        instructions = """INSTRUCTIONS - INFORMATION MODE:
This is an INFORMATION REQUEST.
1. Use the retrieved context above to answer the question accurately
2. Provide code examples when relevant (ensure syntax is perfect and properly formatted)
3. Be concise but thorough
4. If the context doesn't contain the answer, say so and offer to help differently
5. For CV questions, be specific and actionable with real examples
6. For technical questions, include working code examples with proper syntax
"""

    if category == 'tech':
        instructions += "\n\nIMPORTANT: Ensure all code examples are properly formatted and syntactically correct. Use code blocks with proper syntax."

    instructions += "\n\nRespond in a friendly, mentoring tone. Be helpful but not condescending."

    return base_prompt + instructions


def generate_response(query: str, conversation_id: str) -> Dict:
    """Generate response using RAG pipeline and Groq API."""

    # Get or create conversation history
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    conversation_history = conversations[conversation_id]

    try:
        # ── Step 1: RAG retrieval ─────────────────────────────────────────────
        print(f"[CHAT] Processing query: {query[:80]}...")
        retrieved_docs, routing_info = rag_pipeline.process_query(
            query, conversation_history
        )
        print(f"[CHAT] Retrieved {len(retrieved_docs)} docs | category={routing_info['category']}")

        # ── Step 2: Build context ─────────────────────────────────────────────
        context_parts = []
        for doc in retrieved_docs:
            source = doc['metadata']['source']
            content = doc['content']
            context_parts.append(f"[Source: {source}]\n{content}")
        context = "\n\n---\n\n".join(context_parts)

        # ── Step 3: Build system prompt ───────────────────────────────────────
        system_prompt = build_system_prompt(context, routing_info)

        # ── Step 4: Call Groq API ─────────────────────────────────────────────
        print(f"[CHAT] Calling Groq API (model={GROQ_MODEL})...")
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=MAX_TOKENS
        )

        response_text = response.choices[0].message.content
        print(f"[CHAT] ✅ Groq responded ({len(response_text)} chars)")

        # Update conversation history
        conversation_history.append(f"User: {query}")
        conversation_history.append(f"Assistant: {response_text}")

        return {
            'success': True,
            'message': response_text,
            'metadata': {
                'category': routing_info['category'],
                'is_evaluation': routing_info.get('is_evaluation', False),
                'sources': [doc['metadata']['source'] for doc in retrieved_docs],
                'num_retrieved': len(retrieved_docs)
            }
        }

    except Exception as e:
        # Print the FULL traceback to terminal so you can see exactly what failed
        full_error = traceback.format_exc()
        print(f"\n{'='*60}")
        print(f"[CHAT ERROR] Exception in generate_response:")
        print(full_error)
        print(f"{'='*60}\n")

        return {
            'success': False,
            'error': str(e),
            # Show the real error in the UI so you know what's wrong
            'message': f"❌ Error: {str(e)}"
        }


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint for sending messages"""
    data = request.json
    query = data.get('message', '').strip()
    conversation_id = data.get('conversation_id', 'default')

    if not query:
        return jsonify({'success': False, 'error': 'Empty message'}), 400

    response = generate_response(query, conversation_id)
    return jsonify(response)


@app.route('/api/test', methods=['GET'])
def test():
    """Test endpoint to verify API is running"""
    return jsonify({
        'status': 'ok',
        'message': 'Front-End Mentor Bot API is running',
        'corpus_loaded': rag_pipeline.collection.count() > 0,
        'total_chunks': rag_pipeline.collection.count()
    })


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get statistics about the knowledge base"""
    return jsonify({
        'total_chunks': rag_pipeline.collection.count(),
        'active_conversations': len(conversations),
        'categories': ['tech', 'interview', 'cv'],
        'model': GROQ_MODEL
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

    print("\n" + "=" * 60)
    print("🎓 FRONT-END MENTOR BOT STARTING")
    print("=" * 60)
    print(f"Server: http://localhost:{port}")
    print(f"API Test: http://localhost:{port}/api/test")
    print(f"Total Chunks Loaded: {rag_pipeline.collection.count()}")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=port, debug=debug)