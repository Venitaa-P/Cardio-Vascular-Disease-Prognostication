import os
import logging
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# -----------------------------
# LOGGING CONFIGURATION
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# CONFIGURATION
# -----------------------------
DOC_PATH = os.path.join("knowledge_base", "heart_health_guide.txt")
FAISS_DB_PATH = "faiss_index"
MODEL_ID = "google/flan-t5-base"

# -----------------------------
# FUNCTION: Create or Load FAISS Vector Store
# -----------------------------
def create_or_load_vector_store():
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if os.path.exists(FAISS_DB_PATH):
        logger.info("✅ Loading existing FAISS index...")
        db = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        return db

    logger.info("📘 Creating new FAISS index...")

    if not os.path.exists(DOC_PATH):
        # Create a basic knowledge base if it doesn't exist
        os.makedirs("knowledge_base", exist_ok=True)
        basic_knowledge = """
        Heart Disease Information:
        - Risk factors include high blood pressure, high cholesterol, smoking, diabetes, obesity
        - Symptoms can include chest pain, shortness of breath, palpitations
        - Prevention methods: regular exercise, healthy diet, no smoking
        - Common types: coronary artery disease, heart failure, arrhythmias
        - Regular check-ups are important for early detection
        """
        with open(DOC_PATH, 'w', encoding='utf-8') as f:
            f.write(basic_knowledge)
        logger.info(f"✅ Created basic knowledge base at {DOC_PATH}")

    loader = TextLoader(DOC_PATH, encoding='utf-8')
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(FAISS_DB_PATH)

    logger.info("✅ FAISS index created and saved successfully.")
    return db

# -----------------------------
# GLOBAL INITIALIZATION
# -----------------------------
RAG_READY = False
vector_store = None
generator = None

try:
    # ✅ Create or load FAISS vector store
    vector_store = create_or_load_vector_store()

    # ✅ Load local model for text generation
    logger.info("⚙️ Loading local FLAN-T5 model...")
    generator = pipeline("text2text-generation", model=MODEL_ID)

    logger.info("✅ RAG System initialized successfully (Local Mode).")
    RAG_READY = True

except Exception as e:
    logger.error(f"❌ RAG initialization failed: {e}")
    RAG_READY = False

# -----------------------------
# FUNCTION: Ask the RAG System
# -----------------------------
def ask_rag_system(query: str) -> dict:
    """
    Enhanced RAG system that returns both response and sources
    """
    if not RAG_READY:
        return {
            "response": "⚠️ The health knowledge base is currently unavailable. Please try again later.",
            "sources": []
        }
    
    try:
        # Simple retrieval and generation without complex chain
        if vector_store:
            docs = vector_store.similarity_search(query, k=3)
            context = "\n\n".join([f"Source {i+1}: {doc.page_content}" 
                                 for i, doc in enumerate(docs)])
            
            prompt = f"""Based on the following medical information, answer the user's question clearly and helpfully.

Context Information:
{context}

User Question: {query}

Please provide a helpful, accurate response based on the available medical information:"""
            
            if generator:
                response = generator(prompt, max_length=400, do_sample=False)[0]['generated_text']
            else:
                response = "I understand you're asking about heart health. Based on general knowledge: This appears to be related to cardiovascular health, but I recommend consulting with a healthcare professional for personalized medical advice."
            
            sources = [doc.page_content[:200] + "..." for doc in docs]
            
            return {
                "response": response,
                "sources": sources
            }
        else:
            return {
                "response": "I'm here to help with heart health questions. While I can provide general information, please consult a healthcare provider for medical advice specific to your situation.",
                "sources": []
            }
            
    except Exception as e:
        logger.error(f"Error in RAG system: {e}")
        return {
            "response": "I apologize, but I'm having trouble accessing the health information right now. Please try again later or consult a healthcare professional for immediate concerns.",
            "sources": []
        }

# -----------------------------
# TEST BLOCK
# -----------------------------
if __name__ == '__main__':
    print("🔍 Testing RAG System:")
    test_questions = [
        "What are common heart disease symptoms?",
        "How can I prevent heart disease?",
        "What is high blood pressure?"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        result = ask_rag_system(question)
        print(f"A: {result['response']}")
        if result['sources']:
            print("Sources:", result['sources'][:1])  # Show first source only