import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from extract_documents import DocumentExtractor


# =====================================================
# Configuration
# =====================================================

CHROMA_DB_DIR = Path("chroma_db")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =====================================================
# Load Documents
# =====================================================

print("=" * 60)
print("Loading documents...")
print("=" * 60)

extractor = DocumentExtractor()

documents = extractor.load_all_documents()

print(f"Loaded {len(documents)} documents.")


# =====================================================
# Split Documents
# =====================================================

print("\nSplitting documents into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)

split_documents = text_splitter.split_documents(documents)

print(f"Created {len(split_documents)} chunks.")


# =====================================================
# Load Embedding Model
# =====================================================

print("\nLoading HuggingFace Embedding Model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Embedding model loaded.")


# =====================================================
# Delete Existing Database (Optional)
# =====================================================

if CHROMA_DB_DIR.exists():

    print("\nExisting Chroma database found.")

    shutil.rmtree(CHROMA_DB_DIR)

    print("Old database deleted.")


# =====================================================
# Create Chroma Vector Store
# =====================================================

print("\nCreating Chroma Vector Database...")

vectordb = Chroma.from_documents(
    documents=split_documents,
    embedding=embeddings,
    persist_directory=str(CHROMA_DB_DIR),
)

print("Vector database created successfully.")


# =====================================================
# Verification
# =====================================================

print("\nDatabase Statistics")

print("-" * 60)

collection = vectordb._collection

print(f"Total Chunks Stored : {collection.count()}")

print(f"Database Location   : {CHROMA_DB_DIR.resolve()}")

print(f"Embedding Model     : {EMBEDDING_MODEL}")


# =====================================================
# Test Retrieval
# =====================================================

print("\nTesting Retrieval...")

query = "Patient experienced severe bleeding after taking Warfarin"

results = vectordb.similarity_search(query, k=3)

print("\nTop 3 Retrieved Documents\n")

for i, doc in enumerate(results, start=1):

    print("=" * 70)

    print(f"Result {i}")

    print("-" * 70)

    print("Metadata")

    print(doc.metadata)

    print("\nContent")

    print(doc.page_content[:500])

    print()


print("=" * 60)
print("Vector Database Ready.")
print("=" * 60)