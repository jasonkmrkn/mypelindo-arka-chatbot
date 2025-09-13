import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

SOURCE_DOCUMENTS_DIR = "dokumen_sumber"
CHROMA_PERSIST_DIR = "db_chroma"
COLLECTION_NAME = "pelindo_docs"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 250

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def extract_text_from_pdfs(pdf_folder_path):
    all_docs_content = []
    print(f"Membaca file PDF dari folder: {pdf_folder_path}")
    if not os.path.isdir(pdf_folder_path):
        print(f"Error: Folder '{pdf_folder_path}' tidak ditemukan.")
        return []
    
    for filename in os.listdir(pdf_folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder_path, filename)
            try:
                reader = PdfReader(file_path, filename)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    all_docs_content.append({
                        "text": text,
                        "metadata": {
                            "source_document": filename,
                            "page": page_num + 1
                        }
                    })
                print(f"  - Berhasil memproses '{filename}'")
            except Exception as e:
                print(f"  - Gagal memproses '{filename}': {e}")

    return all_docs_content

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks_with_metadata = []
    for doc in documents:
        cleaned_text = clean_text(doc["text"])
        splits = text_splitter.split_text(cleaned_text)
        for split in splits:
            chunks_with_metadata.append({
                "text": split,
                "metadata": doc["metadata"]
            })

    return chunks_with_metadata

def main():
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    except Exception as e:
        print(f"Error configuring Google AI: {e}")
        exit()

    documents = extract_text_from_pdfs(SOURCE_DOCUMENTS_DIR)
    if not documents:
        print("Tidak ada dokumen yang ditemukan atau berhasil diproses. Proses dihentikan.")
        return
    
    text_chunks_with_metadata = get_text_chunks(documents)
    print(f"\nTotal dokumen dipecah menjadi {len(text_chunks_with_metadata)} chunks.")

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    print(f"Mengakses atau membuat koleksi: {COLLECTION_NAME}")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("Memulai proses embedding dan penyimpanan ke ChromaDB...")
    batch_size = 100
    total_chunks = len(text_chunks_with_metadata)

    for i in range(0, total_chunks, batch_size):
        batch = text_chunks_with_metadata[i:i+batch_size]

        texts_to_embed = [item['text'] for item in batch]
        metadatas = [item['metadata'] for item in batch]
        ids = [f"chunk_{i+j}" for j in range(len(batch))]

        try:
            result = genai.embed_content(
                model='models/embedding-001',
                content=texts_to_embed,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings = result['embedding']

            collection.add(
                embeddings=embeddings,
                documents=texts_to_embed,
                metadatas=metadatas,
                ids=ids
            )
            print(f"  - Batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
        except Exception as e:
            print(f"  - Gagal memproses batch {i//batch_size + 1}: {e}")

        print("\nProses pembangunan basis data vektor selesai.")
        print(f"Total dokument yang tersimpan di koleksi '{COLLECTION_NAME}': {collection.count()}")

if __name__ == "__main__":
    main()