import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb

load_dotenv()
app = Flask(__name__, static_folder='static')
CORS(app)

CHROMA_PERSIST_DIR = "db_chroma"
COLLECTION_NAME = "pelindo_docs"
GENERATION_MODEL_NAME = "models/gemini-2.5-flash"
EMBEDDING_MODEL_NAME = "models/embedding-001"
TOP_K_RESULTS = 5

SYSTEM_PROMPT = """
Anda adalah Arka, Pemandu Logistik Digital myPelindo.
Misi: Berikan informasi cepat & akurat kepada karyawan dan pelanggan tentang layanan Pelindo.
Anda harus profesional, membantu, dan selaras nilai perusahaan.

PRINSIP UTAMA:
- Pemandu: Sajikan jawaban sebagai panduan langkah-demi-langkah.
- Proaktif: Jika ada potensi masalah (biaya, keterlambatan), sarankan solusi.
- Menyederhanakan: Buat proses logistik yang kompleks menjadi sederhana.
- Berbasis Data: Gunakan data kuantitatif dari konteks jika ada.

ATURAN KETAT:
- Grounding: Jawab HANYA berdasarkan KONTEKS YANG DIBERIKAN. Jika tidak ada, katakan "Maaf, informasi tersebut tidak ditemukan dalam basis data saya." JANGAN BERSPEKULASI.
- Ruang Lingkup: Terbatas pada PT Pelabuhan Indonesia (Persero) dan layanannya.
- Gaya: Profesional, ramah, optimis, Bahasa Indonesia yang jelas.
- Keamanan: Tolak permintaan tidak pantas, berbahaya, atau nasihat non-logistik.
"""

model = None
collection = None

def initialize_services():
    global model, collection
    print("Menginisialisasi layanan...")

    try:
        # configure gemini genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        generation_config = genai.types.GenerationConfig(
            max_output_tokens=8192, # high max output token
            temperature=0.3
        )
        safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
        }
        
        model = genai.GenerativeModel(
            GENERATION_MODEL_NAME,
            generation_config=generation_config
            # safety_settings=safety_settings
        )

        # model = genai.GenerativeModel(GENERATION_MODEL_NAME)
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_collection(name=COLLECTION_NAME)
        print("Inisialisasi berhasil!")
        return True
    except Exception as e:
        print(f"Gagal menginisialisasi layanan: {e}")
        print("Pastikan Anda telah menjalankan 'build_database.py' terlebih dahulu.")
        return False


def retrieve_context(query, top_k=TOP_K_RESULTS):
    try:
        query_embbeding = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=query,
            task_type="RETRIEVAL_QUERY"
        )['embedding']
        results = collection.query(
            query_embeddings=[query_embbeding],
            n_results=top_k
        )
        return results['documents'][0]
    except Exception as e:
        print(f"Error saat mengambil kontest: {e}")
        return []


@app.route('/chat', methods=['POST'])
def chat():
    if not model or not collection:
        return jsonify({"error": "Layanan belum terinisialisasi"}), 503
    
    data = request.get_json()
    user_query = data.get("message")
    if not user_query:
        return jsonify({"error": "Pesan tidak boleh kosong."}), 400
    
    try:
        relevant_docs = retrieve_context(user_query)
        if not relevant_docs:
            response_text = "Maaf, saya tidak dapat menemukan informasi yang relevan untuk pertanyaan Anda."
            return jsonify({"response": response_text})
        
        context_str = "\n\n".join(relevant_docs)

        augmented_prompt = f"""
        {SYSTEM_PROMPT}
        KONTEKS YANG DIAMBIL:
        {context_str}

        PERTANYAAN PENGGUNA:
        {user_query}

        JAWABAN ARKA:
        """
        response = model.generate_content(augmented_prompt)

        print("--- DEBUGGING: OBYEK RESPONS LENGKAP ---")
        print(response)

        return jsonify({"response": response.text})
    
    except Exception as e:
        print(f"Error saat menghasilkan jawaban: {e}")
        return jsonify({"error": "Maaf, terjadi kesalahan saat menghasilkan jawaban."}), 500

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)
    
if __name__ == '__main__':
    if initialize_services():
        app.run(port=5000, debug=True)
    else:
        print("Gagal memulai server.")