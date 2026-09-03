# app.py
import os
import gc
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from services.pdf_reader import extract_text
from services.chunker import create_chunks
from services.embeddings import create_embeddings as get_embeddings
from services.vector_store import create_index, search_index
from services.llm_service import generate_answer

app = Flask(__name__)

UPLOAD_FOLDER = "data/uploads"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory storage for active session (for single-user local dev)
CURRENT_DATA = {
    "chunks": [],
    "index": None
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("pdf")

    if not file or file.filename == "":
        return "Please select a PDF file.", 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        return "Please upload a PDF file.", 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(file_path)
        CURRENT_DATA["chunks"] = []
        CURRENT_DATA["index"] = None
        gc.collect()

        text = extract_text(file_path)
        chunks = create_chunks(text)
        if not chunks:
            return "The PDF did not contain extractable text.", 400

        chunk_embeddings = get_embeddings(chunks)
        index = create_index(chunk_embeddings)
        CURRENT_DATA["chunks"] = chunks
        CURRENT_DATA["index"] = index
    except Exception:
        app.logger.exception("PDF upload and indexing failed for %s", filename)
        return jsonify({
            "error": "The PDF could not be processed. Try a smaller, text-based PDF."
        }), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return render_template(
        "index.html",
        text=text,
        chunks=chunks,
        filename=filename,
        uploaded=True
    )

@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True) or {}
    user_query = payload.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "Please enter a question."}), 400
    
    if CURRENT_DATA["index"] is None or not CURRENT_DATA["chunks"]:
        return jsonify({"error": "Please upload a PDF document first."}), 400

    try:
        query_embedding = get_embeddings([user_query])[0]
        relevant_chunks = search_index(
            query_embedding,
            CURRENT_DATA["index"],
            CURRENT_DATA["chunks"],
            k=3
        )
        answer = generate_answer(user_query, relevant_chunks)
    except ValueError as error:
        return jsonify({"error": str(error)}), 503
    except Exception as error:
        app.logger.exception("Document agent request failed")
        if "401" in str(error) or "invalid_api_key" in str(error).lower():
            return jsonify({
                "error": "The Groq API key is invalid. Update GROQ_API_KEY in .env and restart Flask."
            }), 503
        return jsonify({"error": "The AI agent could not answer right now."}), 502

    return jsonify({
        "answer": answer,
        "sources": relevant_chunks
    })


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(error):
    return jsonify({"error": "PDF uploads must be 10 MB or smaller."}), 413

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1"
    )