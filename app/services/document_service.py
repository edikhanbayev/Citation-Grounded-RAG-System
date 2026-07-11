import os
from concurrent.futures import ThreadPoolExecutor
from flask import current_app
from werkzeug.utils import secure_filename
from app.models import RagEngine

# Local background task pool (2 concurrent workers)
executor = ThreadPoolExecutor(max_workers=2)
rag = RagEngine()


def process_async_upload(file_data, clearance_level):
    """Saves file to disk instantly, then schedules background RAG vector indexing."""
    filename = secure_filename(file_data.filename)

    # Safely target the uploads directory relative to current runtime
    upload_dir = os.path.join(current_app.root_path, '..', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    saved_path = os.path.join(upload_dir, filename)

    # 1. Fast disk write
    file_data.save(saved_path)

    #  DECOUPLED HANDOFF: Process PDF text parsing & vector math in a background thread
    executor.submit(rag.process_and_index_pdf, saved_path, clearance_level)
    return filename


def delete_document_source(filename):
    """Completely expunges text vectors from ChromaDB and wipes the file from local disk."""
    # 1. Purge matching data chunks from ChromaDB
    vector_success = rag.delete_source(filename)
    if not vector_success:
        return False, "Failed to fully clear vector fragments from memory."

    # 2. Local physical file deletion
    try:
        upload_dir = os.path.join(current_app.root_path, '..', 'uploads')
        file_path = os.path.join(upload_dir, filename)

        if os.path.exists(file_path):
            os.remove(file_path)
            return True, f"'{filename}' has been completely expunged from ChromaDB and local storage."
        return True, "Vectors cleared, but file did not exist on the local disk path."

    except Exception as e:
        return True, f"Vectors cleared, but local disk cleanup threw an issue: {str(e)}"