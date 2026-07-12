import os
from concurrent.futures import ThreadPoolExecutor
from flask import current_app
from werkzeug.utils import secure_filename
from app.models import RagEngine, Document
from app import db, app

# Local background task pool (2 concurrent workers)
executor = ThreadPoolExecutor(max_workers=2)
rag = RagEngine()


def process_async_upload(file_data, clearance_level):
    """Saves file to disk instantly, then schedules background RAG vector indexing."""
    filename = secure_filename(file_data.filename)

    # 1. Instantly register the file in SQLite with a 'Processing' state
    new_doc = Document(filename=filename, clearance_level=clearance_level, status='Processing')
    db.session.add(new_doc)
    db.session.commit()

    # Safely target the uploads directory relative to current runtime
    upload_dir = os.path.join(current_app.root_path, '..', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    saved_path = os.path.join(upload_dir, filename)

    # 1. Fast disk write
    file_data.save(saved_path)

    #  DECOUPLED HANDOFF: Process PDF text parsing & vector math in a background thread
    executor.submit(async_vector_pipeline, new_doc.id, saved_path, clearance_level)
    return filename


def async_vector_pipeline(doc_id, file_path, clearance_level):
    """Wrapper function execution block running inside the background thread pool."""
    from app import app, db
    from app.models import Document

    # We use an application context wrapper block to ensure thread safety with SQLite
    with app.app_context():
        db_document = db.session.get(Document, doc_id)
        try:
            # Execute your PyMuPDF chunking and parsing pipelines here...
            rag.process_and_index_pdf(file_path, clearance_level)

            # Update status to Ready upon completion
            db_document.status = 'Ready'
            db.session.commit()
        except Exception as e:
            db_document.status = 'Failed'
            db.session.commit()

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