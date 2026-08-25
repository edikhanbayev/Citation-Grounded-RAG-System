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
    # Saving file to disk instantly, handling duplicates, then schedules background RAG vector indexing
    filename = secure_filename(file_data.filename)

    # Query the SQLite to see if this filename is a duplicate
    existing_doc = Document.query.filter_by(filename=filename).first()

    if existing_doc:
        print(f"[*] Re-upload Detected: '{filename}' already exists. Initiating overwrite sequence...")

        # Clear old matching vector fragments and selective cache entries from ChromaDB
        rag.delete_source(filename)

        # Repurpose the existing record instead of inserting a duplicate row
        existing_doc.clearance_level = clearance_level
        existing_doc.status = 'Processing'
        db.session.commit()

        doc_id = existing_doc.id
    else:
        # If the file is genuinely new, execute a standard structural insert
        new_doc = Document(filename=filename, clearance_level=clearance_level, status='Processing')
        db.session.add(new_doc)
        db.session.commit()

        doc_id = new_doc.id

    # Safely target the uploads directory relative to current runtime
    upload_dir = os.path.join(current_app.root_path, '..', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    saved_path = os.path.join(upload_dir, filename)

    # 4. Fast disk write (Werkzeug's save automatically overwrites the old physical file)
    file_data.save(saved_path)

    # Decoupled handoff: Pass the tracked doc_id into the background worker pool
    executor.submit(async_vector_pipeline, doc_id, saved_path, clearance_level)
    return filename


def async_vector_pipeline(doc_id, file_path, clearance_level):
    # Wrapper function execution block running inside the background thread pool
    from app import app, db
    from app.models import Document

    # Using an application context wrapper block to ensure thread safety with SQLite
    with app.app_context():
        db_document = db.session.get(Document, doc_id)
        try:
            # Execute PyMuPDF chunking, selective cache invalidation, and RAM refresh pipelines
            rag.process_and_index_pdf(file_path, clearance_level)

            # Update status to Ready upon completion
            db_document.status = 'Ready'
            db.session.commit()
        except Exception as e:
            print(f"[!] Background Indexing Failure for Doc ID {doc_id}: {str(e)}")
            db_document.status = 'Failed'
            db.session.commit()


def delete_document_source(filename):
    # Completely expunges text vectors from ChromaDB, selectively invalidates dependent cache entries, and wipes the file from local disk
    # 1. Purge matching data chunks and dependent cache entries from ChromaDB
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