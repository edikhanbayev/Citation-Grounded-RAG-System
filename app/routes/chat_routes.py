from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from app import db
from app.models import ChatMessage, RagEngine, Citation, Document
import re

chat_bp = Blueprint('chat', __name__)
rag = RagEngine()

@chat_bp.route('/', methods=['GET'])
@login_required
def index():
    history = ChatMessage.query.filter_by(user_id=current_user.id).order_by(ChatMessage.timestamp.asc()).all()
    return render_template('index.html', history=history)

@chat_bp.route('/ask', methods=['POST'])
@login_required
def ask():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({"error": "Empty payload"}), 400

    # 1. Instantly log and commit the incoming question text string
    user_msg = ChatMessage(user_id=current_user.id, sender='user', message=question)
    db.session.add(user_msg)
    db.session.commit() # 🚀 Safe state separation

    # Resolve clearance array parameters based on user access tokens
    if current_user.role == 'admin':
        allowed = ["public", "internal", "restricted"]
    elif current_user.role == 'faculty':
        allowed = ["public", "internal"]
    else:
        allowed = ["public"]

    try:
        # 2. Query the vector engine (Returns answer string and raw citation string)
        answer, citation_raw = rag.search_and_generate(question, allowed)

        # 3. Parse the raw string citation into a structured dictionary layout
        citation_data = None
        filename = None
        page_number = 1

        if citation_raw and isinstance(citation_raw, str):
            # Matches patterns like "Document_Name.pdf (Page X)" case-insensitively
            match = re.search(r"(.*?)\s*\(Page\s*(\d+)\)", citation_raw, re.IGNORECASE)
            if match:
                filename = match.group(1).strip()
                page_number = int(match.group(2))
                citation_data = {"filename": filename, "page": page_number}
            else:
                # Fallback safeguard: if no page structure matches, treat the whole string as the filename
                filename = citation_raw.strip()
                citation_data = {"filename": filename, "page": 1}

        # 4. Format and save the clean bot response string text
        bot_msg = ChatMessage(user_id=current_user.id, sender='bot', message=answer)
        db.session.add(bot_msg)
        db.session.commit()

        # 5. Map the relational citation entry structurally using explicit foreign keys
        if citation_data and filename:
            linked_doc = Document.query.filter_by(filename=filename).first()

            new_citation = Citation(
                message_id=bot_msg.id,
                document_id=linked_doc.id if linked_doc else None,
                page_number=page_number
            )
            db.session.add(new_citation)
            db.session.commit()

        # Return the clean text and the structured dictionary context back to index.html
        return jsonify({"answer": answer, "citation": citation_data})

    except Exception as e:
        #  Structural Guardrail: If AI or vector searches fail, flush memory state cleanly
        db.session.rollback()
        print(f"[!] Engine Error encountered during runtime execution: {str(e)}")
        return jsonify({
            "answer": "The core AI engine encountered a processing error. Please try again.",
            "citation": None
        }), 500