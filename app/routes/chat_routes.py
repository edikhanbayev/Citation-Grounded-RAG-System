from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from app import db
from app.models import ChatMessage, RagEngine, Citation, Document

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
    db.session.commit()  # 🚀 Safe state separation

    # Resolve clearance array parameters based on user access tokens
    if current_user.role == 'admin':
        allowed = ["public", "internal", "restricted"]
    elif current_user.role == 'faculty':
        allowed = ["public", "internal"]
    else:
        allowed = ["public"]

    try:
        # 2. Query the vector engine (Now returns an answer string and a structured LIST of citations)
        answer, citations_list = rag.search_and_generate(question, allowed)

        # 3. Refusal Detection Guardrail
        # If the LLM indicates it lacks context, suppress all vector citations
        refusal_indicators = [
            "cannot locate", "not found", "insufficient information",
            "don't know", "do not know", "no context", "does not mention",
            "provided context does not"
        ]

        # Scans the LLM output for any phrases indicating a lookup failure
        has_failed_to_find = any(phrase in answer.lower() for phrase in refusal_indicators)
        if has_failed_to_find:
            citations_list = []

        # 4. Format and save the clean bot response string text
        bot_msg = ChatMessage(user_id=current_user.id, sender='bot', message=answer)
        db.session.add(bot_msg)
        db.session.commit()

        # 🚀 5. NEW: 1:N Relational Loop Mapping
        # Processes every distinct matching source chunk and maps them to SQLite
        for cit in citations_list:
            filename = cit.get('filename')
            page_number = cit.get('page', 1)

            if filename:
                linked_doc = Document.query.filter_by(filename=filename).first()

                new_citation = Citation(
                    message_id=bot_msg.id,
                    document_id=linked_doc.id if linked_doc else None,
                    page_number=page_number
                )
                db.session.add(new_citation)

        # Commit all relational citation references together safely
        db.session.commit()

        # Return the clean text and the structured list back to your AJAX frontend
        return jsonify({"answer": answer, "citations": citations_list})

    except Exception as e:
        # Structural Guardrail: If AI or vector searches fail, flush memory state cleanly
        db.session.rollback()
        print(f"[!] Engine Error encountered during runtime execution: {str(e)}")
        return jsonify({
            "answer": "The core AI engine encountered a processing error. Please try again.",
            "citations": []
        }), 500