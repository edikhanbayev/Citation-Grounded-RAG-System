import os
import requests
from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from app import db
from app.models import ChatMessage, RagEngine, Citation, Document, Conversation

chat_bp = Blueprint('chat', __name__)
rag = RagEngine()


def generate_chat_title(first_question, api_key):
    """Generates a brief 3-to-5 word title summarizing the user's initial query."""
    prompt = (
        "You are an automated UI title generator. Summarize the following user question "
        "into a brief, descriptive chat title (maximum of 5 words). "
        "Do NOT use quotes, punctuation, or wrapping text. Output ONLY the title.\n\n"
        f"Question: {first_question}"
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=5
        )
        if res.status_code == 200:
            return res.json()['choices'][0]['message']['content'].strip()
    except Exception:
        pass
    return "New Conversation"


@chat_bp.route('/', methods=['GET'])
@login_required
def index():
    """Renders the workspace. Thread collections are populated asynchronously via AJAX."""
    return render_template('index.html')


@chat_bp.route('/conversations', methods=['GET'])
@login_required
def get_conversations():
    """Fetches all past conversations for the authenticated user."""
    convs = Conversation.query.filter_by(user_id=current_user.id).order_by(Conversation.created_at.desc()).all()
    return jsonify([{
        "id": c.id,
        "title": c.title,
        "created_at": c.created_at.isoformat()
    } for c in convs])


@chat_bp.route('/conversation/<conv_id>', methods=['GET'])
@login_required
def get_conversation_history(conv_id):
    """Loads past history and mapped citations for a specific memory context container."""
    conv = Conversation.query.filter_by(id=conv_id, user_id=current_user.id).first_or_404()
    msgs = ChatMessage.query.filter_by(conversation_id=conv_id).order_by(ChatMessage.timestamp.asc()).all()

    history_payload = []
    for m in msgs:
        cits = []
        for c in m.citations.all():
            cits.append({
                "filename": c.document.filename if c.document else "[Document Archive Purged]",
                "page": c.page_number
            })

        history_payload.append({
            "sender": m.sender,
            "message": m.message,
            "citations": cits
        })
    return jsonify(history_payload)


@chat_bp.route('/ask', methods=['POST'])
@login_required
def ask():
    """Ingests multi-turn conversational sequences and returns targeted search matches."""
    data = request.get_json() or {}
    question = data.get('question', '').strip()
    conv_id = data.get('conversation_id')

    if not question:
        return jsonify({"error": "Empty question text field"}), 400

    api_key = os.environ.get('GROQ_API_KEY')
    is_new_thread = False
    active_conv = None

    # Step 1: Thread validation
    if conv_id:
        active_conv = Conversation.query.filter_by(id=conv_id, user_id=current_user.id).first()

    if not active_conv:
        is_new_thread = True
        title_summary = generate_chat_title(question, api_key)
        active_conv = Conversation(title=title_summary, user_id=current_user.id)
        db.session.add(active_conv)
        db.session.commit()  # Save first to establish UUID

    # Step 2: Retrieve history scoped strictly to this thread for context memory
    past_messages = ChatMessage.query.filter_by(conversation_id=active_conv.id).order_by(
        ChatMessage.timestamp.asc()).all()
    chat_history_list = []
    for m in past_messages:
        if m.sender == 'user':
            chat_history_list.append({"user": m.message, "bot": ""})
        elif m.sender == 'bot' and chat_history_list:
            chat_history_list[-1]["bot"] = m.message

    # Step 3: Resolve clearance levels
    if current_user.role == 'admin':
        allowed = ["public", "internal", "restricted"]
    elif current_user.role == 'faculty':
        allowed = ["public", "internal"]
    else:
        allowed = ["public"]

    try:
        # Step 4: Run search and generate using current thread memory
        answer, citations_list = rag.search_and_generate(question, allowed, chat_history=chat_history_list)

        # Refusal Detection Guardrail
        refusal_indicators = [
            "cannot locate", "not found", "insufficient information",
            "don't know", "do not know", "no context", "does not mention",
            "provided context does not"
        ]
        has_failed_to_find = any(phrase in answer.lower() for phrase in refusal_indicators)
        if has_failed_to_find:
            citations_list = []

        # Step 5: Save exchange logs
        user_msg = ChatMessage(conversation_id=active_conv.id, user_id=current_user.id, sender='user', message=question)
        bot_msg = ChatMessage(conversation_id=active_conv.id, user_id=current_user.id, sender='bot', message=answer)
        db.session.add(user_msg)
        db.session.add(bot_msg)
        db.session.commit()  # Generates IDs for messaging assets

        # Step 6: Map citation relationships with Self-Healing validation
        for cit in citations_list:
            filename = cit.get('filename')
            page_number = cit.get('page', 1)

            if filename:
                linked_doc = Document.query.filter_by(filename=filename).first()

                # 🔥 SELF-HEALING: Document in ChromaDB but missing from SQL table
                if not linked_doc:
                    linked_doc = Document(
                        filename=filename,
                        clearance_level="public",  # Safe baseline clearance for dynamic records
                        status="Ready"
                    )
                    db.session.add(linked_doc)
                    db.session.flush()  # Instantly writes to SQL transaction state to fetch a valid ID

                new_citation = Citation(
                    message_id=bot_msg.id,
                    document_id=linked_doc.id,
                    page_number=page_number
                )
                db.session.add(new_citation)

        db.session.commit()

        return jsonify({
            "answer": answer,
            "citations": citations_list,
            "conversation_id": active_conv.id,
            "conversation_title": active_conv.title if is_new_thread else None
        })

    except Exception as e:
        db.session.rollback()
        print(f"[!] Engine Error encountered during runtime execution: {str(e)}")
        return jsonify({
            "answer": "The core AI engine encountered a processing error. Please try again.",
            "citations": []
        }), 500