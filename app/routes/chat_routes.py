from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from app import db
from app.models import ChatMessage, RagEngine

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

    user_msg = ChatMessage(user_id=current_user.id, sender='user', message=question)
    db.session.add(user_msg)

    # Resolve clearance array parameters
    if current_user.role == 'admin':
        allowed = ["public", "internal", "restricted"]
    elif current_user.role == 'faculty':
        allowed = ["public", "internal"]
    else:
        allowed = ["public"]

    answer, citation = rag.search_and_generate(question, allowed)

    # Format data for DB message logs cleanly
    bot_msg = ChatMessage(user_id=current_user.id, sender='bot', message=f"{answer} | Source: {citation}")
    db.session.add(bot_msg)
    db.session.commit()

    return jsonify({"answer": answer, "citation": citation})