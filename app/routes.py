from flask import render_template, redirect, url_for, flash, request, jsonify
import sqlalchemy as sa
from flask_login import login_user, current_user, login_required, logout_user
from urllib.parse import urlsplit
import os
from werkzeug.utils import secure_filename
from app import app, db
from app.models import Student, ChatMessage, RagEngine
from app.forms import LoginForm, RegistrationForm, DocumentUploadForm

rag = RagEngine()


@app.route('/', methods=['GET'])
@login_required
def index():
    history = ChatMessage.query.filter_by(user_id=current_user.id).order_by(ChatMessage.timestamp.asc()).all()
    return render_template('index.html', history=history)


@app.route('/ask', methods=['POST'])
@login_required
def ask():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({"error": "Empty payload"}), 400

    user_msg = ChatMessage(user_id=current_user.id, sender='user', message=question)
    db.session.add(user_msg)

    if current_user.role == 'admin':
        allowed = ["public", "internal", "restricted"]
    elif current_user.role == 'faculty':
        allowed = ["public", "internal"]
    else:
        allowed = ["public"]

    answer, citation = rag.search_and_generate(question, allowed)

    bot_msg = ChatMessage(user_id=current_user.id, sender='bot', message=f"{answer} (Source: {citation})")
    db.session.add(bot_msg)
    db.session.commit()

    return jsonify({"answer": answer, "citation": citation})


#  Handles form post transactions and processes text chunks
@app.route('/admin_panel', methods=['GET', 'POST'])
@login_required
def admin_panel():
    if current_user.role != 'admin':
        flash('Access Denied. Resource restricted to administrators.')
        return redirect(url_for('index'))

    upload_form = DocumentUploadForm()
    pending = Student.query.filter_by(is_approved=False).all()

    if upload_form.validate_on_submit():
        file = upload_form.document.data
        filename = secure_filename(file.filename)

        # Build local uploads directory trail safely
        upload_dir = os.path.join(app.root_path, '..', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        saved_path = os.path.join(upload_dir, filename)

        # 1. Write the file data array onto your local machine disk path
        file.save(saved_path)

        # 2. Extract texts using pypdf and index chunks right into ChromaDB
        rag.process_and_index_pdf(saved_path, clearance_level=upload_form.clearance.data)

        flash(f"Success! Document '{filename}' has been processed into the Vector DB.")
        return redirect(url_for('admin_panel'))

    return render_template('admin_panel.html', pending=pending, upload_form=upload_form)


@app.route('/approve/<int:user_id>')
@login_required
def approve_user(user_id):
    if current_user.role != 'admin':
        return "Forbidden", 403
    target_user = db.session.get(Student, user_id)
    if target_user:
        target_user.is_approved = True
        db.session.commit()
        flash(f"Account for {target_user.username} approved.")
    return redirect(url_for('admin_panel'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = db.session.scalar(sa.select(Student).where(Student.username == form.username.data))
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))

        if not user.is_approved:
            flash('Your account is currently pending administrator verification approval.')
            return redirect(url_for('login'))

        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or urlsplit(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        is_auto_approved = True if form.role.data == 'student' else False

        user = Student(
            username=form.username.data,
            email=form.email.data,
            student_id=form.student_id.data,
            role=form.role.data,
            is_approved=is_auto_approved
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()

        if not is_auto_approved:
            flash('Registration successful. Awaiting administrator approval.')
            return redirect(url_for('login'))

        flash('Registration completed successfully!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/admin/manage-sources', methods=['GET'])
@login_required
def manage_sources():
    """Renders the dashboard menu showing all active files inside ChromaDB."""
    if current_user.role != 'admin':
        flash("Access Denied. Resource restricted to administrators.")
        return redirect(url_for('index'))

    # Fetches unique filenames directly out of ChromaDB metadata
    sources = rag.get_all_sources()
    return render_template('manage_sources.html', sources=sources)


@app.route('/admin/delete-source', methods=['POST'])
@login_required
def delete_source_endpoint():
    """Handles real-time asynchronous data purge requests from the front-end."""
    if current_user.role != 'admin':
        return jsonify({"error": "Unauthorized endpoint access block."}), 403

    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({"error": "No filename specified for deletion."}), 400

    # 1. Purge all matching vector chunks from ChromaDB collection
    vector_success = rag.delete_source(filename)

    # 2. 🚀 NEW: Physically delete the PDF from the 'uploads' directory on disk
    if vector_success:
        try:
            # Reconstruct the exact absolute path matching your upload logic
            upload_dir = os.path.join(app.root_path, '..', 'uploads')
            file_path = os.path.join(upload_dir, filename)

            # Check if the file physically exists on the operating system before deleting it
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Disk Cleanup Success: Removed {file_path}")
            else:
                print(f"Disk Cleanup Warning: File not found on disk path {file_path}")

        except Exception as e:
            # Use a non-blocking try-except block so an OS file lock doesn't halt the whole route
            print(f"Disk Cleanup Failure: Failed to remove physical file. Detail: {str(e)}")

        return jsonify({
            "success": True,
            "message": f"'{filename}' has been completely expunged from ChromaDB and the uploads folder."
        })
    else:
        return jsonify({"error": "Failed to fully clear vector fragments from memory."}), 500