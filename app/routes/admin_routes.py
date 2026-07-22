import os
from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, send_from_directory, abort, \
    current_app
from flask_login import login_required, current_user
from app import db
from app.models import User, RagEngine
from app.forms import DocumentUploadForm
from app.services.document_service import process_async_upload, delete_document_source

admin_bp = Blueprint('admin', __name__)
rag = RagEngine()

@admin_bp.route('/admin_panel', methods=['GET', 'POST'])
@login_required
def admin_panel():
    if current_user.role != 'admin':
        flash('Access Denied. Resource restricted to administrators.')
        return redirect(url_for('chat.index'))

    upload_form = DocumentUploadForm()
    pending = User.query.filter_by(is_approved=False).all()

    if upload_form.validate_on_submit():
        # Step 1: Invoke our decoupled background processor service
        filename = process_async_upload(upload_form.document.data, upload_form.clearance.data)

        # STEP 2: Invalidate Layer 1 Semantic Cache & Refresh Layer 2 RAM BM25 Buffer
        rag.clear_all_caches()

        flash(f"Success! '{filename}' uploaded safely. Parsing and vector indexing are running in the background.")
        return redirect(url_for('admin.admin_panel'))

    return render_template('admin_panel.html', pending=pending, upload_form=upload_form)

@admin_bp.route('/view-source/<path:filename>', methods=['GET'])
@login_required
def view_source(filename):
    """Safely streams uploaded local raw PDF data paths to authenticated sessions."""
    try:
        upload_dir = os.path.join(current_app.root_path, '..', 'uploads')
        return send_from_directory(upload_dir, filename)
    except FileNotFoundError:
        abort(404)

@admin_bp.route('/approve/<int:user_id>')
@login_required
def approve_user(user_id):
    if current_user.role != 'admin':
        return "Forbidden", 403
    target_user = db.session.get(User, user_id)
    if target_user:
        target_user.is_approved = True
        db.session.commit()
        flash(f"Account for {target_user.username} approved.")
    return redirect(url_for('admin.admin_panel'))

@admin_bp.route('/admin/manage-sources', methods=['GET'])
@login_required
def manage_sources():
    if current_user.role != 'admin':
        flash("Access Denied.")
        return redirect(url_for('chat.index'))
    sources = rag.get_all_sources()
    return render_template('manage_sources.html', sources=sources)

@admin_bp.route('/admin/delete-source', methods=['POST'])
@login_required
def delete_source_endpoint():
    if current_user.role != 'admin':
        return jsonify({"error": "Unauthorized Access Blocked."}), 403

    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({"error": "No filename specified."}), 400

    # Step 1: Call our isolated business service layer
    success, message = delete_document_source(filename)
    if success:
        #  STEP 2: Flush all cache layers to purge stale citations and vectors
        rag.clear_all_caches()

        return jsonify({"success": True, "message": message})
    return jsonify({"error": message}), 500