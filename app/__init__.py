from config import Config

import re
from urllib.parse import quote
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager

# 1. Initialize Global Application Context
app = Flask(__name__)

# Load configurations cleanly from standalone config.py file
app.config.from_object(Config)

# 2. Bind Core Application Extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login = LoginManager(app)

# Configure authentication security blocks
login.login_view = 'auth.login'
login.login_message_category = 'info'


# 3. User Loader Callback for Session Tracking
@login.user_loader
def load_user(user_id):
    from app.models import User
    return db.session.get(User, int(user_id))


# 4. Architectural Blueprint Ingestion
from app.routes import auth_bp, admin_bp, chat_bp

app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(chat_bp)


# # 5. Global Jinja2 Template Filters
# @app.template_filter('render_citations')
# def render_citations(text):
#     """
#     Server-side template parser. Inspects database message strings for citations
#     and normalizes them into safe, clickable anchor links for the UI layout history.
#     """
#     if not text:
#         return ""
#
#     # Matches patterns matching the format: "| Source: Filename.pdf (Page X)"
#     pattern = r"\|\s*Source:\s*(.*?)\s*\(Page\s*(\d+)\)"
#     match = re.search(pattern, text)
#
#     if match:
#         filename = match.group(1).strip()
#         page = match.group(2)
#         encoded_filename = quote(filename)
#
#         # Build standard production HTML target links pointing to our blueprint view endpoint
#         link_html = (
#             f'| Source: <a href="/view-source/{encoded_filename}#page={page}" '
#             f'target="_blank" style="color: #007bff; text-decoration: underline; '
#             f'font-weight: 500;">{filename} (Page {page}) ↗️</a>'
#         )
#
#         # Swaps raw matching text patterns out for our live HTML link elements
#         return re.sub(pattern, link_html, text)
#
#     return text