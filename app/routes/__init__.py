"""
Routing Package Aggregation Interface (Facade Pattern).
Bundles individual context sub-modules into a unified API surface area
for the core app initializer to ingest in a single line.
"""

from .auth_routes import auth_bp
from .admin_routes import admin_bp
from .chat_routes import chat_bp