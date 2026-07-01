from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_mail import Mail

from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

app = Flask(__name__)
app.config.from_object(Config)

# Bind authentication and state controllers
login = LoginManager(app)
login.login_view = 'login'
db = SQLAlchemy(app)

# Defer imports to avoid cyclical dependencies
from app import routes, models