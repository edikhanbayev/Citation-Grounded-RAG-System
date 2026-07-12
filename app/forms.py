from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField, BooleanField, SelectField
from wtforms.validators import DataRequired, Email, EqualTo, ValidationError
from flask_wtf.file import FileField, FileRequired, FileAllowed
import sqlalchemy as sa
from app.models import User
from app import db

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    student_id = StringField('Student/Staff ID', validators=[DataRequired()])
    role = SelectField('Account Type', choices=[
        ('student', 'Student (Instant Access)'),
        ('faculty', 'Faculty Staff (Requires Admin Approval)'),
        ('admin', 'Administrator (Requires Admin Approval)')
    ], validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    password2 = PasswordField('Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = db.session.scalar(sa.select(User).where(User.username == username.data))
        if user is not None:
            raise ValidationError('Please use a different username.')

    def validate_email(self, email):
        user = db.session.scalar(sa.select(User).where(User.email == email.data))
        if user is not None:
            raise ValidationError('Please use a different email address.')

    def validate_student_id(self, student_id):
        existing_id = db.session.scalar(
            sa.select(User).where(User.student_id == student_id.data)
        )
        if existing_id is not None:
            raise ValidationError('This Student/Staff ID has already been registered.')

class DocumentUploadForm(FlaskForm):
    """Form to handle academic regulation PDF uploads in the Admin Panel."""
    document = FileField('Select Regulation PDF Document', validators=[
        FileRequired(),
        FileAllowed(['pdf'], 'PDF documents only!')
    ])
    clearance = SelectField('Document Access Tier', choices=[
        ('public', 'Public (Accessible by Everyone)'),
        ('internal', 'Internal (Faculty & Admins Only)'),
        ('restricted', 'Restricted (Admins Only)')
    ], validators=[DataRequired()])
    submit = SubmitField('Upload & Parse to Vector Database')