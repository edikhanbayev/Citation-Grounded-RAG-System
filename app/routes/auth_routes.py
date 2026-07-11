from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, current_user, logout_user
import sqlalchemy as sa
from urllib.parse import urlsplit
from app import db
from app.models import Student
from app.forms import LoginForm, RegistrationForm

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('chat.index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = db.session.scalar(sa.select(Student).where(Student.username == form.username.data))
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('auth.login'))

        if not user.is_approved:
            flash('Your account is currently pending administrator verification approval.')
            return redirect(url_for('auth.login'))

        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or urlsplit(next_page).netloc != '':
            next_page = url_for('chat.index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)

@auth_bp.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('chat.index'))

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('chat.index'))
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
            return redirect(url_for('auth.login'))

        flash('Registration completed successfully!')
        return redirect(url_for('auth.login'))
    return render_template('register.html', title='Register', form=form)