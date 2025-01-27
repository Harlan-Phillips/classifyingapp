import os
import json
import csv
import random
from collections import Counter, defaultdict
from datetime import datetime
from io import BytesIO
import time
import logging
from celery import shared_task

import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, session, send_file, make_response
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from flask_paginate import Pagination, get_page_parameter, get_page_args
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo

# Imports from local files
from models import db, User, Transient, Classification
from utils import (
    get_pos, get_galactic, get_lc, logon,  
    plot_ps1_cutout, plot_ls_cutout, plot_light_curve, 
    xmatch_ls, get_dets, plot_polar_coordinates, get_most_confident_classification, 
    plot_big_light_curve, plot_big_polar_coordinates, 
    analyze_ps1_photoz, get_drb, get_span, plot_wise, filter_and_plot_alerts, alert_table,
    get_ecliptic, make_celery, fetch_transient_data
)
from vlass_utils import get_vlass_data, run_search

from threading import Thread
from cachetools import TTLCache

# Initialize the Kowalski session
kowalski_session = logon()

basedir = os.path.abspath(os.path.dirname(__file__))

# Setup logging for debugging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[
                        logging.FileHandler("debug.log"),
                        logging.StreamHandler()
                    ])
# Create flask app instance
class_app = Flask(__name__)
class_app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///' + os.path.join(basedir, 'class_app.db')
class_app.config['SECRET_KEY'] = 'your_secret_key_here'

# Create Celery instance for background info fetching
class_app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0'
)

celery = make_celery(class_app)

# Create a cache to store prefetched transient data
transient_cache = TTLCache(maxsize=10, ttl=600)

# Initialize CSRF Protection
class_app.config['SECRET_KEY'] = 'your_secret_key_here'
class_app.config['WTF_CSRF_ENABLED'] = False 

csrf = CSRFProtect(class_app)

# Initializing database, and login manager with Flask 
db.init_app(class_app)
login_manager = LoginManager()
login_manager.init_app(class_app)
login_manager.login_view = 'login'

# Define forms for search, registration, and login
class SearchForm(FlaskForm):
    source_id = StringField('Source ID', validators=[DataRequired()])
    submit = SubmitField('Fetch Data')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), EqualTo('confirm', message='Passwords must match')])
    confirm = PasswordField('Repeat Password')
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

@login_manager.user_loader
def load_user(user_id):
    """Load user by ID."""
    return User.query.get(int(user_id))

@class_app.context_processor
def inject_search_form():
    """Inject the search form into the context of all templates."""
    return dict(search_form=SearchForm())

@class_app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration"""
    form = RegistrationForm()
    if form.validate_on_submit():
        existing_user_username = User.query.filter_by(username=form.username.data).first()
        existing_user_email = User.query.filter_by(email=form.email.data).first()
        
        if existing_user_username:
            flash('Username already exists. Please choose a different username.')
            return redirect(url_for('register'))
        
        if existing_user_email:
            flash('Email already registered. Please use a different email address.')
            return redirect(url_for('register'))
        
        username = form.username.data
        email = form.email.data
        password = form.password.data
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@class_app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid email or password')
            return redirect(url_for('login'))
        login_user(user, remember=True)
        return redirect(url_for('index'))
    return render_template('login.html', form=form)

@class_app.route('/logout')
@login_required
def logout():
    """Logout current user."""
    logout_user()
    return redirect(url_for('index'))



@class_app.route('/', methods=['GET', 'POST'])
def index():
    """Render the main search form and handle search requests."""
    form = SearchForm()
    if form.validate_on_submit():
        source_id = form.source_id.data.strip()

        if len(source_id) != 12 or source_id[:3].lower() != 'ztf' or not source_id[3:5].isdigit() or not source_id[5:].isalpha():
            flash('Invalid source name.')
            return redirect(url_for('index'))
        
        try:
            # Attempt to redirect to classify_source, this will invoke classify_source route logic
            return redirect(url_for('classify_source', source_id=source_id))
        except TypeError:
            flash('Source does not exist or data could not be retrieved.')
            return redirect(url_for('index'))
        except Exception as e:
            flash(f'An error occurred: {str(e)}')
            return redirect(url_for('index'))
    return render_template('index.html', form=form)


@class_app.route('/classify/<source_id>', methods=['POST'])
@login_required
def classify(source_id):
    """Handle classification of a source by the current user."""
    classification = request.form.get('classification')
    subtype = request.form.get('subtype', None)  # Use None if not provided
    confidence = request.form.get('confidence')

    # Check for missing fields
    if not classification or not confidence:
        flash('Both classification and confidence are required.')
        return redirect(url_for('classify_source', source_id=source_id))

    # Combine classification and subtype if subtype is provided
    existing_classification = Classification.query.filter_by(source_id=source_id, user_id=current_user.id).first()

    classification_text = classification
    if subtype:
        classification_text += f" {subtype}"

    if existing_classification:
        existing_classification.classification = classification_text
        existing_classification.confidence = confidence
        existing_classification.timestamp = datetime.utcnow()
    else:
        new_classification = Classification(
            source_id=source_id,
            user_id=current_user.id,
            classification=classification_text,
            confidence=confidence,
            timestamp=datetime.utcnow()
        )
        db.session.add(new_classification)

    # Save to database
    db.session.commit()
    
    flash(f'Your classification for {source_id} as "{classification}" has been recorded.', 'success')
    return redirect(url_for('random_transient'))

@class_app.route('/classify/<source_id>', methods=['GET'])
@login_required
def classify_source(source_id):
    """Render the classification page for a given source."""
    try:
        # Fetch the data for the current transient
        data = fetch_transient_data(kowalski_session, source_id)
        if not data:
            flash('An error occurred while fetching the transient data.')
            return redirect(url_for('index'))

        dets = data.get('dets')
        logging.debug(f"dets: {dets}")  
        if dets:
            alerts_raw = alert_table(dets)
            raw_alerts = alerts_raw.to_dict(orient='records') if alerts_raw is not None else []
            logging.debug(f"alerts_raw DataFrame: {alerts_raw}")
            logging.debug(f"raw_alerts: {raw_alerts}")
        else:
            raw_alerts = []

        data['raw_alerts'] = raw_alerts
        data['vlass_images'] = session.pop('vlass_images', [])

        # Render the current transient page
        response = render_template('classify.html', **data)

        # Start prefetching the next transient in a separate thread
        user_id = current_user.get_id()
        thread = Thread(target=prefetch_transient_data, args=(kowalski_session, user_id))
        thread.start()

        return response

    except ValueError as e:
        logging.error(f"ValueError: {e}")
        flash('Source does not exist or data could not be retrieved.')
        return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"Exception: {e}")
        flash(f'An error occurred: {str(e)}')
        return redirect(url_for('index'))

def prefetch_transient_data(kowalski_session, user_id):
    """Prefetch data for the next transient."""
    with class_app.app_context():  # Push application context manually
        try:
            next_source_id = get_random_id()
            prefetched_data = fetch_transient_data(kowalski_session, next_source_id)
            if prefetched_data:
                # Store the prefetched data in the cache instead of session
                transient_cache[user_id] = {
                    'data': prefetched_data,
                    'source_id': next_source_id,
                    'status': 'complete'
                }
        except Exception as e:
            logging.error(f"Error while prefetching transient data: {e}")
            transient_cache[user_id] = {'status': 'error'}

@class_app.route('/prefetch_status', methods=['GET'])
@login_required
def prefetch_status():
    user_id = current_user.get_id()
    status = transient_cache.get(user_id, {}).get('status', 'not_started')
    return jsonify({'status': status})

@class_app.route('/retrieve_vlass_data/<source_id>', methods=['POST'])
@login_required
def retrieve_vlass_data(source_id):
    """Retrieve VLASS data for the given source."""
    kowalski_session = logon()
    ra, dec, scat_sep = get_pos(kowalski_session, source_id)
    cutout_dir = os.path.join(basedir, 'static')

    vlass_images_dir = os.path.join(cutout_dir, 'vlass_images')
    search_images = [f'vlass_images/{file_name}' for file_name in os.listdir(vlass_images_dir) if file_name.startswith(source_id) and file_name.endswith(".png")]

    if not search_images:
        get_vlass_data()
        c = SkyCoord(ra, dec, unit='deg')
        run_search(source_id, c)
        search_images = [f'vlass_images/{file_name}' for file_name in os.listdir(vlass_images_dir) if file_name.startswith(source_id) and file_name.endswith(".png")]

    session['vlass_images'] = search_images
    return redirect(url_for('classify_source', source_id=source_id))

def load_transients():
    """Load transients from a CSV file into the database."""
    with class_app.app_context():
        if not Transient.query.first():  # Only load if the table is empty
            with open('transients.csv', newline='', encoding='utf-8-sig') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    transient = Transient(source_id=row[0])
                    db.session.add(transient)
                db.session.commit()

def load_test_transients_ids():
    """Load source_id values from test_transients.csv"""
    df = pd.read_csv('test_transients.csv')
    return df['source_id'].tolist()

@class_app.route('/transients', methods=['GET'])
@login_required
def list_transients():
    """List all transients with pagination."""
    page, per_page, offset = get_page_args(
        page_parameter='page', 
        per_page=50  
    )

    transients = Transient.query.offset(offset).limit(per_page).all()
    total = Transient.query.count()

    # Fetch classifications and associated users for each transient
    transients_with_classifications = []
    for transient in transients:
        classifications = Classification.query.filter_by(source_id=transient.source_id).all()
        classified_by_users = [User.query.get(classification.user_id).username for classification in classifications] if classifications else []        
        transients_with_classifications.append({
            'transient': transient,
            'classified_by_users': classified_by_users
        })

    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap5')

    return render_template('transients.html', transients_with_classifications=transients_with_classifications, page=page, per_page=per_page, pagination=pagination)

@class_app.route('/test_transients')
def list_test_transients():
    """List transients from test_transients.csv with pagination."""
    test_transients_ids = load_test_transients_ids()
    
    page, per_page, offset = get_page_args(
        page_parameter='page', 
        per_page=50  
    )
    
    # Query only the transients that are in the test_transients.csv
    transients = Transient.query.filter(Transient.source_id.in_(test_transients_ids)).offset(offset).limit(per_page).all()
    total = Transient.query.filter(Transient.source_id.in_(test_transients_ids)).count()
    
    # Fetch classifications and associated users for each transient
    transients_with_classifications = []
    for transient in transients:
        classifications = Classification.query.filter_by(source_id=transient.source_id).all()
        classified_by_users = [User.query.get(classification.user_id).username for classification in classifications] if classifications else []
        transients_with_classifications.append({
            'transient': transient,
            'classified_by_users': classified_by_users
        })
    
    pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap4')
    
    return render_template('test_transients.html', transients_with_classifications=transients_with_classifications, page=page, per_page=per_page, pagination=pagination)


@class_app.route('/export_test_transients', methods=['GET'])
@login_required
def export_test_transients():
    """Export test transients data to Excel."""
    test_transients_ids = load_test_transients_ids()
    
    # Query the transients and their classifications
    transients = Transient.query.filter(Transient.source_id.in_(test_transients_ids)).all()
    
    data = []
    for transient in transients:
        classifications = Classification.query.filter_by(source_id=transient.source_id).all()
        classified_by_users = [User.query.get(classification.user_id).username for classification in classifications]
        most_confident_classification = get_most_confident_classification(classifications)
        
        data.append({
            'source_id': transient.source_id,
            'classified_by': ', '.join(classified_by_users),
            'classification': most_confident_classification
        })

    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create a BytesIO buffer to save the Excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Transients')
    
    # Seek to the beginning of the stream
    output.seek(0)

    return send_file(output, as_attachment=True, download_name='test_transients.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

def get_random_id():
    test_transients_ids = load_test_transients_ids()
    if not test_transients_ids:
        flash('No test transients available.', 'danger')
        return redirect(url_for('index'))
    random_source_id = random.choice(test_transients_ids)

    return random_source_id

@class_app.route('/random_transient', methods=['GET'])
@login_required
def random_transient():
    """Fetch a random transient, using prefetched data if available."""
    user_id = current_user.get_id()

    # Check if we have prefetched data ready in the cache
    cached_transient = transient_cache.pop(user_id, None)
    if cached_transient and cached_transient.get('status') == 'complete':
        # Use the prefetched data
        source_id = cached_transient['source_id']
        data = cached_transient['data']
        
        dets = data.get('dets')
        logging.debug(f"dets: {dets}")  
        if dets:
            alerts_raw = alert_table(dets)
            raw_alerts = alerts_raw.to_dict(orient='records') if alerts_raw is not None else []
            logging.debug(f"alerts_raw DataFrame: {alerts_raw}")
            logging.debug(f"raw_alerts: {raw_alerts}")
        else:
            raw_alerts = []

        data['raw_alerts'] = raw_alerts
        data['vlass_images'] = session.pop('vlass_images', [])
        
        # Start prefetching the next transient in a separate thread
        thread = Thread(target=prefetch_transient_data, args=(kowalski_session, user_id))
        thread.start()

        return render_template('classify.html', **data)
    else:
        # No prefetched data, fetch a new random transient
        new_source_id = get_random_id()
        return redirect(url_for('classify_source', source_id=new_source_id))

@class_app.route('/user_classifications')
@login_required
def user_classifications():
    """Display a table of user's classifications."""
    user_id = current_user.id
    classifications = Classification.query.filter_by(user_id=user_id).all()
    
    return render_template('user_classifications.html', classifications=classifications, userid=user_id)

@class_app.route('/delete_classification/<int:classification_id>', methods=['POST'])
@login_required
def delete_classification(classification_id):
    """Delete a classification by its ID."""
    classification = Classification.query.get_or_404(classification_id)
    
    # Ensure the classification belongs to the current user
    if classification.user_id != current_user.id:
        flash('You are not authorized to delete this classification.', 'danger')
        return redirect(url_for('user_classifications'))
    
    db.session.delete(classification)
    db.session.commit()
    
    flash('Classification deleted successfully.', 'success')
    return redirect(url_for('user_classifications'))

if __name__ == '__main__':
    # Initialize databases and load transients from csv
    with class_app.app_context():
        db.create_all()
        load_transients()
    class_app.run(debug=True)