import os
import json
import csv
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd
from astropy.coordinates import SkyCoord
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from flask_paginate import Pagination, get_page_parameter, get_page_args
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo

# Imports from local files
from models import db, User, Transient, Classification
from utils import (
    get_pos, get_galactic, get_lc, logon, plot_ztf_cutout, 
    plot_ps1_cutout, plot_ls_cutout, plot_light_curve, 
    xmatch_ls, get_dets
)
from vlass_utils import get_vlass_data, run_search


# Initialize the Kowalski session
kowalski_session = logon()

basedir = os.path.abspath(os.path.dirname(__file__))

# Create flask app instance
class_app = Flask(__name__)
class_app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///' + os.path.join(basedir, 'class_app.db')
class_app.config['SECRET_KEY'] = 'your_secret_key_here'

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
    
    flash('Your classification has been recorded.', 'success')
    return redirect(url_for('classify_source', source_id=source_id))

@class_app.route('/classify/<source_id>', methods=['GET'])
@login_required
def classify_source(source_id):
    """Render the classification page for a given source."""   
    try:
        # Fetch positional and galactic data        
        ra, dec, scat_sep = get_pos(kowalski_session, source_id)
        galactic_lat = get_galactic(ra, dec)
        
        cutout_dir = os.path.join(basedir, 'static')
        light_curve_path = os.path.join(cutout_dir, f"{source_id}_light_curve.png")
        ztf_cutout_path = os.path.join(cutout_dir, f"{source_id}_triplet.png")
        ps1_cutout_path = os.path.join(cutout_dir, f"{source_id}_ps1.png")
        ls_cutout_path = os.path.join(cutout_dir, f"{source_id}_ls.png")

        # Checking if cutouts exist reducing processing time
        if os.path.exists(light_curve_path):
            plot_filename = f"static/{source_id}_light_curve.png"
        else:
            light_curve = get_lc(kowalski_session, source_id)
            plot_filename = plot_light_curve(light_curve, source_id)

        if os.path.exists(ztf_cutout_path):
            ztf_cutout_basename = f"{source_id}_triplet.png"
        else:
            ztf_cutout = plot_ztf_cutout(kowalski_session, cutout_dir, source_id)
            ztf_cutout_basename = os.path.basename(ztf_cutout) if ztf_cutout else None

        if os.path.exists(ps1_cutout_path):
            ps1_cutout_basename = f"{source_id}_ps1.png"
        else:
            ps1_cutout = plot_ps1_cutout(kowalski_session, cutout_dir, source_id, ra, dec)
            ps1_cutout_basename = os.path.basename(ps1_cutout) if ps1_cutout else None

        if os.path.exists(ls_cutout_path):
            ls_cutout_basename = f"{source_id}_ls.png"
        else:
            ls_cutout = plot_ls_cutout(kowalski_session, cutout_dir, source_id, ra, dec)
            ls_cutout_basename = os.path.basename(ls_cutout) if ls_cutout else None
    
        legacy_survey_data = xmatch_ls(ra, dec) # Fetch Legacy Survey data
     
        dets = get_dets(kowalski_session, source_id) # Extract data from get_dets
        
        # Determine if there is SDSS data
        sdss_data = None
        if dets[0]['candidate']['ssdistnr'] != -999.0 and dets[0]['candidate']['ssmagnr'] != -999.0:
            sdss_data = {
                'ssdistnr': dets[0]['candidate']['ssdistnr'],
                'ssmagnr': dets[0]['candidate']['ssmagnr']
            }

        # Aggregate Pan-STARRS data and remove duplicates
        pan_starrs_data = []
        seen_sgscore1 = set()
        for det in dets:
            candidate = det['candidate']
            if 'distpsnr1' in candidate and candidate['distpsnr1'] != -999.0 and 'sgscore1' in candidate and candidate['sgscore1'] != -999.0:
                if candidate['sgscore1'] not in seen_sgscore1:
                    pan_starrs_data.append({
                        'distpsnr1': candidate['distpsnr1'],
                        'sgscore1': candidate['sgscore1']
                    })
                    seen_sgscore1.add(candidate['sgscore1'])
        pan_starrs_df = pd.DataFrame(pan_starrs_data)
        
        vlass_images = session.pop('vlass_images', []) # Retrieve VLASS images

    except ValueError as e:
        flash('Source does not exist or data could not be retrieved.')
        return redirect(url_for('index'))

    # Retrieve classifications and determine the most confident classification
    classifications = Classification.query.filter_by(source_id=source_id).all()
    classification_counts = defaultdict(lambda: {'count': 0, 'confidence': 0})
    classified_by_users = []

    for classification in classifications:
        classification_counts[classification.classification]['count'] += 1
        classified_by_users.append(User.query.get(classification.user_id).username)
        if classification.confidence == 'Not confident':
            classification_counts[classification.classification]['confidence'] += 1
        elif classification.confidence == 'Confident':
            classification_counts[classification.classification]['confidence'] += 2
        elif classification.confidence == 'Certain':
            classification_counts[classification.classification]['confidence'] += 3

    if classification_counts:
        most_confident_classification = max(
            classification_counts.items(),
            key=lambda x: (x[1]['confidence'], x[1]['count'])
        )[0]
    else:
        most_confident_classification = None

    return render_template('classify.html', 
                            source_id=source_id, 
                            ra=ra, 
                            dec=dec, 
                            scat_sep=scat_sep, 
                            galactic_lat=galactic_lat, 
                            light_curve=light_curve.to_dict(orient='records') if 'light_curve' in locals() else None,
                            plot_filename=plot_filename,
                            ztf_cutout=ztf_cutout_basename,
                            ps1_cutout=ps1_cutout_basename,
                            ls_cutout=ls_cutout_basename,
                            classifications=classifications,
                           most_confident_classification=most_confident_classification,
                           classified_by_users=classified_by_users,
                            vlass_images=vlass_images,
                           legacy_survey_data=legacy_survey_data,
                           sdss_data=sdss_data,
                            pan_starrs_df=pan_starrs_df
                           )
    
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

if __name__ == '__main__':
    # Initialize databases and load transients from csv
    with class_app.app_context():
        db.create_all()
        load_transients()
    class_app.run(debug=True)