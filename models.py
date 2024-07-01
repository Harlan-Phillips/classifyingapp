from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

# Association table for the many-to-many relationship
user_transient = db.Table('user_transient',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('transient_id', db.Integer, db.ForeignKey('transient.id'), primary_key=True),
    db.Column('classification', db.String(64)),
    db.Column('confidence', db.String(64))
)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    @property
    def is_authenticated(self):
        return True

    @property
    def is_active(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)

# Each Transient tied to classifications
class Transient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.String(64), unique=True, nullable=False)
    classified_by = db.relationship('Classification', backref='transient', lazy=True)

# Each Classification tied to a user and a transient
class Classification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.String(50), db.ForeignKey('transient.source_id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    classification = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)