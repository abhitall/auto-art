# Initializes the AutoART API package
from flask import Blueprint

# This __init__.py can be used to define a Blueprint for API routes
# For now, it's simple, but could be expanded later.
# Example:
# api_blueprint = Blueprint('api', __name__, url_prefix='/api/v1')
# from . import app # Import routes from app.py or other route files if they use this blueprint

# For a very simple structure where app.py defines all routes directly on app object,
# this __init__.py might just be empty or have a docstring.
# Adding a docstring for now.
"""
AutoART API Package
-------------------
This package contains the REST API for the AutoART framework.
The main Flask application and route definitions are typically found in `app.py`.
"""
