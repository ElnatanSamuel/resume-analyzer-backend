import os
import sys

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.app import app

# This is the application variable that Gunicorn expects
application = app 