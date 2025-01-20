import sys
import os

# Add your virtual environment's site-packages
VENV_PATH = os.path.expanduser('/home/curiouco/virtualenv/resume-analyzer/resume-analyzer-backend-master/3.9/lib/python3.9/site-packages')
sys.path.insert(0, VENV_PATH)

# Add your application directory to Python path
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

from app import app as application 