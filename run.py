# run.py
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from app.views import app
from backend.app.views import app


if __name__ == "__main__":
    app.run(debug=True)
