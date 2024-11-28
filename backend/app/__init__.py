from flask import Flask

app = Flask(__name__)

# Import views after app is defined
from app import views

# Use views to avoid the "not accessed" error
app.register_blueprint(views.blueprint)
