#import sqlite3
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from wtforms import StringField, validators
from flask_wtf import FlaskForm

import os

class URLForm(FlaskForm):
    url = StringField('URL', [validators.DataRequired(message='Field required')])

app = Flask(__name__)
Bootstrap(app)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

templates_path = ''

@app.route('/') # converts the return value into an HTTP response to be displayed by an HTTP client
def index():
    form = URLForm()
    return render_template(templates_path+'index.html', form=form)

@app.route('/', methods=['POST'])
def handle_data():
    form = URLForm()
    url = request.form.get('url',0)
    from ...backend import pipeline
    results = pipeline.start(url)
    return render_template(templates_path+'results.html', likelihood=results['likelihood'])