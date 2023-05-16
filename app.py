from flask import Flask,render_template,url_for, request

from src.initial_population import *

import sys

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('pruebaPlotly.html')

@app.route('/population', methods=['GET', 'POST'])
def population():
    print("population", file=sys.stderr)
    if request.method == 'POST':
        form = request.form
        population_routes=population_form(form)
    return render_template('pruebaPlotly.html',population=population_routes)
    
if __name__ == "__main__":
    app.run(debug=True)