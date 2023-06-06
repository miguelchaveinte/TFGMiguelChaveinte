from flask import Flask,render_template,url_for, request

from src.initial_population import *

from src.algorithmNSGA2 import algorithm_form

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


@app.route('/runAlgorithm', methods=['GET', 'POST'])
def runAlgorithm():
    print('ALGORITMO', file=sys.stderr)
    if request.method == 'POST':
        form = request.form
        routes_display,gridTime,ds_lat,ds_lon=algorithm_form(form)

    return render_template('pruebaPlotly.html',population=routes_display,gridTime=gridTime[2].reshape(-1).tolist(),ds_lat=ds_lat,ds_lon=ds_lon)
    
if __name__ == "__main__":
    app.run(debug=True)