# Flask Libs
from flask import Flask
from flask import request, Response, send_from_directory, render_template, request
from flask_cors import CORS, cross_origin
# System libs
import json
# Other libs
app = Flask(__name__)
CORS(app)

@app.route("/ruta", methods = ["GET", "POST"])
def ruta():
    if request.method == "GET":
        result = 'GET request is not allowed'
        status = 501

    elif request.method == "POST":
        status = 200

    resp = Response(json.dumps(result), status=status, mimetype='application/json')
    return resp

if __name__ ==  '__main__':
    app.run()