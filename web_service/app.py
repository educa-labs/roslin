# Flask Libs
from flask import Flask
from flask import request, Response, send_from_directory, render_template, request
from flask_cors import CORS, cross_origin
# System libs
import json
# Other libs
app = Flask(__name__)
CORS(app)

@app.route("/api/v1/recommendations", methods = ["POST"])
def recommendations():
    params = request.get_json()
    print(params)

    # Dummmy response
    dummy = json.loads("""{
                "data": {
                    "type": "recommendations",
                    "id": "1",
                    "attributes": {
                    "project_metas": [
                        {
                        "id": 1929,
                        "score": 67.1
                        },
                        {
                        "id": 8988,
                        "score": 57.7
                        }
                    ],
                    "reflection": {
                        "heavier_tags": ["Color:Blanco","Volumetría:Basal"]
                        }
                    }   
                }
            }
    """)


    resp = Response(json.dumps(dummy), status=200, mimetype='application/json')
    return resp


@app.route("/api/v1/projectmetas", methods = ["POST"])
def project_loading():
    params = request.get_json()
    print(params)

    # Dummmy response

    resp = Response(json.dumps(params), status=200, mimetype='application/json')
    return resp





if __name__ ==  '__main__':
    app.run()