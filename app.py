# Flask Libs
from flask import Flask
from flask import request, Response, send_from_directory, render_template, request
from flask_cors import CORS, cross_origin
# System libs
import json
# Other libs
from sklearn.pipeline import Pipeline

from transformers.trees import BallTreePredictor
from transformers.json_transformers import JsonToTagsTransform
from transformers.embedders import LdaTransformer
from utils.json_utils import get_by_id, NumpyEncoder


#Constants
DATA_PATH= "data.json"


# Data and initial pipe training
pipe_components =[("json",JsonToTagsTransform()),("embedder",LdaTransformer()),("tree",BallTreePredictor())] 
pipe = Pipeline(pipe_components)
with open(DATA_PATH) as file:
    data =  json.load(file)
pipe.fit(data)

app = Flask(__name__)
CORS(app)

@app.route("/api/v1/recommendations", methods = ["GET"])
def recommendations():
    params = request.args.get('project_meta_ids')
    try:
        query = [int(x) for x in params[1:-1].split(",")]
    except TypeError:
        return Response(json.dumps({"error": "bad params"}, status=400))
    print(query)

    docs = get_by_id(data,query)
    result = pipe.predict(docs)

    # Dummmy response
    response_template = json.loads("""
            {
                "data": {
                    "type": "recommendations",
                    "id": "1",
                    "attributes": {
                        "project_metas": [],
                        "reflection": {}
                    }   
                }
            }
    """)


    response_template['data']['attributes']['project_metas'] = result
    print(response_template)
    resp = Response(json.dumps(response_template, cls=NumpyEncoder), status=200, mimetype='application/json')
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
