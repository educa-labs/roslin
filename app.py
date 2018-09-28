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
from transformers.outputs import DumbOutput, GreedyOutput, Output
from utils.json_utils import get_by_id, NumpyEncoder, map_from_id, map_to_id, get_by_id_map

from init import init

models, data, index_to_id, id_to_index = init()

app = Flask(__name__)
CORS(app)


@app.route("/api/v1/recommendations", methods=["GET"])
def recommendations():
    pipe = models[int(request.args.get('model'))]
    k = int(request.args.get('num_recs'))
    params = request.args.get('project_meta_ids')
    try:
        query = [x for x in params.split(",")]
    except [TypeError, ValueError]:
        return Response(json.dumps({"error": "bad params"}), status=400)

    try:
        docs = get_by_id_map(data, query, id_to_index)
    except KeyError:
        return Response(json.dumps({"error": "not found"}), status=404)
    set_output(pipe,k)
    result = pipe.predict(docs)
    output = [{"score": d, "quick_code": index_to_id[index]}
              for d, index in zip(result[0], result[1])]
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

    response_template['data']['attributes']['project_metas'] = output
    resp = Response(json.dumps(response_template, cls=NumpyEncoder),
                    status=200, mimetype='application/json')
    return resp


@app.route("/api/v1/projectmetas", methods=["GET"])
def project_loading():
    try:
        models, data, index_to_id, id_to_index = init(clear=True)

        return Response(status=204, mimetype='application/json')
    except Exception as e:
        return Response(str(e), status=500, mimetype='application/json')

def set_output(pipe,k):
    print(pipe.named_steps)
    pipe.named_steps['output'].set_k(k)
    pipe.named_steps['tree'].set_k(k)



if __name__ == '__main__':
    app.run()
