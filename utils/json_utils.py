import json
import numpy as np


# Encoder to save numpy arrays
class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super.default(self, obj)



#returns documents with id in ids
def get_by_id(data,ids):
    if type(ids) == list:
        return list(filter(lambda x: x['id'] in ids, data))

