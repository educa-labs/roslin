import json as js
import pickle

from collections import OrderedDict
from sklearn.pipeline import Pipeline
from utils.json_utils import map_to_id, map_from_id, get_by_id

from loaders.data_loader import load_data
from loaders.models_loader import load_models

from builders import KNN_BUILDER
from builders import LDA_TAGS_AVERAGE_BUILDER, LDA_TAGS_GREEDY_BUILDER, LDA_WORDS_AVERAGE_BUILDER, LDA_WORDS_GREEDY_BUILDER,LDA_TAGS_NUM_AVERAGE_BUILDER,LDA_TAGS_NUM_GREEDY_BUILDER
from builders import GLOVE_AVERAGE_BUILDER, GLOVE_GREEDY_BUILDER

'''
Loads the models and data to memory,  returns all the needed  variables
for the main app to work.

After creating a new model on builders.py add it here, to the models hash for 
it to be accesible through the api.
'''
def init(clear=False):
    collection_name = 'models'

    load_data()

    data = js.load(open('data.json', encoding='utf-8'))

    models = OrderedDict()

    models['knn'] = (KNN_BUILDER, data)

    models['lda_av'] = (LDA_TAGS_AVERAGE_BUILDER, data)
    models['lda_greedy'] = (LDA_TAGS_GREEDY_BUILDER, data)
    models['ldaw_av'] = (LDA_WORDS_AVERAGE_BUILDER, data)
    models['ldaw_greedy'] = (LDA_WORDS_GREEDY_BUILDER, data)

    models['ldan_av'] = (LDA_TAGS_NUM_AVERAGE_BUILDER, data)
    models['ldan_greedy'] = (LDA_TAGS_NUM_GREEDY_BUILDER, data)

    models['glove_av'] = (GLOVE_AVERAGE_BUILDER, data)
    models['glove_greedy'] = (GLOVE_GREEDY_BUILDER, data)

    models = load_models(
        collection_name=collection_name,
        models=models,
        clear=clear
    )

    return models, data, map_to_id(data), map_from_id(data)


if __name__ == '__main__':
    init(clear=True)