import json as js
import pickle

from sklearn.pipeline import Pipeline
from utils.json_utils import map_to_id, map_from_id
from collections import OrderedDict

from loaders.models_loader import load_models
from loaders.data_loader import load_data

from transformers.outputs import GreedyOutput, DumbOutput

# KNN
from transformers.trees import KNNPredictor
from transformers.json_transformers import JSONTransformer


# LDA
from transformers.trees import BallTreePredictor
from transformers.embedders import LdaTransformer, TfIdfGloveTransformer
from transformers.json_transformers import JsonToTagsTransform, JsonTransform


def KNN_BUILDER():
    pipeline_components = [('json', JSONTransformer()),
                           ('tree', KNNPredictor()),
                           ('ouput',GreedyOutput())]

    return Pipeline(pipeline_components)

#LDA MODELS
def LDA_TAGS_AVERAGE_BUILDER():
    pipeline_components = [('json', JsonToTagsTransform(
    )), ('embedder', LdaTransformer()), ('tree', BallTreePredictor(average=True)), ('output', DumbOutput())]

    return Pipeline(pipeline_components)

def LDA_TAGS_GREEDY_BUILDER():
    pipeline_components = [('json', JsonToTagsTransform(
    )), ('embedder', LdaTransformer()), ('tree', BallTreePredictor()), ('output', GreedyOutput())]

    return Pipeline(pipeline_components)

def LDA_WORDS_AVERAGE_BUILDER():
    pipeline_components = [('json', JsonTransform(
    )), ('embedder', LdaTransformer()), ('tree', BallTreePredictor(average=True)), ('output', DumbOutput())]

    return Pipeline(pipeline_components)

def LDA_WORDS_GREEDY_BUILDER():
    pipeline_components = [('json', JsonTransform(
    )), ('embedder', LdaTransformer()), ('tree', BallTreePredictor()), ('output', GreedyOutput())]

    return Pipeline(pipeline_components)

# Glove Models

with open("word_hash","rb") as file:
    word_hash = pickle.load(file)

def GLOVE_AVERAGE_BUILDER():
    pipeline_components = [('json', JsonTransform(
    )), ('embedder', TfIdfGloveTransformer(word_embedder=word_hash)), ('tree', BallTreePredictor(average=True)), ('output', DumbOutput())]

    return Pipeline(pipeline_components)

def GLOVE_GREEDY_BUILDER():
    pipeline_components = [('json', JsonTransform(
    )), ('embedder', TfIdfGloveTransformer(word_embedder=word_hash)), ('tree', BallTreePredictor()), ('output', GreedyOutput())]

    return Pipeline(pipeline_components)



def init(clear=False):
    collection_name = 'models'

    load_data()

    data = js.load(open('data.json', encoding='utf-8'))

    models = OrderedDict()
    #models['knn'] = (KNN_BUILDER, data)
    models['lda_av_builder'] = (LDA_TAGS_AVERAGE_BUILDER, data)
    models['lda_greedy_builder'] = (LDA_TAGS_GREEDY_BUILDER,data)

    models['ldaw_av_builder'] = (LDA_WORDS_AVERAGE_BUILDER, data)
    models['ldaw_greedy_builder'] = (LDA_WORDS_GREEDY_BUILDER,data)

    models['glove_av_builder'] = (GLOVE_AVERAGE_BUILDER,data)
    models['glove_greedy_builder'] = (GLOVE_GREEDY_BUILDER,data)
    

    models = load_models(
        collection_name=collection_name,
        models=models,
        clear=clear
    )

    return list(models.values()), data, map_to_id(data), map_from_id(data)


if __name__ == '__main__':
    init(clear=False)
