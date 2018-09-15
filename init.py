import json as js
from sklearn.pipeline import Pipeline
from utils.json_utils import map_to_id, map_from_id

from loaders.models_loader import load_models
from loaders.data_loader import load_data


# KNN
from transformers.trees import KNNPredictor
from transformers.json_transformers import JSONTransformer


# Hielo
from transformers.trees import BallTreePredictor
from transformers.embedders import LdaTransformer
from transformers.json_transformers import JsonToTagsTransform


def KNN_BUILDER():
    pipeline_components = [('json', JSONTransformer()),
                           ('tree', KNNPredictor())]

    return Pipeline(pipeline_component)


def HIELO_BUILDER():
    pipeline_components = [('json', JsonToTagsTransform(
    )), ('embedder', LdaTransformer()), ('tree', BallTreePredictor())]

    return Pipeline(pipeline_components)


def init(clear=False):
    db_name = 'test-database'
    collection_name = 'models'

    load_data()

    data = js.load(open('data.json', encoding='utf-8'))

    models = {
        'knn': (KNN_BUILDER, data),
        'hielo': (HIELO_BUILDER, data)
    }

    models = load_models(
        db_name=db_name,
        collection_name=collection_name,
        models=models,
        clear=clear
    )

    return list(models.values()), data, map_to_id(data), map_from_id(data)


if __name__ == '__main__':
    print(init())
