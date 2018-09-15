from sklearn.pipeline import Pipeline

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


def init():
    db_name = 'test-database'
    collection_name = 'models'

    DATA_PATH = 'data.json'

    MODELS = {
        'knn': (KNN_BUILDER, DATA_PATH),
        'hielo': (HIELO_BUILDER, DATA_PATH)
    }

    models = load_models(
        db_name=db_name,
        collection_name=collection_name,
        models=MODELS
    )

    load_data()

    return list(models.values())


if __name__ == '__main__':
    print(init())
