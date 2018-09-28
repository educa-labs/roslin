import pickle

from sklearn.pipeline import Pipeline

# KNN transformers
from transformers.trees import KNNPredictor
from transformers.json_transformers import JSONTransformer

# LDA transformers
from transformers.trees import BallTreePredictor
from transformers.embedders import LdaTransformer, TfIdfGloveTransformer
from transformers.json_transformers import JsonToTagsTransform, JsonTransform

# GLOVE transformers
from transformers.outputs import GreedyOutput, DumbOutput


json_layer = 'json'
embedder = 'embedder'
tree = 'tree'
output = 'output'

# KNN pipelines

def KNN_BUILDER():
    pipeline_components = [(json_layer, JSONTransformer()),
                           (tree, KNNPredictor()),
                           (output, GreedyOutput())]

    return Pipeline(pipeline_components)


# LDA pipelines


def LDA_TAGS_AVERAGE_BUILDER():
    pipeline_components = [(json_layer, JsonToTagsTransform(
    )), (embedder, LdaTransformer()), (tree, BallTreePredictor(average=True)), (output, DumbOutput())]

    return Pipeline(pipeline_components)


def LDA_TAGS_GREEDY_BUILDER():
    pipeline_components = [(json_layer, JsonToTagsTransform(
    )), (embedder, LdaTransformer()), (tree, BallTreePredictor()), (output, GreedyOutput())]

    return Pipeline(pipeline_components)


def LDA_WORDS_AVERAGE_BUILDER():
    pipeline_components = [(json_layer, JsonTransform(
    )), (embedder, LdaTransformer()), (tree, BallTreePredictor(average=True)), (output, DumbOutput())]

    return Pipeline(pipeline_components)


def LDA_WORDS_GREEDY_BUILDER():
    pipeline_components = [(json_layer, JsonTransform(
    )), (embedder, LdaTransformer()), (tree, BallTreePredictor()), (output, GreedyOutput())]

    return Pipeline(pipeline_components)


# GLOVE pipelines


def GLOVE_AVERAGE_BUILDER():
    with open("word_hash", "rb") as file:
        word_hash = pickle.load(file)

        pipeline_components = [(json_layer, JsonTransform(
        )), (embedder, TfIdfGloveTransformer(word_embedder=word_hash)), (tree, BallTreePredictor(average=True)), (output, DumbOutput())]

        return Pipeline(pipeline_components)


def GLOVE_GREEDY_BUILDER():
    with open("word_hash", "rb") as file:
        word_hash = pickle.load(file)

        pipeline_components = [(json_layer, JsonTransform(
        )), (embedder, TfIdfGloveTransformer(word_embedder=word_hash)), (tree, BallTreePredictor()), (output, GreedyOutput())]

        return Pipeline(pipeline_components)
