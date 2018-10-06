# Roslin

Content-Based recommender system for roslin.

## Requirements

* `python 3.6.6`
* Mongo DB  ` >= 2.6`

For more see `requirements.txt`

## Set up 

Using a python virtual env is recommended. 

First install requirements:

```bash
pip install -r requirements.txt
```

Install nltk dependencies.

```bash
python
>>> import nltk
>>> nltk.download('punkt')
```

Run mongo DB

```bash
service mongod start
```

Run App

```
python app.py
```


## API

API has 2 endpoints.

1. `GET /api/v1/recommendations`

    * Query string params

        * `project_meta_ids(string)` (required): comma separated quick_codes for sample items to base recommendations on. Example:  DXHT,UU4L

        * `model (string)` (required): specifies model to compute recommendations.  For more info check models section

        * `num_recs(int)` (required): ammount of recommendations to return.

    * Response format

    ``` json
    {
        "data": {
            "type": "recommendations",
            "id": "1",
            "attributes": {
                "project_metas": [
                    {
                        "score": 0.85,
                        "quick_code": "PECN"
                    },
                    {
                        "score": 0.85,
                        "quick_code": "KYJF"
                    },
                    {
                        "score": 0.85,
                        "quick_code": "WJE7"
                    },
                    {
                        "score": 0.85,
                        "quick_code": "VK3N"
                    }
                ],
                "reflection": {}
            }
        }
    }
    ```

2. `GET /api/v1/projectmetas`

    Downloads data, retrains models and adds them to mongo DB. Takes time.


## Models


0. Traditional Knn ('knn'):

    Interprets the tags as a tabular dataset and measures similarity between items using a custom function. Then returns the most similar items to the ones in the input.

1. Averaged LDA ('lda_av'):

    Maps the sets of tags to a n dimensional vector using, TF-IDF embedding and the LDA algorithm. The dimensions of the vector represent "topics", and each component i represents the probability that the item belongs to topic i.  The output consists of averaging the vector of the input items and then returning the items with more similar distributions to the  resulted average.

2.  Greedy LDA ('lda_greedy'):
    
    Same as the previous one but instead of taking the average of the input vector, it greedily choses the items most similar  to the ones in the input.

3. Averaged Word LDA ('ldaw_av'):

    Same as 1 but bases the TF-IDF and LDA algorithms in words instead of complete tags.

4. Greedy Word LDA('ldaw_greedy'):

    Same as 2 but bases the TF-IDF and LDA algorithms in words instead of complete tags.

5. Averaged Word LDA ('ldan_av'):

    Same as 1 but adds the numerical tags as if they were normal tags.

6. Greedy Word LDA('ldan_greedy'):

    Same as 2 but adds the numerical tags as if they were normal tags.

7. Averaged Glove ('glove_av'):

    Maps every document to a 300 dimensional vector, using a pre trained model (Glove) that considers the semantics of a word. The document vector is computed as a weighted average of the word vectors using weights computed with TF-IDF algorithm.

8. Greedy Glove ('glove_greedy'):

    Same as but instead of taking the average of the input vector, it greedily choses the most similar items to the ones in the input.


## Components / Design

Expendable directories and files:

* `jupyter` (deprecated): contains jupyter notebooks for the initial model and data testing. Deprecated after first version of the functioning web service was developed.

* `visualization`: files needed to run the D3.js visualization. More information below.

* `reduce_dimensionality.py`: script to download the data and process it to create 2 dimensional vector embeddings for the visualization.

Main modules and files:

* `app.py`: this is the main flask app, imports all other modules, loads the models, and defines the routes.

* `builders.py`: Defines the models as sklearn pipeline objects, in this file model structure should be updated, and new models should be added.

* `init.py`: Defines function that starts the web service, downloads the needed data, and generates the necessary variables.

* `transformers`: Module that defines the components to create the model pipelines, includes the json parsers, vector embedders, trees for Knn search and output formatters.

* `loaders`: Modules that define the functions that download the data, and load the models from mongoDB.

* `utils`: Module with useful json and data manipulation functions.

* `word_hash`: serialized dictionary with word embeddings from glove.

## Visualization

The visualization consists of a 2D scatter plot, that uses the vector based models and a dimensionality reduction algorithm, map the items into a two dimensional vector. To use first run the `reduce_dimensionality.py` script, and then open  `graphs.html` with firefox browser or serve the file with a local server and open with any browser. hover over the points to see the kitchen images.

