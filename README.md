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

        * `model (int)` (required): specifies model to compute recommendations. int in range [0,6] For mor info check models section./api/v1/projectmetas

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


0. Traditional Knn:

    Interprets the tags as a tabular dataset and measures similarity between items using a custom function. Then returns the most similar items to the ones in the input.

1. Averaged LDA:

    Maps the sets of tags to a n dimensional vector using, TF-IDF embedding and the LDA algorithm. The dimensions of the vector represent "topics", and each component i represents the probability that the item belongs to topic i.  The output consists of averaging the vector of the input items and then returning the items with more similar distributions to the  resulted average.

2.  Greedy LDA:
    
    Same as the previous one but instead of taking the average of the input vector, it greedily choses the items most similar  to the ones in the input.

3. Averaged Word LDA:

    Same as 1 but bases the TF-IDF and LDA algorithms in words instead of complete tags.

4. Greedy Word LDA:

    Same as 2 but bases the TF-IDF and LDA algorithms in words instead of complete tags.

5. Averaged Glove:

    Maps every document to a 300 dimensional vector, using a pre trained model (Glove) that considers the semantics of a word. The document vector is computed as a weighted average of the word vectors using weights computed with TF-IDF algorithm.

6. Greedy Glove:

    Same as but instead of taking the average of the input vector, it greedily choses the most similar items to the ones in the input.
