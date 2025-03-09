## Project Overview

Built a RESTful API using Flask to recommend courses based on content similarity:

- Extracted course data from MongoDB with `PyMongo`.
- Processed text using `NLTK` and `SpaCy` for tokenization and lemmatization.
- Trained a `Word2Vec` model with `Gensim` to vectorize course descriptions.
- Calculated cosine similarity of course vectors using `scikit-learn`.
- Returned course recommendations in JSON format via the Flask API.
