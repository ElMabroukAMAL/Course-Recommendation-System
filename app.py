from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
import re
from nltk.corpus import stopwords
import spacy
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from bson import ObjectId

# Configuration de l'application Flask
app = Flask(__name__)

# Connexion à la base de données MongoDB
client = MongoClient('mongodb+srv://massarrabenjebiri:VLafkV0BnzehFsOy@cluster0.34hr6si.mongodb.net/')
db = client['test']
courses_collection = db['courses']

# Télécharger les stopwords français et charger le modèle de langue française
nltk.download('stopwords')
nltk.download('punkt')
fr_stopwords = stopwords.words("french")
nlp = spacy.load('fr_core_news_sm')

# Fonction de lemmatisation
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# Définir la classe recommender
class recommender:
    def __init__(self):
        self.corpus = self.load_courses()
        self.course_names = list(self.corpus['title'])

    def load_courses(self):
        courses = []
        all_courses = courses_collection.find()
        for course in all_courses:
            courses.append({
                'title': course['title'],
                'sousTitre': course.get('sousTitre', ''),
                'description': course['description'],
                'apprendreCours': course['apprendreCours'],
                'image': course['image'],
                'categorys': course.get('categorys', []),
                'text': f"{course['title']} {course.get('sousTitre', '')} {course['description']} {course['apprendreCours']}"
            })
        return pd.DataFrame(courses)

    # Fonction pour nettoyer le texte
    def clean_word2vec(self, text):
        text = re.sub("[^A-Za-z1-9 ]", "", text)
        text = text.lower()
        tokens = word_tokenize(text)
        clean_list = [lemmatize_text(token) for token in tokens if token not in fr_stopwords]
        return clean_list

    # Fonction pour obtenir l'embedding moyen d'un cours
    def doc_vectorizer(self, doc, model):
        doc_vector = np.mean([model.wv[word] for word in doc if word in model.wv] or [np.zeros(model.vector_size)], axis=0)
        return doc_vector

    # Fonction pour recommander des cours
    def course_recommender_w2v(self, course, X, category):
        corpus = self.corpus[self.corpus['categorys'].apply(lambda x: category in x if isinstance(x, list) else False)]
        corpus = corpus.reset_index(drop=True)
        X = X[corpus.index]
        cosine_similarities = cosine_similarity(X, X)
        courses = corpus[['title']]
        indices = pd.Series(corpus.index, index=corpus['title']).drop_duplicates()
        idx = indices[course]
        sim_scores = list(enumerate(cosine_similarities[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:5]
        course_indices = [i[0] for i in sim_scores]
        recommend = courses.iloc[course_indices]
        recommendations = recommend['title'].tolist()
        return recommendations

    # Fonction principale de recommandation
    def w2v_recommend(self, course_title, category):
        cleaned_corpus = self.corpus['text'].apply(self.clean_word2vec)
        model = Word2Vec(sentences=cleaned_corpus, vector_size=100, window=5, min_count=1, workers=4)
        #model.save("modele/word2vec_model.bin")
        corpus_vectors = np.array([self.doc_vectorizer(doc, model) for doc in cleaned_corpus])
        return self.course_recommender_w2v(course_title, corpus_vectors, category)

    
@app.route('/api/recommendations/<course_id>', methods=['POST'])
def recommend(course_id):
    # Trouver le cours correspondant à l'ID donné
    course_data = courses_collection.find_one({'_id': ObjectId(course_id)})
    
    if not course_data:
        return jsonify({"error": "Course not found"}), 404

    course_title = course_data['title']
    category = course_data.get('categorys', [])[0] if course_data.get('categorys') else ''

    recommender_obj = recommender()
    recommendation = recommender_obj.w2v_recommend(course_title, category)
   
    recommendations = []
    for rec in recommendation:
        rec_data = courses_collection.find_one({'title': rec})
        recommendations.append({
            '_id' : str(rec_data['_id']),
            'title': rec_data['title'],
            'sousTitre': rec_data.get('sousTitre', ''),
            'image': rec_data['image'],
            'categorys': rec_data.get('categorys', [])
        })

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=4000)
