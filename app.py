from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template

tokenizer = Tokenizer()

stemmer = SnowballStemmer('english')

stop_words = stopwords.words('english')
text_cleaning_regex = "@S+|https?:S+|http?:S|[^A-Za-z0-9]+"

def clean_tweets(text, stem=False):
    # Text passed to the regex equatio
    text = re.sub(text_cleaning_regex, ' ', str(text).lower()).strip()
    # Empty list created to store final tokens
    tokens = []
    for token in text.split():
    # check if the token is a stop word or not
        if token not in stop_words:
            if stem:
                # Paased to the snowball stemmer
                tokens.append(stemmer.stem(token))
            else:
                # A
                tokens.append(token)
    return " ".join(tokens)

def predict_tweet_sentiment(score):
    return "Positive" if score>0.5 else "Negative"

def y_pred(score):
    return 1 if score>0.5 else 0


app = Flask(__name__)

## Load the model
model = pickle.load(open('model_keras_LSTM.pkl','rb'))

@app.route('/predict_api',methods=['POST'])
def predict_api():
 
    #récupération de la donnée
    data = request.json['data']

    #traitement de la donnée
    text_cleaned = clean_tweets(data)

    text_arrayed = []
    text_arrayed.append(text_cleaned)

    text_pad_sequences = pad_sequences(tokenizer.texts_to_sequences(text_arrayed), maxlen = 30)

    scores = model.predict(text_pad_sequences, verbose=1, batch_size=10000)

    model_predictions = [predict_tweet_sentiment(score) for score in scores]

    return json.dumps(model_predictions)
 
if __name__=="__main__":
    app.run(debug=True)