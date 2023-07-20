import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
import pickle

#pyhtonanywhere model path
#'/home/Jupiter/mysite/model_keras_LSTM'
var_model_path = 'D:/anaconda3/envs/env1/notebooks/OP Notebooks/p7/Github/openclasroom-projet-7/model_keras_LSTM'
#pyhtonanywhere word embedding path
#'/home/Jupiter/mysite/tokenizer_GLOVE_LSTM_traite.pkl'
var_word_embeding_path = 'D:/anaconda3/envs/env1/notebooks/OP Notebooks/p7/Github/openclasroom-projet-7/tokenizer_GLOVE_LSTM_traite.pkl'
 

def decode_sentiment(score):
    if score < 0.5:
        label = "NEGATIVE"

        return label
    else:
        label = "POSITIVE"

        return label

def predict(text):
    model = tf.keras.models.load_model(var_model_path)


    tokenizer = pickle.load(open(var_word_embeding_path, "rb"))


    x_test = pad_sequences(tokenizer.texts_to_sequences([text]),maxlen=30)

    score = model.predict([x_test])[0]

    label = decode_sentiment(score)

    return label