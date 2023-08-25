import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
import pickle

#pyhtonanywhere variables
var_model_path = '/home/Alwis/openclasroom-projet-7/model_keras_LSTM'
var_word_embeding_path = '/home/Alwis/openclasroom-projet-7/tokenizer_GLOVE_LSTM_traite.pkl'

#local variables
#var_model_path = 'D:/anaconda3/envs/env1/notebooks/OP Notebooks/p7/Github/openclasroom-projet-7/model_keras_LSTM'
#var_word_embeding_path = 'D:/anaconda3/envs/env1/notebooks/OP Notebooks/p7/Github/openclasroom-projet-7/tokenizer_GLOVE_LSTM_traite.pkl'


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


# Test unitaire

def test_predict():

    # check le path du model
    expected_output = '/home/Alwis/openclasroom-projet-7/model_keras_LSTM'
    assert var_model_path == expected_output

    print("check le path du model test unitaire passé")

    # Cas de test avec score > 0.5
    score = 0.8
    expected_output = "POSITIVE"
    assert decode_sentiment(score) == expected_output

    print("cas de test avec score > 0.5 test unitaire passé")

    # Cas de test avec score <= 0.5
    score = 0.3
    expected_output = "NEGATIVE"
    assert decode_sentiment(score) <= expected_output

    print("Cas de test avec score <= 0.5 test unitaire passé")
