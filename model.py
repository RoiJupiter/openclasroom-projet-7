import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
import pickle



def decode_sentiment(score):
    if score < 0.5:
        label = "NEGATIVE"

        return label
    else:
        label = "POSITIVE"

        return label

def predict(text):
    model = tf.keras.models.load_model('/home/Jupiter/mysite/model_keras_LSTM')


    tokenizer = pickle.load(open('/home/Jupiter/mysite/tokenizer_GLOVE_LSTM_traite.pkl', "rb"))


    x_test = pad_sequences(tokenizer.texts_to_sequences([text]),maxlen=30)

    score = model.predict([x_test])[0]

    label = decode_sentiment(score)

    return label