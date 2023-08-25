from model import decode_sentiment
from keras_preprocessing.sequence import pad_sequences

# Test unitaire

def test_predict():

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


test_predict()