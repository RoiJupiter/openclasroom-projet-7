from flask import Flask , request, render_template
from model import predict

app = Flask('Prediction des sentiments sur twitter',template_folder='/home/Jupiter/mysite/templates/')

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        user_input = request.form["user_tweet"]
        label = predict(user_input)
        return render_template('form.html', prediction_text = label)
    return render_template('form.html')

if __name__ == '__main__':
    app.run()