from flask import Flask , request, render_template
from model import predict
import git

#pyhtonanywhere templates folder path
var_template_folder = '/home/Alwis/openclasroom-projet-7/templates/'
# var_template_folder = 'D:/anaconda3/envs/env1/notebooks/OP Notebooks/p7/Github/openclasroom-projet-7/templates' 
 
app = Flask('Prediction des sentiments sur twitter',template_folder = var_template_folder)

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        user_input = request.form["user_tweet"]
        label = predict(user_input)
        return render_template('form.html', prediction_text = label)
    return render_template('form.html')


@app.route('/update_server', methods=['POST'])
def webhook():
    if request.method == 'POST':
        repo = git.Repo('https://github.com/RoiJupiter/openclasroom-projet-7')
        origin = repo.remotes.origin
        origin.pull()
        return 'Updated PythonAnywhere successfully', 200
    else:
        return 'Wrong event type', 400


if __name__ == '__main__':
    app.run()