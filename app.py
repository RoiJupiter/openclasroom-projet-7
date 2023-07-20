from flask import Flask , request, render_template
from model import predict

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

if __name__ == '__main__':
    app.run()