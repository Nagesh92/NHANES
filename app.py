from flask import Flask,request,render_template
import numpy as np
import pickle

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('results.html')

@app.route('/predict',methods=['POST'])
def predict():
    float_val = [float(a) for a in request.form.values()]
    final_val = np.array(float_val).reshape(1,8)
    predictions = model.predict(final_val)

    if predictions==0:
        output="The age group details belongs to Non-Senior Class"
    
    else:
        output="The Age Group details belongs to Senior Class"

    return render_template('results.html',output=output)

if __name__=="__main__":
    app.run(debug=True)

