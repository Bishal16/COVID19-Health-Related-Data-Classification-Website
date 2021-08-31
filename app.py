import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    #if request.method == 'POST':
	message = request.form['message']
	data = [message]
	
	# load the vectorizer
	loaded_vectorizer = joblib.load(open('vectorizer.pkl', 'rb'))
	# load the model
	loaded_model = joblib.load(open('classification.pkl', 'rb'))

	#data = "win lottery prize money"
	my_prediction = loaded_model.predict(loaded_vectorizer.transform(data))

	str1 = ""    
	str1 = str1.join(message)
	print(type(str1))
	print(str1)

	return render_template('index.html',prediction = my_prediction,str1=str1)


    



if __name__ == "__main__":
    app.run(host='127.0.0.9',port=4455,debug=True)