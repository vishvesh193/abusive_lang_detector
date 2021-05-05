#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
from scipy import sparse
import pickle
import json

from flask import request,make_response


#Initialize the flask App
app = Flask(__name__)
model0 = pickle.load(open('model0.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
model3 = pickle.load(open('model3.pkl', 'rb'))
model4 = pickle.load(open('model4.pkl', 'rb'))
model5 = pickle.load(open('model5.pkl', 'rb'))

vec_word = pickle.load(open('vec_word.pkl', 'rb'))
vec_char = pickle.load(open('vec_char.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [float(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]

    ra_ts_vect = vec_word.transform([request.form['user_input_text']])
    ra_ts_vect_char = vec_char.transform([request.form['user_input_text']])
    ra_x_test = sparse.hstack([ra_ts_vect, ra_ts_vect_char])

    prediction0 = model0.predict(ra_x_test)
    prediction1 = model1.predict(ra_x_test)
    prediction2 = model2.predict(ra_x_test)
    prediction3 = model3.predict(ra_x_test)
    prediction4 = model4.predict(ra_x_test)
    prediction5 = model5.predict(ra_x_test)

    # if(prediction0[0] + prediction1[0] + 0.6 or prediction2.any() > 0.6 or prediction3.any() > 0.6 or prediction4.any() > 0.6 or prediction5.any() > 0.6):
    #     output = "You are bad boy!"
    # else:
    #     output = "You are good boy!"

    output = ""

    if prediction0.any() == 1 :
        output += " Toxic,"
    if prediction1.any() == 1 :
        output += " Severe_Toxic,"
    if prediction2.any() == 1 :
        output += " Obscene,"
    if prediction3.any() == 1 :
        output += " Threat,"
    if prediction4.any() == 1 :
        output += " Insult,"
    if prediction5.any() == 1 :
        output += " Identity_Hate."
    

    if output == "" :   
        return render_template('index.html', prediction_text='No abusive language detected')


    return render_template('index.html', prediction_text='Categories identified :{}'.format(output))

@app.route('/again')
def again():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [float(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    text = request.args.get('text')

    ra_ts_vect = vec_word.transform([text])
    ra_ts_vect_char = vec_char.transform([text])
    ra_x_test = sparse.hstack([ra_ts_vect, ra_ts_vect_char])

    prediction0 = model0.predict(ra_x_test)
    prediction1 = model1.predict(ra_x_test)
    prediction2 = model2.predict(ra_x_test)
    prediction3 = model3.predict(ra_x_test)
    prediction4 = model4.predict(ra_x_test)
    prediction5 = model5.predict(ra_x_test)

    # if(prediction0[0] + prediction1[0] + 0.6 or prediction2.any() > 0.6 or prediction3.any() > 0.6 or prediction4.any() > 0.6 or prediction5.any() > 0.6):
    #     output = "You are bad boy!"
    # else:
    #     output = "You are good boy!"

    output = ""

    if prediction0.any() == 1 :
        output += " Toxic,"
    if prediction1.any() == 1 :
        output += " Severe_Toxic,"
    if prediction2.any() == 1 :
        output += " Obscene,"
    if prediction3.any() == 1 :
        output += " Threat,"
    if prediction4.any() == 1 :
        output += " Insult,"
    if prediction5.any() == 1 :
        output += " Identity_Hate."

    if output == "" : 
        output="NO"
    else :
        output="YES"

    answer=json.dumps(output)
    res=make_response(answer)
    res.headers['Access-Control-Allow-Origin'] = '*'
    res.headers['Access-Control-Allow-Headers']= "Origin, X-Requested-With, Content-Type, Accept"

    return res
    


    

    


if __name__ == "__main__":
    app.run(debug=True)
