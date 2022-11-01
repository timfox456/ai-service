import pickle

from flask import Flask, request, jsonify

import model_serving

app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    
    
    with open('churn-model.bin', 'rb') as f_in: 
        dv, model = pickle.load(f_in)

    prediction = model_serving.predict_single(customer, dv, model)
    churn = prediction > 0.5
    
    result = {
        'churn_probability': float(prediction),
        'churn' : bool(churn)
    }
    return jsonify(result)
    
    
if __name__ == '__main__': 
    app.run(debug=True, host='0.0.0.0', port=9696)
    
    

