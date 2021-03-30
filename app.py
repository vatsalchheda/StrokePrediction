
from flask import Flask,request
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)

RandomClassifier = pickle.load(open('RandomForestmodel.pkl','rb'))
fs = pickle.load(open('featurescale.pkl','rb'))
OH_Encoder = pickle.load(open('OH_Encoder.pkl','rb'))

@app.route("/")
def index():
    return 'This is the homepage'

@app.route("/predict",methods=["POST"])

def predict():
    data = request.get_json()
    print(data)
    test_data = pd.DataFrame(data,index=[0])
    print(type(test_data))
    print(test_data)
    Categorical_cols = ['gender','ever_married','work_type','Residence_type','smoking_status']
    X_encoded = pd.DataFrame(OH_Encoder.transform(test_data[Categorical_cols]))
    X_encoded.index = test_data.index
    num_X = test_data.drop(Categorical_cols, axis=1)
    print(num_X)
    OH_X = pd.concat([num_X, X_encoded], axis=1)
    OH_X.iloc[:,0:5] = fs.transform(OH_X.iloc[:,0:5])
    preds = RandomClassifier.predict(OH_X)
    output = str(preds[0])
      
    return output
    


if __name__ == "__main__":
    app.run(debug=True)
    