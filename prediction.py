import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import pandas as pd 

###load all the model 
model = load_model('model.h5')

with open('label_enconder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)  

with open('scale.pkl', 'rb') as file:
    scaler = pickle.load(file) 

input_data = {
    "CreditScore": 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age' : 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts':2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}   

geo_encoded = onehot_encoder_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_df=pd.DataFrame([input_data])

#Encocde Categorical 
input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])



input_df = pd.concat([input_df.drop('Geography',axis=1),geo_encoded_df],axis=1)


input_scaled=scaler.transform(input_df)

prediction=model.predict(input_scaled)
prediction_proba = prediction[0][0]

if prediction_proba >0.5:
    print("Customer is likely to churn")
else:
    print("Customer is not likely to churn")
