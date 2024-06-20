import pandas as pd 
import numpy as np
import sklearn
import joblib
import streamlit

model=open('linier_regression_model.pkl','rb')
lr_model = joblib.load(model)

def lr_prediction(alc,vol,sul,dio):
    """_summary_
    Predicts the target variable using a linear regression model based on the input features.
    """    
    pred_arr = np.array([alc,vol,sul,dio])
    preds = pred_arr.reshape(1,-1)
    preds = preds.astype(int)
    model_prediction = lr_model.predict(preds)
    return model_prediction

def run():
    streamlit.title('Wine Quality Prediction')
    html_temp = """
    """
    streamlit.markdown(html_temp, unsafe_allow_html=True)
    alc = streamlit.number_input("Alcohol")
    vol = streamlit.number_input("Volatile acidity")
    sul = streamlit.number_input("Sulphates")
    dio = streamlit.number_input("Total sulfur dioxide")
    result = ""
    if streamlit.button("Predict"):
        result = lr_prediction(alc,vol,sul,dio)
        streamlit.success('The quality of the wine is {}'.format(result))
        
        
if __name__ == '__main__':
    run()      
    
