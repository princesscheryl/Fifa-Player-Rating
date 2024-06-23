import streamlit as st
import pandas as pd
import joblib as my_joblib
from sklearn.preprocessing import StandardScaler

st.title('FIFA Player Rating Prediction')

st.write('Kindly enter ratings for each category to obtain the overall rating for the player.')

#create input boxes for each feature
weak_foot = st.number_input('Weak Foot', min_value=0)
pace = st.number_input('Pace', min_value=0)
shooting = st.number_input('Shooting', min_value=0)
passing = st.number_input('Passing', min_value=0)
dribbling = st.number_input('Dribbling', min_value=0)
defending = st.number_input('Defending', min_value=0)
physic = st.number_input('Physic', min_value=0)


#create a submit button for the user
submit = st.button('Predict')

if submit:
    #create a dictionary with the user input
    user_input = {
        'weak_foot': weak_foot,
        'pace': pace,
        'shooting': shooting,
        'passing': passing,
        'dribbling': dribbling,
        'defending': defending,
        'physic': physic
        
    }

    #create a dataframe from user's input
    input_df = pd.DataFrame([user_input])

    #match column order to the training data order
    expected_columns = ['weak_foot','pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
    input_df = input_df[expected_columns]

    xgb_filename = 'modelxgb.pkl'
    xgb_model = joblib.load(xgb_filename)

    #load scaler 
    scaler_filename = 'scaler.pkl'
    scaler = joblib.load(scaler_filename)

    #print the model type
    st.write(f"Loaded model type: {xgb_model.__class__.__name__}")

    #scale input data to match training data
    input_df_scaled = scaler.transform(input_df)

    try:
        #predict using the selected best performing model: xgb 
        prediction = xgb_model.predict(input_df_scaled)
    except Exception as e:
        st.write(f"Error during prediction: {e}")
        prediction = [0] 

    #display overall player's rating
    st.write(f'Predicted Overall Player Rating: {round(prediction[0], 2)}')
