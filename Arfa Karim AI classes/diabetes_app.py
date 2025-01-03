import pandas
import numpy as np
import streamlit as st
import pickle
model=pickle.load(open("C:\\Users\\U$ER\\Documents\\AKTI\\model.pkl", "rb"))
prob=pickle.load(open("C:\\Users\\U$ER\\Documents\\AKTI\\lr_mode.pkl", "rb"))


st.title("Lets predict diabetes")
st.header("Enter patient details")



st.sidebar.title("Enter you details")
name = st.sidebar.text_input("Name")
email = st.sidebar.text_input("Email")
message = st.sidebar.text_input("contact no: ")
if st.sidebar.button("submit"):
    st.success("Your detailes have been entered")

col1, col2=st.columns(2)

with col1:
    pregnancies=st.number_input("Enter number of pregnancies", step=0.1, min_value=0.0)
    glucose=st.number_input("Enter glucose level", step=0.1, min_value=0.0)
    bp=st.number_input("Blood pressure", step=0.1, min_value=0.0)
    skin_thickness=st.number_input("level of skin thickness", step=0.1, min_value=0.0)

with col2:
    insulin=st.number_input("insulin level", step=0.1, min_value=0.0)
    bmi=st.number_input("BMI value", step=0.1, min_value=0.0)
    db_func=st.number_input("diabetes function", step=0.1, min_value=0.0)
    age=st.number_input("Age", step=0.1, min_value=0.0)


if st.button("Predict"):
    input_features=np.array([pregnancies, glucose, bp, skin_thickness, insulin, bmi,db_func, age ]).reshape(1, -1)
    pred=model.predict(input_features)
    if(pred==1):
        st.success("The person is likely to have diabetes")
        list_=prob.predict_proba(input_features)[0]
        probability=list_[1]*100
        st.success(f"You are {probability:.2f} % susceptible to having diabetes")
        st.markdown("[enter your message](www.google.com)")
        
    else:
        st.success("The person will not have diabetes")
    