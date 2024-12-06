import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Machine Learning App')

st.info('This is app builds a machine learning model!')

with st.expander('Data'):
  st.write('**Raw Data*')
  df = pd.read_csv('https://raw.githubusercontent.com/Abdulmalek008/Graduated-Project-46-1/refs/heads/master/Student_Info%202.csv')
  df

st.write('**X**')
X = df.drop('Level', axis=1)
X

st.write('**y**')
y = df.Level
y

with st.expander('Data visualization'):
 st.scatter_chart(data=df , x='Attendance_Score', y='Total', color='Level') 

with st.sidebar:
  st.header('Input features')
  attendance = st.selecbox('Attendance',('1','2','3','4','5'))
  gender = st.selectbox('Gender',('Female','Male'))
 
  
