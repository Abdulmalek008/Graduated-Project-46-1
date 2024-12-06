
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Machine Learning App')

st.info('This is app builds a machine learning model!')

with st.expander('Data'):
  st.write('**Raw Data**')
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
  gender = st.selectbox('Gender',('Female','Male'))
  
  attendance_score = st.slider('attendance_score' , 1 , 5 , 3)
  mid_exam_score = st.slider('mid_exam_score' , 0 , 15 , 10)
  lap_exam_score = st.slider('lap_exam_score' , 0 , 15 , 10)
  activity_score = st.slider('activity_score' , 0 , 25 , 10)
  final_score = st.slider('final_score' , 0 , 100 , 50)

  data = {'Gender': gender,
        'Attendance_Score': attendance_score,
        'Mid_Exam_Score': mid_exam_score,
        'Lap_Exam_Score': lap_exam_score,
        'Activity_Score': activity_score,
        'Final_Score': final_score}
  input_df = pd.DataFrame(data, index=[0])
  input_student = pd.concat([input_df, X], axis=0)

with st.expander('input features'):
  st.write('**input student**')
  input_df
  st.write('**Comined student data**')
  input_student

encode = ['Gender']
df_student = pd.get_dummies(input_student, prefix=encode)
df_student[:1]






        
  
  
  
  
