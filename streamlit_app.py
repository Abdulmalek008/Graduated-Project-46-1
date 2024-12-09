import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

st.title('🤖 Machine Learning Application for Predicting Students Final Grade and Level')

st.info('This app predicts both the student\'s final grade and performance level!')

with st.expander('Data'):
    st.write('**Raw Data**')
    df = pd.read_csv('https://raw.githubusercontent.com/Abdulmalek008/Graduated-Project-46-1/refs/heads/master/Student_Info%202.csv')
    st.write(df.head())

    # فصل المدخلات والمخرجات
    X_raw = df.drop(['Level', 'Final_Score'], axis=1)
    y_level = df['Level']
    y_score = df['Final_Score']

    st.write('**X (Features)**')
    st.write(X_raw)
    st.write('**y_level (Target - Level)**')
    st.write(y_level.head())
    st.write('**y_score (Target - Final Score)**')
    st.write(y_score.head())

with st.sidebar:
    st.header('Input Features')
    student_ID = st.selectbox('Student ID', sorted(df['Student_ID'].unique()))
    gender = st.selectbox('Gender', ('Female', 'Male'))
    attendance_score = st.slider('Attendance Score', 1, 5, 3)
    mid_exam_score = st.slider('Mid Exam Score', 0, 15, 10)
    lab_exam_score = st.slider('Lab Exam Score', 0, 15, 10)
    activity_score = st.slider('Activity Score', 0, 25, 10)

    data = {
        'Student_ID': student_ID,
        'Gender': gender,
        'Attendance_Score': attendance_score,
        'Mid_Exam_Score': mid_exam_score,
        'Lab_Exam_Score': lab_exam_score,
        'Activity_Score': activity_score,
    }

    input_df = pd.DataFrame(data, index=[0])
    input_student = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input Features'):
    st.write('**Input Data for Prediction**')
    st.write(input_df)

    st.write('**Combined Data for Encoding**')
    st.write(input_student)

# معالجة البيانات
encode_cols = ['Student_ID', 'Gender']
df_encoded = pd.get_dummies(input_student, columns=encode_cols, drop_first=True)

X_encoded = df_encoded[1:]
input_row = df_encoded.iloc[0:1]

# تحويل الهدف إلى قيم عددية
level_mapper = {'A': 0, 'B': 1, 'C': 2}
y_level_encoded = y_level.map(level_mapper)

# التحقق من القيم المفقودة
X_encoded = X_encoded.fillna(0)
input_row = input_row.fillna(0)

# التأكد من تطابق الأعمدة
X_encoded, input_row = X_encoded.align(input_row, join="inner", axis=1)

# تحويل القيم المستهدفة إلى أرقام صحيحة
y_level_encoded = y_level_encoded.astype(int)

# تقسيم البيانات
X_train_level, X_test_level, y_train_level, y_test_level = train_test_split(X_encoded, y_level_encoded, test_size=0.2, random_state=42)
X_train_score, X_test_score, y_train_score, y_test_score = train_test_split(X_encoded, y_score, test_size=0.2, random_state=42)

# تدريب النماذج
level_model = RandomForestClassifier(random_state=42)
level_model.fit(X_train_level, y_train_level)

score_model = RandomForestRegressor(random_state=42)
score_model.fit(X_train_score, y_train_score)

# التنبؤات
level_prediction = level_model.predict(input_row)
level_proba = level_model.predict_proba(input_row)
score_prediction = score_model.predict(input_row)

# عرض النتائج
st.subheader('Predicted Final Score')
st.success(f"Predicted Final Score: {score_prediction[0]:.2f}")

st.subheader('Predicted Performance Level')
level_labels = ['A', 'B', 'C']
st.success(f"Predicted Level: {level_labels[level_prediction[0]]}")

# عرض احتمالات الأداء
df_level_proba = pd.DataFrame(level_proba, columns=level_labels)
st.write('**Level Prediction Probabilities**')
st.dataframe(df_level_proba.style.format("{:.2%}"))
