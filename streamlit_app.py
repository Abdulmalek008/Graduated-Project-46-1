import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Machine Learning Application for Predicting Students Final Grade')

st.info('This app builds a machine learning model!')

# تحميل البيانات
with st.expander('Data'):
    st.write('**Raw Data**')
    df = pd.read_csv('https://raw.githubusercontent.com/Abdulmalek008/Graduated-Project-46-1/refs/heads/master/Student_Info%202.csv')
    st.write(df)

# إضافة التصنيف بناءً على المجموع الإجمالي
def assign_grade(total):
    if total > 80:
        return 'A'
    elif 60 <= total <= 80:
        return 'B'
    else:
        return 'C'

df['Total'] = df[['Mid_Exam_Score', 'Lab_Exam_Score', 'Activity_Score', 'Final_Score']].sum(axis=1)
df['Grade'] = df['Total'].apply(assign_grade)

with st.expander('Processed Data'):
    st.write('**Data with Total and Grade**')
    st.write(df)

# تحضير X و y للنموذج
X = df.drop(['Grade', 'Level', 'Total'], axis=1)
y = df['Grade']

# تشفير المدخلات
X_encoded = pd.get_dummies(X, columns=['Student_ID', 'Gender'], drop_first=True)

# تشفير المخرجات
target_mapper = {'A': 0, 'B': 1, 'C': 2}
y_encoded = y.map(target_mapper)

# تدريب النموذج
clf = RandomForestClassifier()
clf.fit(X_encoded, y_encoded)

# إدخال بيانات طالب جديد
st.sidebar.header('Input features')
student_ID = st.sidebar.selectbox('Student_ID', [f"S{str(i).zfill(3)}" for i in range(1, 151)])
gender = st.sidebar.selectbox('Gender', ('Female', 'Male'))
attendance_score = st.sidebar.slider('Attendance_Score', 1, 5, 3)
mid_exam_score = st.sidebar.slider('Mid_Exam_Score', 0, 15, 10)
lab_exam_score = st.sidebar.slider('Lab_Exam_Score', 0, 15, 10)
activity_score = st.sidebar.slider('Activity_Score', 0, 25, 10)
final_score = st.sidebar.slider('Final_Score', 0, 40, 20)

# حساب المجموع والتصنيف اليدوي
total_score = mid_exam_score + lab_exam_score + activity_score + final_score
manual_grade = assign_grade(total_score)

# إنشاء إدخال الطالب
input_data = pd.DataFrame({
    'Student_ID': [student_ID],
    'Gender': [gender],
    'Attendance_Score': [attendance_score],
    'Mid_Exam_Score': [mid_exam_score],
    'Lab_Exam_Score': [lab_exam_score],
    'Activity_Score': [activity_score],
    'Final_Score': [final_score]
})

# تشفير بيانات الطالب
input_encoded = pd.get_dummies(input_data, columns=['Student_ID', 'Gender'], drop_first=True)
input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# تنبؤ النموذج
prediction = clf.predict(input_encoded)
prediction_proba = clf.predict_proba(input_encoded)

# عرض النتائج
st.subheader('Predicted Grade')
grades = {0: 'A', 1: 'B', 2: 'C'}
st.write(f"Predicted grade by model: **{grades[prediction[0]]}**")

st.subheader('Prediction Probabilities')
proba_df = pd.DataFrame(prediction_proba, columns=['A', 'B', 'C'])
st.write(proba_df)

st.subheader('Manual Grade Classification')
st.write(f"Manual grade based on Total ({total_score}): **{manual_grade}**")
