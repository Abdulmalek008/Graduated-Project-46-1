import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.title('🤖 Predicting Final Score (out of 40)')

st.info('This app predicts the final score (out of 40) for students based on their performance metrics.')

with st.expander('Data'):
    st.write('**Raw Data**')
    df = pd.read_csv('https://raw.githubusercontent.com/Abdulmalek008/Graduated-Project-46-1/refs/heads/master/Student_Info%202.csv')
    st.write(df.head())

    # فصل المدخلات والمخرجات
    X_raw = df.drop(['Level', 'Final_Score', 'Student_ID'], axis=1)  # حذف الأعمدة غير المهمة للتنبؤ
    y_score = df['Final_Score']

    st.write('**X (Features)**')
    st.write(X_raw)
    st.write('**y_score (Target - Final Score)**')
    st.write(y_score.head())

with st.sidebar:
    st.header('Input Features')
    gender = st.selectbox('Gender', ('Female', 'Male'))
    attendance_score = st.slider('Attendance Score', 1, 5, 3)
    mid_exam_score = st.slider('Mid Exam Score', 0, 15, 10)
    lab_exam_score = st.slider('Lab Exam Score', 0, 15, 10)
    activity_score = st.slider('Activity Score', 0, 25, 10)

    # إعداد البيانات المدخلة
    data = {
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
encode_cols = ['Gender']
df_encoded = pd.get_dummies(input_student, columns=encode_cols, drop_first=True)

X_encoded = df_encoded[1:]
input_row = df_encoded.iloc[0:1]

# التحقق من القيم المفقودة
X_encoded = X_encoded.fillna(0)
input_row = input_row.fillna(0)

# التأكد من تطابق الأعمدة
X_encoded, input_row = X_encoded.align(input_row, join="inner", axis=1)

# تقسيم البيانات
X_train_score, X_test_score, y_train_score, y_test_score = train_test_split(X_encoded, y_score, test_size=0.2, random_state=42)

# تدريب النموذج للتنبؤ بالدرجة النهائية
score_model = RandomForestRegressor(random_state=42)
score_model.fit(X_train_score, y_train_score)

# التنبؤ بالدرجة النهائية
score_prediction = score_model.predict(input_row)

# عرض النتائج
st.subheader('Predicted Final Score')
st.success(f"Predicted Final Score (out of 40): {score_prediction[0]:.2f}")

# حساب المجموع الكلي
total_score = score_prediction[0] + (
    attendance_score * 2  # افتراض توزيع الوزن على المجموع الكلي
    + mid_exam_score
    + lab_exam_score
    + activity_score
)

st.subheader('Total Score')
st.success(f"Total Score (out of 100): {total_score:.2f}")
