import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# عنوان التطبيق
st.title('🎓 Student Grade Prediction App')

st.info('This app predicts the final grade (A, B, C) of students based on their performance scores.')

# تحميل البيانات
with st.expander('📊 Dataset'):
    # قراءة البيانات
    df = pd.read_csv('https://raw.githubusercontent.com/Abdulmalek008/Graduated-Project-46-1/refs/heads/master/Student_Info%202.csv')
    
    # حذف العمود غير المستخدم
    df.drop(columns=['Total'], inplace=True)
    
    # إضافة عمود الدرجات النهائية
    def calculate_level(row):
        total_score = row['Mid_Exam_Score'] + row['Lab_Exam_Score'] + row['Activity_Score'] + row['Final_Score']
        if total_score >= 80:
            return 'A'
        elif total_score >= 60:
            return 'B'
        else:
            return 'C'

    df['Level'] = df.apply(calculate_level, axis=1)
    st.write('### Raw Data:')
    st.dataframe(df)

# تجهيز البيانات للتعلم الآلي
with st.expander('⚙️ Data Preparation'):
    # ترميز عمود الجنس
    df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    
    # تحديد الميزات والهدف
    X = df_encoded.drop(columns=['Level', 'Student_ID'])
    y = df['Level']
    
    st.write('### Features (X):')
    st.dataframe(X)
    st.write('### Target (y):')
    st.dataframe(y)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# تدريب النموذج
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# واجهة المستخدم
with st.sidebar:
    st.header('🔍 Enter Student Data:')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    mid_exam_score = st.slider('Mid Exam Score', 0, 15, 10)
    lab_exam_score = st.slider('Lab Exam Score', 0, 15, 10)
    activity_score = st.slider('Activity Score', 0, 25, 15)
    final_exam_score = st.slider('Final Exam Score', 0, 40, 20)

# تجهيز بيانات المستخدم
new_data = pd.DataFrame({
    'Mid_Exam_Score': [mid_exam_score],
    'Lab_Exam_Score': [lab_exam_score],
    'Activity_Score': [activity_score],
    'Final_Score': [final_exam_score],
    'Gender_Male': [1 if gender == 'Male' else 0]
})

# التأكد من تطابق الأعمدة مع النموذج
new_data = new_data.reindex(columns=X.columns, fill_value=0)

# تنبؤ النموذج
predicted_level = model.predict(new_data)[0]

# عرض النتائج
with st.expander('📈 Prediction Results'):
    st.write('### Predicted Grade:')
    st.success(f'The predicted grade for the student is: **{predicted_level}**')

# رسم بياني يوضح توزيع المستويات
with st.expander('📊 Grade Distribution by Total Score'):
    df['Total_Score'] = df['Mid_Exam_Score'] + df['Lab_Exam_Score'] + df['Activity_Score'] + df['Final_Score']
    st.write('### Distribution of Total Scores by Grade:')
    scatter_data = df[['Total_Score', 'Level']]
    st.scatter_chart(scatter_data, x='Total_Score', y='Level')
