import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# عنوان التطبيق
st.title('🎓 Student Final Grade Prediction')

st.info('This app predicts the student’s final grade (A, B, C) based on their scores.')

# تحميل البيانات
with st.expander('📊 Dataset'):
    # قراءة البيانات
    df = pd.read_csv('https://raw.githubusercontent.com/Abdulmalek008/Graduated-Project-46-1/refs/heads/master/Student_Info%202.csv')
    
    # حذف عمود "Total"
    df.drop(columns=['Total'], inplace=True)
    
    # تصنيف الطلاب بناءً على الدرجات التفصيلية
    def classify_level(row):
        total_score = row['Mid_Exam_Score'] + row['Lab_Exam_Score'] + row['Activity_Score'] + row['Final_Score']
        if total_score > 80:
            return 'A'
        elif total_score >= 60:
            return 'B'
        else:
            return 'C'

    df['Level'] = df.apply(classify_level, axis=1)
    st.write('### Raw Data:')
    st.dataframe(df)

# تجهيز البيانات للتعلم الآلي
with st.expander('⚙️ Data Preparation'):
    st.write('### Features and Target:')
    
    # ترميز العمود النصي (Gender)
    df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    
    X = df_encoded.drop(columns=['Level', 'Student_ID'])
    y = df['Level']
    
    st.write('#### Features (X):')
    st.dataframe(X)
    st.write('#### Target (y):')
    st.dataframe(y)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# تدريب النموذج
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# واجهة المستخدم لتنبؤ درجة طالب جديد
with st.sidebar:
    st.header('🔍 Input Features')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    attendance_score = st.slider('Attendance Score', 1, 5, 3)
    mid_exam_score = st.slider('Mid Exam Score', 0, 15, 10)
    lab_exam_score = st.slider('Lab Exam Score', 0, 15, 10)
    activity_score = st.slider('Activity Score', 0, 25, 15)
    final_score = st.slider('Final Exam Score', 0, 40, 20)

# تجهيز البيانات للتنبؤ
new_data = pd.DataFrame({
    'Attendance_Score': [attendance_score],
    'Mid_Exam_Score': [mid_exam_score],
    'Lab_Exam_Score': [lab_exam_score],
    'Activity_Score': [activity_score],
    'Final_Score': [final_score],
    'Gender_Male': [1 if gender == 'Male' else 0]
})

# التأكد من أن الأعمدة في new_data تتطابق مع الأعمدة في X_train
new_data = new_data.reindex(columns=X.columns, fill_value=0)

# حساب مجموع الدرجات للتأكد من التنبؤ
total_score = attendance_score + mid_exam_score + lab_exam_score + activity_score + final_score
st.write(f"Total Score: {total_score}")

# تصنيف الطالب بناءً على المجموع الكلي
if total_score > 80:
    level = 'A'
elif total_score >= 60:
    level = 'B'
else:
    level = 'C'

# عرض التنبؤ
with st.expander('📈 Prediction Results'):
    st.write('### Predicted Level:')
    st.success(f'The predicted grade based on the total score is: **{level}**')
    
    # عرض النتيجة النهائية من النموذج
    prediction = model.predict(new_data)
    st.write(f"Model Prediction: {prediction[0]}")

# رسم بياني باستخدام st.scatter_chart
with st.expander('📊 Total Score vs Level'):
    st.write('### Distribution of Total Score by Level:')
    
    # حساب المجموع الكلي للدرجات
    df['Total_Score'] = df['Mid_Exam_Score'] + df['Lab_Exam_Score'] + df['Activity_Score'] + df['Final_Score']
    
    # إعداد البيانات للرسم البياني
    scatter_data = df[['Total_Score', 'Level']]
    
    # رسم بياني باستخدام st.scatter_chart
    st.scatter_chart(data=scatter_data, x='Total_Score', y='Level')
