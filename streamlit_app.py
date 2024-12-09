import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# عنوان التطبيق
st.title('🎓 Student Final Exam Score Prediction App')

st.info('This app predicts the final exam score of students based on their performance scores in attendance, mid exam, lab exam, and activity.')

# تحميل البيانات
with st.expander('📊 Dataset'):
    df = pd.read_csv('https://raw.githubusercontent.com/Abdulmalek008/Graduated-Project-46-1/refs/heads/master/Student_Info%202.csv')
    
    # حذف العمود "Total" غير المستخدم
    df.drop(columns=['Total'], inplace=True)
    
    # عرض البيانات
    st.write('### Raw Data:')
    st.dataframe(df)

# تجهيز البيانات للتعلم الآلي
with st.expander('⚙️ Data Preparation'):
    df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    
    X = df_encoded[['Attendance_Score', 'Mid_Exam_Score', 'Lab_Exam_Score', 'Activity_Score']]
    y = df['Final_Score']  # الهدف هو الفاينل سكور
    
    st.write('### Features (X):')
    st.dataframe(X)
    st.write('### Target (y):')
    st.dataframe(y)

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# تحسين البيانات باستخدام StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# استخدام RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# واجهة المستخدم
with st.sidebar:
    st.header('🔍 Enter Student Data:')
    attendance_score = st.slider('Attendance Score', 0, 5, 3)
    mid_exam_score = st.slider('Mid Exam Score', 0, 15, 10)
    lab_exam_score = st.slider('Lab Exam Score', 0, 15, 10)
    activity_score = st.slider('Activity Score', 0, 25, 15)

# تجهيز بيانات المستخدم للتنبؤ
new_data = np.array([[attendance_score, mid_exam_score, lab_exam_score, activity_score]])
new_data_scaled = scaler.transform(new_data)

# التنبؤ بدرجة الفاينل سكور
predicted_final_score = model.predict(new_data_scaled)[0]

# التأكد من أن الفاينل سكور لا يتجاوز 40 درجة
predicted_final_score = min(predicted_final_score, 40)

# حساب المجموع الكلي من الدرجات المدخلة
total_score = attendance_score + mid_exam_score + lab_exam_score + activity_score + predicted_final_score

# تصنيف الطالب بناءً على المجموع الكلي
if total_score >= 80:
    grade = 'A'
elif total_score >= 60:
    grade = 'B'
else:
    grade = 'C'

# عرض النتائج
st.write(f"### Predicted Final Exam Score: {predicted_final_score:.2f}")
st.write(f"### Total Score: {total_score:.2f}")
st.write(f"### Predicted Grade: {grade}")

# إنشاء جدول يعرض البيانات المدخلة والتنبؤ
input_data = {
    'Attendance Score': [attendance_score],
    'Mid Exam Score': [mid_exam_score],
    'Lab Exam Score': [lab_exam_score],
    'Activity Score': [activity_score],
    'Predicted Final Exam Score': [predicted_final_score],
    'Total Score': [total_score],
    'Predicted Grade': [grade]
}

input_df = pd.DataFrame(input_data)

# عرض الجدول للمستخدم
with st.expander('📊 Prediction Table'):
    st.write('### Entered Data and Predicted Grade:')
    st.dataframe(input_df)

# رسم بياني لتوزيع درجات الفاينل المتوقعة
with st.expander('📈 Prediction Distribution'):
    st.write('### Distribution of Predicted Final Exam Scores:')
    fig, ax = plt.subplots()
    ax.scatter(df['Attendance_Score'] + df['Mid_Exam_Score'] + df['Lab_Exam_Score'] + df['Activity_Score'], df['Final_Score'], color='blue', label='Actual Final Score')
    ax.scatter(total_score, predicted_final_score, color='red', label='Predicted Final Score', zorder=5)
    ax.set_xlabel('Total Performance (Attendance, Mid Exam, Lab Exam, Activity)')
    ax.set_ylabel('Final Exam Score')
    ax.legend()
    st.pyplot(fig)
