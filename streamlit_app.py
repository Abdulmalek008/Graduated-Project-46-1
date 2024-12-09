import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# عنوان التطبيق
st.title('🎓 Student Final Score Prediction App')

st.info('This app predicts the final score of students based on their performance scores.')

# تحميل البيانات
with st.expander('📊 Dataset'):
    # قراءة البيانات
    df = pd.read_csv('https://raw.githubusercontent.com/Abdulmalek008/Graduated-Project-46-1/refs/heads/master/Student_Info%202.csv')
    
    # حذف العمود غير المستخدم
    df.drop(columns=['Total'], inplace=True)
    
    # عرض البيانات
    st.write('### Raw Data:')
    st.dataframe(df)

# تجهيز البيانات للتعلم الآلي
with st.expander('⚙️ Data Preparation'):
    # ترميز عمود الجنس
    df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    
    # تحديد الميزات والهدف
    X = df_encoded.drop(columns=['Final_Score', 'Level', 'Student_ID'])  # استخدام Final_Score كهدف
    y = df['Final_Score']  # الهدف الآن هو Final_Score
    
    st.write('### Features (X):')
    st.dataframe(X)
    st.write('### Target (y):')
    st.dataframe(y)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# تدريب النموذج
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# تقييم النموذج
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
st.write(f"### Mean Absolute Error (MAE): {mae:.2f}")

# واجهة المستخدم
with st.sidebar:
    st.header('🔍 Enter Student Data:')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    attendance_score = st.slider('Attendance Score', 0, 5, 3)
    mid_exam_score = st.slider('Mid Exam Score', 0, 15, 10)
    lab_exam_score = st.slider('Lab Exam Score', 0, 15, 10)
    activity_score = st.slider('Activity Score', 0, 25, 15)

# التأكد أن مجموع الدرجات اليدوية لا يتجاوز 60
manual_score = attendance_score + mid_exam_score + lab_exam_score + activity_score
if manual_score == 60:
    st.warning(f"Note: The sum of manual scores (attendance, mid exam, lab exam, and activity) is exactly 60. Current sum: {manual_score}")
else:
    st.write(f"Total of manual scores: {manual_score} (does not need to be exactly 60)")

# الدرجة النهائية (Final Exam Score) ستكون من 40
final_exam_score = st.slider('Final Exam Score', 0, 40, 20)

# تجهيز بيانات المستخدم
new_data = pd.DataFrame({
    'Attendance_Score': [attendance_score],
    'Mid_Exam_Score': [mid_exam_score],
    'Lab_Exam_Score': [lab_exam_score],
    'Activity_Score': [activity_score],
    'Final_Score': [final_exam_score],
    'Gender_Male': [1 if gender == 'Male' else 0]
})

# التأكد من تطابق الأعمدة مع النموذج
new_data = new_data.reindex(columns=X.columns, fill_value=0)

# التنبؤ بالدرجة النهائية
predicted_score = model.predict(new_data)

# عرض النتيجة للمستخدم
st.write(f"The predicted final score for the student is: **{predicted_score[0]:.2f}**")

# عرض الجدول مع البيانات المدخلة والدرجة المتوقعة
input_data = {
    'Gender': [gender],
    'Attendance Score': [attendance_score],
    'Mid Exam Score': [mid_exam_score],
    'Lab Exam Score': [lab_exam_score],
    'Activity Score': [activity_score],
    'Final Exam Score': [final_exam_score],
    'Predicted Final Score': [predicted_score[0]]
}

input_df = pd.DataFrame(input_data)

with st.expander('📊 Prediction Table'):
    st.write('### Entered Data and Predicted Final Score:')
    st.dataframe(input_df)

# رسم بياني يوضح توزيع الدرجات النهائية
with st.expander('📊 Final Score Distribution'):
    st.write('### Distribution of Final Scores:')
    st.bar_chart(df['Final_Score'])
