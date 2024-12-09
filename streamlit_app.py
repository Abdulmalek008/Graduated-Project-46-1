import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# عنوان التطبيق
st.title('🎓 Student Grade Prediction App')

st.info('This app predicts the final grade (A, B, C) of students and estimates their Final Exam Score.')

# تحميل البيانات
with st.expander('📊 Dataset'):
    # قراءة البيانات
    df = pd.read_csv('https://raw.githubusercontent.com/Abdulmalek008/Graduated-Project-46-1/refs/heads/master/Student_Info%202.csv')
    
    # حذف العمود غير المستخدم
    df.drop(columns=['Total'], inplace=True)
    
    # إضافة عمود الدرجات النهائية
    def calculate_level(row):
        total_score = row['Attendance_Score'] + row['Mid_Exam_Score'] + row['Lab_Exam_Score'] + row['Activity_Score']
        if total_score >= 80:
            return 'A'
        elif total_score >= 60:
            return 'B'
        else:
            return 'C'

    df['Level'] = df.apply(calculate_level, axis=1)
    st.write('### Raw Data:')
    st.dataframe(df)

# تحليل البيانات
with st.expander('📊 Data Analysis'):
    st.write('### Correlation Matrix:')
    # حذف الأعمدة غير الرقمية
    numeric_df = df.select_dtypes(include=[np.number])
    # عرض مصفوفة الارتباط
    st.write(numeric_df.corr())
    
    st.write('### Pairplot (relationship between features):')
    st.line_chart(numeric_df)

# تجهيز البيانات للتعلم الآلي
with st.expander('⚙️ Data Preparation'):
    st.write('### Features and Target:')
    
    # ترميز عمود الجنس
    df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    
    # تحديد الميزات والهدف
    X = df_encoded.drop(columns=['Level', 'Student_ID', 'Final_Score'])
    y = df['Final_Score']
    
    st.write('### Features (X):')
    st.dataframe(X)
    st.write('### Target (y):')
    st.dataframe(y)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# تدريب النموذج لتنبؤ درجة الامتحان النهائي
final_score_model = RandomForestClassifier(random_state=42)
final_score_model.fit(X_train, y_train)

# واجهة المستخدم
with st.sidebar:
    st.header('🔍 Enter Student Data:')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    attendance_score = st.slider('Attendance Score', 0, 5, 3)
    mid_exam_score = st.slider('Mid Exam Score', 0, 15, 10)
    lab_exam_score = st.slider('Lab Exam Score', 0, 15, 10)
    activity_score = st.slider('Activity Score', 0, 25, 15)

# تجهيز بيانات المستخدم
new_data = pd.DataFrame({
    'Attendance_Score': [attendance_score],
    'Mid_Exam_Score': [mid_exam_score],
    'Lab_Exam_Score': [lab_exam_score],
    'Activity_Score': [activity_score],
    'Gender_Male': [1 if gender == 'Male' else 0]
})

# التأكد من تطابق الأعمدة مع النموذج
new_data = new_data.reindex(columns=X.columns, fill_value=0)

# تنبؤ درجة الامتحان النهائي
predicted_final_score = final_score_model.predict(new_data)[0]

# حساب المجموع الكلي
total_score = attendance_score + mid_exam_score + lab_exam_score + activity_score + predicted_final_score

# التنبؤ بالمستوى بناءً على المجموع الكلي
if total_score >= 80:
    level = 'A'
elif total_score >= 60:
    level = 'B'
else:
    level = 'C'

# إنشاء جدول يعرض البيانات المدخلة والتنبؤ
input_data = {
    'Gender': [gender],
    'Attendance Score': [attendance_score],
    'Mid Exam Score': [mid_exam_score],
    'Lab Exam Score': [lab_exam_score],
    'Activity Score': [activity_score],
    'Predicted Final Score': [predicted_final_score],
    'Total Score': [total_score],
    'Predicted Level': [level]
}

input_df = pd.DataFrame(input_data)

# عرض الجدول للمستخدم
with st.expander('📊 Prediction Table'):
    st.write('### Entered Data and Predicted Results:')
    st.dataframe(input_df)

# عرض النتيجة النهائية
with st.expander('📈 Prediction Results'):
    st.write('### Predicted Final Exam Score:')
    st.success(f'The predicted final exam score is: **{predicted_final_score:.2f}**')
    st.write('### Predicted Grade:')
    st.success(f'The predicted grade based on the total score is: **{level}**')
