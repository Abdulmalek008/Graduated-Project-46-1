import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# عنوان التطبيق
st.title('🎓 Student Final Exam Score Prediction App')

st.info('This app predicts the final exam score of students based on their performance scores in attendance, mid exam, lab exam, and activity.')

# تحميل البيانات
with st.expander('📊 Dataset'):
    df = pd.read_csv('https://raw.githubusercontent.com/Abdulmalek008/Graduated-Project-46-1/refs/heads/master/Student_Info%202.csv')
    st.write(df.head())

# عرض أسماء الأعمدة للتحقق
st.write("Column names in the dataset:", df.columns)

# معالجة البيانات
st.subheader('Data Preprocessing')

# بعد التحقق من الأعمدة الصحيحة، سنحدد المميزات والهدف
# تأكد من تعديل هذه الأعمدة وفقًا لما هو موجود في البيانات
X = df[['attendance', 'mid_exam', 'lab_exam', 'activity']]  # المميزات
y = df['final_score']  # الهدف

# تقسيم البيانات إلى مجموعة تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب نموذج RandomForest
st.subheader('Training the Model')
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# التنبؤ بالنتائج
st.subheader('Making Predictions')
y_pred = model.predict(X_test)

# حساب الدقة
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy of the model: {accuracy * 100:.2f}%')

# إدخال بيانات من المستخدم
st.subheader('Predict the Final Score')
attendance = st.slider('Attendance', 0, 100, 75)
mid_exam = st.slider('Mid Exam Score', 0, 100, 50)
lab_exam = st.slider('Lab Exam Score', 0, 100, 60)
activity = st.slider('Activity Score', 0, 100, 70)

# إجراء التنبؤ بناءً على المدخلات
user_input = np.array([[attendance, mid_exam, lab_exam, activity]])
prediction = model.predict(user_input)

st.write(f'The predicted final score is: {prediction[0]}')
