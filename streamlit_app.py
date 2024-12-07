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
    df

    st.write('**X**')
    X_raw = df.drop('Level', axis=1)
    X_raw

    st.write('**y**')
    y_raw = df.Level
    y_raw

# عرض البيانات باستخدام scatter chart
with st.expander('Data visualization'):
    st.scatter_chart(data=df, x='Attendance_Score', y='Total', color='Level')

# واجهة المستخدم للمدخلات
with st.sidebar:
    st.header('Input features')
    student_ID = st.selectbox('Student_ID', [f"S{str(i).zfill(3)}" for i in range(1, 151)])
    gender = st.selectbox('Gender', ('Female', 'Male'))
    attendance_score = st.slider('Attendance_Score', 1, 5, 3)
    mid_exam_score = st.slider('Mid_Exam_Score', 0, 15, 10)
    lab_exam_score = st.slider('Lab_Exam_Score', 0, 15, 10)
    activity_score = st.slider('Activity_Score', 0, 25, 10)
    final_score = st.slider('Final_Score', 0, 40, 20)

    # حساب مجموع الدرجات
    total_score = mid_exam_score + lab_exam_score + activity_score + final_score

    # إنشاء بيانات الطالب المدخلة
    data = {
        'Student_ID': student_ID,
        'Gender': gender,
        'Attendance_Score': attendance_score,
        'Mid_Exam_Score': mid_exam_score,
        'Lab_Exam_Score': lab_exam_score,
        'Activity_Score': activity_score,
        'Final_Score': final_score,
        'Total': total_score,
    }
    input_df = pd.DataFrame(data, index=[0])
    input_student = pd.concat([input_df, X_raw], axis=0)

# دالة لتصنيف المجموع (Total)
def assign_grade(total):
    if total > 80:
        return 'A'
    elif 60 <= total <= 80:
        return 'B'
    else:
        return 'C'

# تصنيف الطالب بناءً على `Total`
input_df['Grade'] = input_df['Total'].apply(assign_grade)

# تحويل البيانات إلى صيغة يمكن استخدامها مع النموذج
encode = ['Student_ID', 'Gender']
df_student = pd.get_dummies(input_student, prefix=encode)

X = df_student[1:]
input_row = df_student[:1]

# تحويل y إلى قيم رقمية
target_mapper = {'A': 0, 'B': 1, 'C': 2}
def target_encode(val):
    return target_mapper[val]

y = y_raw.apply(target_encode)

# تدريب النموذج
clf = RandomForestClassifier()
clf.fit(X, y)

# التنبؤ باستخدام النموذج
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

# إنشاء جدول لاحتمالات التنبؤ
df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['A', 'B', 'C']
df_prediction_proba.rename(columns={0: 'A', 1: 'B', 2: 'C'})

# عرض النتائج
st.subheader('Predicted Level (using model)')
st.dataframe(
    df_prediction_proba,
    column_config={
        'A': st.column_config.ProgressColumn('A', format='%f', width='medium', min_value=0, max_value=1),
        'B': st.column_config.ProgressColumn('B', format='%f', width='medium', min_value=0, max_value=1),
        'C': st.column_config.ProgressColumn('C', format='%f', width='medium', min_value=0, max_value=1),
    },
    hide_index=True,
)

student_level = np.array(['A', 'B', 'C'])
st.success(f"Predicted Level (from model): {student_level[prediction][0]}")

# عرض التصنيف بناءً على المجموع
st.subheader('Manual Grade Classification')
manual_grade = input_df['Grade'].iloc[0]
st.info(f"The grade based on the total score ({total_score}) is: {manual_grade}")
