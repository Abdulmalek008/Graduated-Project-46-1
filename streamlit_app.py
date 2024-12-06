
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Machine Learning App')

st.info('This is app builds a machine learning model!')

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

with st.expander('Data visualization'):
 st.scatter_chart(data=df , x='Attendance_Score', y='Total', color='Level') 

with st.sidebar:
  st.header('Input features')
  student_ID = st.selectbox('student_ID' , ('S001','S002','S003','S004','S005','S006','S007','S008','S009','S010','S011','S012','S013','S014','S015','S016','S017','S018','S019','S020','S021','S022','S023','S024','S025','S026','S027','S028','S029','S030','S031','S032','S033','S034','S035','S036','S037','S038','S039','S040','S041','S042','S043','S044','S045','S046','S047','S048','S049','S050','S051','S052','S053','S054','S055','S056','S057','S058','S059','S060','S061','S062','S063','S064','S065','S066','S067','S068','S069','S070','S071','S072','S073','S074','S075','S076','S077','S078','S079','S080','S081','S082','S083','S084','S085','S086','S087','S088','S089','S090','S091','S092','S093','S094','S095','S096','S097','S098','S099','S100','S101','S102','S103','S104','S105','S106','S107','S108','S109','S110','S111','S112','S113','S114','S115','S116','S117','S118','S119','S120','S121','S122','S123','S124','S125','S126','S127','S128','S129','S130','S131','S132','S133','S134','S135','S136','S137','S138','S139','S140','S141','S142','S143','S144','S145','S146','S147','S148','S149','S150'))
  

  gender = st.selectbox('Gender',('Female','Male'))
  
  attendance_score = st.slider('attendance_score' , 1 , 5 , 3)
  mid_exam_score = st.slider('mid_exam_score' , 0 , 15 , 10)
  lab_exam_score = st.slider('lab_exam_score' , 0 , 15 , 10)
  activity_score = st.slider('activity_score' , 0 , 25 , 10)
  final_score = st.slider('final_score' , 0 , 40, 20)

  data = {'Student_ID': student_ID,
        'Gender': gender,
        'Attendance_Score': attendance_score,
        'Mid_Exam_Score': mid_exam_score,
        'Lab_Exam_Score': lab_exam_score,
        'Activity_Score': activity_score,
        'Final_Score': final_score}
  input_df = pd.DataFrame(data, index=[0])
  input_student = pd.concat([input_df, X_raw], axis=0)

with st.expander('input features'):
  st.write('**input student**')
  input_df
  st.write('**Comined student data**')
  input_student
#encode x
encode = ['Student_ID', 'Gender']
df_student = pd.get_dummies(input_student, prefix=encode)

X = df_student[1:]
input_row = df_student[:1]

#encode y
target_mapper = {'A': 0,
                 'B': 1,
                 'C': 2}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)


with st.expander('Data preparation'):  
  st.write('**Encode X (input student)**')
  input_row
  st.write('**Encode y**')
  y



clf = RandomForestClassifier()
clf.fit(X, y)

predicyion = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['A', 'B', 'C']
df_prediction_proba.rename(columns={0: 'A',
                                    1: 'B',
                                    2: 'C'})

                                  
st.subheader('Predicted Level')
st.dataframe(df_prediction_proba,
             column_config={
               'A':st.column_config.ProgressColumn(
                 'A',
                 format='%i',
                 width='large',
                 min_value=80,
                 max_value=100
               ),
               'B':st.column_config.ProgressColumn(
                 'B',
                 format='%i',
                 width='large',
                 min_value=70,
                 max_value=80
               ),
               'C':st.column_config.ProgressColumn(
                 'C',
                 format='%i',
                 width='large',
                 min_value=0,
                 max_value=60
               ),
             }, hide_index=True)

student_level = np.array(['A', 'B', 'C',])
st.success(str(student_level[Prediction][0]))
                 




        
  
  
  
  
