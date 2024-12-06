import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('   .  ')

st.error('. ')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://github.com/Abdulmalek008/Graduated-Project-46-1/blob/master/Student_Info%202.csv#start-of-content')
  df

