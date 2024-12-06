import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('   .  ')

st.error('. ')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://drive.google.com/file/d/1JmNRAb3bE7nlQ3ySztNvnEa0i_48dJgi/view?usp=drive_link')
  df

