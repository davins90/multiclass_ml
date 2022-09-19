import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import modules
import random

from scipy.stats import beta
from modules import ml_functions
import matplotlib.pyplot as plt

#################################
st.set_page_config(page_title="FHR Analysis - Prediction")

st.markdown("# Fetal Heart Rate Classification")
st.markdown("## Prediction")

# 1.0 Data retrieval
df = pd.read_pickle('../3_mod.pkl')

lbe = st.number_input("Insert a number of LBE, all other values for this demo are already saved",step=0.5)
st.write("LBE value: ",lbe)

sub = df.sample(1)
sub['LBE'] = lbe

if st.button("Click here for run prediction"):
    model1 = joblib.load('../src/app/model_classe_1.0.pkl')
    model2 = joblib.load('../src/app/model_classe_2.0.pkl')
    model3 = joblib.load('../src/app/model_classe_3.0.pkl')

    cols = ['LBE', 'LB', 'AC', 'FM', 'UC', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'DL',
           'DS', 'DP', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean',
           'Median', 'Variance', 'Tendency']

    sub['normal_score'] = 0
    sub['normal_score'] = model1.predict_proba(sub[cols])[:,1]

    sub['suspicious_score'] = 0
    sub['suspicious_score'] = model2.predict_proba(sub[cols])[:,1]

    sub['abnormal_score'] = 0
    sub['abnormal_score'] = model3.predict_proba(sub[cols])[:,1]

    sub[['normal_score','suspicious_score','abnormal_score']] = sub[['normal_score','suspicious_score','abnormal_score']]*100
    fig = px.bar(sub,y=["normal_score","suspicious_score","abnormal_score"],barmode='group',labels={'value':'Probability % '})
    st.plotly_chart(fig)