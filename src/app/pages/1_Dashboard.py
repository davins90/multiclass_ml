import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import modules
import random

from scipy.stats import beta
from modules import ml_functions
import matplotlib.pyplot as plt

#################################
st.set_page_config(page_title="FHR Analysis - Dashboard")

st.markdown("# Fetal Heart Rate Classification")
st.markdown("## Dashboard")

# 1.0 Data retrieval
df = pd.read_pickle('../src/app/3_mod.pkl')
file = st.selectbox("Select file to visualize analysis",df.index)

# 2.0 Classification summary
sub = df[df.index==file]
sub = sub.rename(columns={'proba_classe_1.0':'normal_score','proba_classe_2.0':'suspicious_score','proba_classe_3.0':'abnormal_score'})
sub[['normal_score','suspicious_score','abnormal_score']] = sub[['normal_score','suspicious_score','abnormal_score']]*100
fig = px.bar(sub,y=["normal_score","suspicious_score","abnormal_score"],barmode='group',labels={'value':'Probability % '})
st.plotly_chart(fig)

# PDF
sub = sub.rename(columns={'proba_classe_1.0_beta':'normal_score_beta','proba_classe_2.0_beta':'suspicious_score_beta','proba_classe_3.0_beta':'abnormal_score_beta'})
xax = np.linspace(0,1.0,100)
sub2 = sub[["normal_score_beta","suspicious_score_beta","abnormal_score_beta"]]
sub4 = []
for i in sub2:
    sub3 = sub2[i].explode(i)
    sub4.append(sub3)
sub4 = pd.concat(sub4,axis=1)
# st.write(sub4.head(1))
fig2 = px.line(sub4,labels={'value':'PDF','index':'Score'})
fig2.update_xaxes(showgrid=False)
fig2.update_yaxes(showgrid=False)
st.plotly_chart(fig2)