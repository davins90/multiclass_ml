import streamlit as st
import pandas as pd
import numpy as np
import modules
import random

from scipy.stats import beta
from modules import ml_functions
import matplotlib.pyplot as plt

#################################
st.set_page_config(page_title="FHR Analysis - Home")

st.markdown("# Fetal Heart Rate Classification")
st.markdown("## Intro")

st.sidebar.success("Select a demo above")

st.markdown("The goal of this tool is twofold: \n - To provide a dashboard to visualize the results of past analysis. \n - To provide a tool that can generate a real-time prediction of fetal cardiological health status. \n Based on the classification provided by the NICHD, 3 states in which the fetus can be found were classified: \n - 1 = normal \n - 2 = suspicious \n- 3 = abnormal/pathological")
