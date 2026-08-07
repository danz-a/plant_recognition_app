import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Plant Recognition")
st.sidebar.title("Table of contents")
pages=["Problem", "Data Exploration", "Preprocessing", "Modelling", "Interpretability"]
page=st.sidebar.radio("Go to", pages)

if page == pages[0]:
  st.write("Problem")
if page == pages[1]:
  st.write("Data Exploration")
if page == pages[2]:
  st.write("Preprocessing")
if page == pages[3]:
  st.write("Modelling")
if page == pages[4]:
  st.write("Interpretability")
