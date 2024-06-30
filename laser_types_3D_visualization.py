import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math

st.set_page_config(layout="wide")

inline_css = """div[data-testid="stExpander"] div[role="button"] p {font-size: 3rem;}"""
st.markdown(f"<style>{inline_css}</style>", unsafe_allow_html=True)

url1 = 'https://link.springer.com/article/10.1007/s11837-023-06363-8' # Bessel, Ring
url2 = 'https://doi.org/10.1080/17445302.2014.937059' # Double Ellipsoide
url3 = 'https://doi.org/10.1080/17452759.2024.2308513' # Super Gaussian
url4 = 'https://doi.org/10.1016/j.matpr.2020.12.842' # Volumetric Gaussian
url5 = 'https://doi.org/10.1016/j.jmatprotec.2018.03.011' # Super Gaussian Eqn 3
url6 = 'https://www.sciencedirect.com/science/article/pii/S0079672714000317' # Bessel with multiple ring
st.write(f'Reference for Equations used in visualization of laser heat soruces: [Gaussian]({url3}),[Flat Top]({url3}) [| Super Gaussian]({url5}), [Double ]({url2}) [Ellipsoide]({url4}), [Ring]({url1}), [Bessel]({url6})')

