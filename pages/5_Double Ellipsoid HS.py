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


def __main__():

    st.title(f'[Double]({url2}) [Ellipsoide HS]({url4})')
    cm1, cm2 = st.columns([0.2,0.8])

    def plot_double_ellipsoide_heat_source(P, eta, a_f_DEsG, a_r_DEsG, b_DEsG, f_f, f_r, A, Ca, Cb, i):
        a_f, a_r, b = a_f_DEsG*1e-6, a_r_DEsG*1e-6, b_DEsG*1e-6,  # scaling unit to  meter

        cmaps = ['balance', 'bluered', 'hsv', 'jet', 'picnic', 'portland', 'rainbow', 'rdylbu_r', 'spectral_r', 'turbo']

        x = np.linspace(-100e-6, 100e-6, 100)
        y = np.linspace(-100e-6, 100e-6, 100)
        x, y= np.meshgrid(x, y)

        r = (x**2 + y**2)**0.5
        F = np.where((a_f_DEsG+a_r_DEsG) - r < 0, 0, 1)
        Qf  =  F*((A*f_f*eta*P*Ca)/(a_f*b*(np.pi)**1.5))*np.exp(-Cb*(x**2/a_f**2)-Cb*(y**2/b**2)) # For pure Double Ellipsoide
        Qr  =  F*((A*f_r*eta*P*Ca)/(a_r*b*(np.pi)**1.5))*np.exp(-Cb*(x**2/a_r**2)-Cb*(y**2/b**2)) # For pure Double Ellipsoide
        Q = Qf+Qr

        cm1.write('Cmap: '+cmaps[i])
        cm1.write(r'$Q_{peak} =$  '+f'{np.max(Q):.3e}'+r'  $W/m^2$')

        camera = dict(eye=dict(x=1.5, y=0.5, z=1.5))
        fig = go.Figure(data=[go.Surface(z=Q, x=x, y=y, colorscale=cmaps[i])])
        fig.update_layout(scene_camera=camera, scene=dict(xaxis_title='X-axis', yaxis_title='Y-axis', zaxis_title='Intensity'), width=2000, height=1000) 
        fig.update_traces(colorbar=dict(title=r'Q W/m^2'), colorbar_title_font=dict(size=30, color='black'), colorbar_exponentformat='B', colorbar_nticks=6, colorbar_len=0.5, colorbar_borderwidth=0.0, colorbar_thickness=70, colorbar_orientation='v', colorbar_tickfont=dict(family='Sans', weight='bold', color='black', size=25))
        return fig

    cm1.header('Parameters')
    power = cm1.slider(r'''Power $$(P)$$''', min_value=1, max_value=500, value=250, step=1)
    eta = cm1.slider(r'''Efficiency $$(\eta)$$ ''', min_value=0.0, max_value=1.0, value=0.9, step=0.1)
    a_f = cm1.slider(r'$a_f \, \, [\mu m]$', min_value=1.0, max_value=75.0, value=50.0, step=0.1)
    a_r = cm1.slider(r'$a_r \, \, [\mu m]$', min_value=1.0, max_value=75.0, value=15.0, step=0.1)
    b = cm1.slider(r'$b \,\, [\mu m]$', min_value=1.0, max_value=50.0, value=35.0, step=0.1)

    A = cm1.slider(r'''Absorptivity $$(A)$$''', min_value=0.00001, max_value=5.0, value=1.0, step=0.1)
    # Ca = cm1.slider(r'''Constant $$(C_a)$$''', min_value=0.0000001, max_value=4.0, value=2.0, step=0.1)
    Cb = cm1.slider(r'''Constant $$(C_b)$$''', min_value=0.0000001, max_value=4.0, value=3.0, step=0.1)
    k = cm1.slider(r'''Constant $$(k)$$''', min_value=0.0000001, max_value=10.0, value=4.2, step=0.1)
    i = cm1.slider('colormap', min_value=0, max_value=9, value=6, step=1)

    Ca = 6.0*np.sqrt(3)
    f_f = 2*a_f/(a_f+a_r)
    f_r = 2*a_r/(a_f+a_r)

    fig = plot_double_ellipsoide_heat_source(power, eta, a_f, a_r, b, f_f, f_r, A, Ca, Cb, i )

    cm3, cm4, cm5 = cm2.columns([0.2,0.4, 0.4])
    cm3.subheader(r'$Q = Q_f + Q_r $')
    cm4.subheader(r'''$$Q_f =  \frac{C_a AP\eta f_f}{a_f b \pi \sqrt{\pi} } \exp\left[-C_b\left(\frac{x^2}{a_f^2} + \frac{y^2}{b^2}\right)\right]   $$''')
    cm5.subheader(r'''$$Q_r =  \frac{C_a AP\eta f_r}{a_r b \pi \sqrt{\pi} } \exp\left[-C_b\left(\frac{x^2}{a_r^2} + \frac{y^2}{b^2}\right)\right]   $$''')
    cm3.subheader(r'$f_{f|r} = \frac{2a_{f|r}}{a_f + a_r}$')
    cm4.write(r'$C_a = 6\sqrt{3}$')

    cm2.plotly_chart(fig, use_container_width=True)
    st.divider()

__main__()