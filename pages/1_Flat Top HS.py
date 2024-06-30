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

    st.title(f'[Super Gaussian | Flat Top Heat Source]({url3})')
    cm1, cm2 = st.columns([0.2,0.8])

    def plot_gaussian_heat_distribution(A, Ca, Cb, k, P, eta, r_0, i):
        r0 = r_0*1.0e-6 ## converting to micrometer
        cmaps = ['balance', 'bluered', 'hsv', 'jet', 'picnic', 'portland', 'rainbow', 'rdylbu_r', 'spectral_r', 'turbo']
        x = np.linspace(-100e-6, 100e-6, 100)
        y = np.linspace(-100e-6, 100e-6, 100)
        x, y = np.meshgrid(x, y)

        r = (x**2 + y**2)**0.5
        F = np.where(r_0 - r < 0, 0, 1)
        Q = F*((Ca**(1/k)*k*P*eta*A)/(np.pi*r0**2*math.gamma(1/k)))*(np.exp(-Cb*(r**2/r0**2)**k))
        
        cm1.write('Colormap: '+cmaps[i])
        cm1.write(r'$\Gamma(1/k) = $ '+f'{math.gamma(1/k):.3f}')
        cm1.write(r'$Q_{peak} =$  '+f'{np.max(Q):.3e}'+r'  $W/m^2$')

        camera = dict(eye=dict(x=1.5, y=1.5, z=1.5))
        fig = go.Figure(data=[go.Surface(z=Q, x=x, y=y, colorscale=cmaps[i])])
        fig.update_layout(scene_camera=camera, scene=dict(xaxis_title='X-axis', yaxis_title='Y-axis', zaxis_title='Intensity'), width=2000, height=1000) 
        fig.update_traces(colorbar=dict(title=r'Q W/m^2'), colorbar_title_font=dict(size=30, color='black'), colorbar_exponentformat='B', colorbar_nticks=6, colorbar_len=0.5, colorbar_borderwidth=0.0, colorbar_thickness=70, colorbar_orientation='v', colorbar_tickfont=dict(family='Sans', weight='bold', color='black', size=25))
        return fig, Q

    cm1.header('Parameters')
    power = cm1.slider(r'''Power $$(P)$$''', min_value=1, max_value=500, value=250, step=1)
    eta = cm1.slider(r'''Efficiency $$(\eta)$$ ''', min_value=0.0, max_value=1.0, value=0.9, step=0.01)
    beam_radius = cm1.slider(r'''Beam Radius $$(r_G$$ $$\mu m)$$''', min_value=10.0, max_value=75.0, value=50.0, step=0.1)
    A = cm1.slider(r'''Absorptivity $$(A)$$''', min_value=0.00001, max_value=5.0, value=1.0, step=0.1)
    Ca = cm1.slider(r'''Constant $$(C_a)$$''', min_value=0.0000001, max_value=4.0, value=2.0, step=0.1)
    Cb = cm1.slider(r'''Constant $$(C_b)$$''', min_value=0.0000001, max_value=4.0, value=2.0, step=0.1)
    k = cm1.slider(r'''Constant $$(k)$$''', min_value=0.0000001, max_value=10.0, value=4.2, step=0.1)
    i = cm1.slider('colormap', min_value=0, max_value=9, value=6, step=1)

    fig, Q = plot_gaussian_heat_distribution(A, Ca, Cb, k, power, eta, beam_radius, i)

    cm3, cm4 = cm2.columns([0.5,0.5])
    cm3.header(r'$Q =  \frac{(C_a)^{1/k}kAP\eta}{\pi r_o^2 \Gamma(1/k)} \exp\left[-C(\frac{r^2}{r_o^2})^{k}\right]$')
    cm4.header(r'$\Gamma(1/k) =  \int_0^\infty t^{\frac{1}{k}-1} e^{-t} dt$')
    # cm2.header(r'$Q_{peak} =$  '+f'{np.max(Q):.3e}'+r'  $W/m^2$')

    cm2.plotly_chart(fig, use_container_width=True)
    st.divider()

__main__()