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

    st.title(f'[Bessel Heat Source]({url1})')
    cm1, cm2 = st.columns([0.2,0.8])

    def plot_bessel_heat_distribution(a0, a1, a2, P, eta, r_g, rr_1, r_t1, rr_2, r_t2, rr_3, r_t3, k, A, C1, C2, i ):
        rg, rr1, rt1, rr2, rt2, rr3, rt3 = r_g*1.0e-6, rr_1*1.0e-6, r_t1*1.0e-6, rr_2*1.0e-6, r_t2*1.0e-6, rr_3*1.0e-6, r_t3*1.0e-6   # scaling unit to meter
        
        a3 = 1-a0-a1-a2
        cmaps = ['balance', 'bluered', 'hsv', 'jet', 'picnic', 'portland', 'rainbow', 'rdylbu_r', 'spectral_r', 'turbo']
        
        x = np.linspace(-100e-6, 100e-6, 200)
        y = np.linspace(-100e-6, 100e-6, 200)
        x, y= np.meshgrid(x, y)

        r = (x**2 + y**2)**0.5
        F = np.where(rr3 + rt3 - r < 0, 0, 1)
        Y1 = np.exp(-rr1**2/(2*rt1**2)) + (rr1/rt1)*(np.pi/2)**0.5 * math.erfc(-rr1/(rt1*2**0.5))
        Y2 = np.exp(-rr2**2/(2*rt2**2)) + (rr2/rt2)*(np.pi/2)**0.5 * math.erfc(-rr2/(rt2*2**0.5))
        Y3 = np.exp(-rr3**2/(2*rt3**2)) + (rr3/rt3)*(np.pi/2)**0.5 * math.erfc(-rr3/(rt3*2**0.5))
     
        Q1 = F*((C1**(1/k)*k*eta*A*a0*P)/(np.pi*rg**2*math.gamma(1/k)))*(np.exp(-C2*(r**2/rg**2)**k))
        Q2 = F*((C1*eta*A*a1*P)/(np.pi*rt1**2*Y1)) * (np.exp(-C2*((r-rr1)**2/(rt1**2))))
        Q3 = F*((C1*eta*A*a2*P)/(np.pi*rt2**2*Y2)) * (np.exp(-C2*((r-rr2)**2/(rt2**2))))
        Q4 = F*((C1*eta*A*a3*P)/(np.pi*rt3**2*Y3)) * (np.exp(-C2*((r-rr3)**2/(rt3**2))))
        Q = Q1 + Q2 + Q3 + Q4

        cm1.write('Cmap: '+cmaps[i])
        cm1.write(r'$a_0 + a_1 + a_2 + a_3 = 1$')
        cm1.write(r'Q$_G^{peak}$: '+f'{np.max(Q1):.3e}'+r'  $W/m^2$')
        cm1.write(r'Q$_{R1}^{peak}$: '+f'{np.max(Q2):.3e}'+r'  $W/m^2$')
        cm1.write(r'Q$_{R2}^{peak}$: '+f'{np.max(Q3):.3e}'+r'  $W/m^2$')
        cm1.write(r'Q$_{R3}^{peak}$: '+f'{np.max(Q4):.3e}'+r'  $W/m^2$')

        camera = dict(eye=dict(x=1.5, y=0.5, z=1.5))
        fig = go.Figure(data=[go.Surface(z=Q, x=x, y=y, colorscale=cmaps[i], )])
        fig.update_layout(scene_camera=camera, scene=dict(xaxis_title='X-axis', yaxis_title='Y-axis', zaxis_title='Intensity'), width=2000, height=1000) 
        fig.update_traces(colorbar=dict(title=r'Q W/m^2'), colorbar_title_font=dict(size=30, color='black'), colorbar_exponentformat='B', colorbar_nticks=6, colorbar_len=0.5, colorbar_borderwidth=0.0, colorbar_thickness=70, colorbar_orientation='v', colorbar_tickfont=dict(family='Sans', weight='bold', color='black', size=25))
        
        return fig


    cm1.header('Parameters')

    P_Bessel = cm1.slider(r'Power [P]',                                                    min_value=1,        max_value=1000,     value=250,     step=1)
    eta_Bessel = cm1.slider(r'Efficiency $[\eta]$',                                        min_value=0.0,      max_value=1.0,      value=0.9,     step=0.0001)
    Bessel_gaussian_radius = cm1.slider(r'''Gaussian Beam Radius $$(r_G$$ $$\mu m)$$''',   min_value=1.0,      max_value=75.0,     value=20.0,    step=0.1)
    BRing1Radius = cm1.slider(r'''Ring 1 radius $$(Rr_1$$ $$\mu m)$$''',                   min_value=1.0,     max_value=100.0,    value=35.0,    step=0.1)
    Ring1_HalfThickness = cm1.slider(r'''Ring 1 half thickness $$(r_{t1}$$ $$\mu m)$$''',  min_value=1.0,      max_value=50.0,     value=5.0,     step=0.01)
    BRing2Radius = cm1.slider(r'''Ring 2 radius $$(Rr_2$$ $$\mu m)$$''',                   min_value=1.0,      max_value=100.0,    value=43.0,    step=0.1)
    Ring2_HalfThickness = cm1.slider(r'''Ring 2 half thickness $$(r_{t2}$$ $$\mu m)$$''',  min_value=1.0,      max_value=25.0,     value=3.0,     step=0.01)
    BRing3Radius = cm1.slider(r'''Ring 3 radius $$(Rr_3$$ $$\mu m)$$''',                   min_value=1.0,      max_value=100.0,    value=48.0,    step=0.1)
    Ring3_HalfThickness = cm1.slider(r'''Ring 3 half thickness $$(r_{t3}$$ $$\mu m)$$''',  min_value=1.0,      max_value=15.0,     value=2.0,     step=0.01)
    

    a1 = cm1.slider(r'P$_{G} \, proportion \, \, (a_0)$',    min_value=0.0, max_value=1.0, value=0.72, step=0.01)
    a2 = cm1.slider(r'P$_{R1} \, proportion \, \, (a_1)$', min_value=0.0, max_value=a1,  value=0.16, step=0.01)
    a3 = cm1.slider(r'P$_{R2} \, proportion \, \, (a_2)$', min_value=0.0, max_value=a2,  value=0.08, step=0.01)
    
    k_Bessel = cm1.slider(r'''SGaussian Order $$(k )$$''', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    A_Bessel = cm1.slider(r'''Absorptivity $$(A )$$''', min_value=0.00001, max_value=5.0, value=1.0, step=0.1)
    C1_Bessel = cm1.slider(r'''Constant $$(C_1)$$''', min_value=0.0000001, max_value=4.0, value=2.0, step=0.1)
    C2_Bessel = cm1.slider(r'''Constant $$(C_2)$$''', min_value=0.0000001, max_value=4.0, value=2.0, step=0.1)
    i_Bessel = cm1.slider('Cmap  ', min_value=0, max_value=9, value=7, step=1)

    fig_Bessel = plot_bessel_heat_distribution(a1, a2, a3, P_Bessel, eta_Bessel, Bessel_gaussian_radius, BRing1Radius, Ring1_HalfThickness, BRing2Radius, Ring2_HalfThickness, BRing3Radius, Ring3_HalfThickness,  k_Bessel, A_Bessel, C1_Bessel, C2_Bessel, i_Bessel, )

    cm2.subheader(r'$Q = Q_G + \displaystyle \sum _{i=1}^{n} Q_{R_i}$')
    cm3, cm4= cm2.columns([0.5, 0.5])
    cm3.subheader(r'$Q_G = \frac{A^{1/k}k}{\Gamma(1/k)} \frac{a_0P\eta C1}{\pi r_G^2} \exp\left[-C_2\left(\frac{r^2}{r_G^2}\right)^k\right]$')
    cm4.subheader(r'$Q_{R_n} =  \frac{a_nP\eta C_1}{\pi r_{t_n}^2 \text{Y}(r_{r_n},r_{t_n})} \exp\left[-C_2\left(\frac{(r-r_{r_n})^2}{r_{t_n}^2}\right)\right]$')
    # cm4.write(r'$a_0 + a_1 + a_2 + a_3 = 1$')

    
    cm2.plotly_chart(fig_Bessel, use_container_width=True)
    st.divider()

__main__()