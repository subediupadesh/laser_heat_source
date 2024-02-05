import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math


############################################################
## Visualization of Heat Source
############################################################
st.set_page_config(layout="wide")

cm1, cm2 = st.columns([0.2,0.8])

def gaussian_3d(A, C, k, x, y, power, eta, beam_radius):
    r0 = beam_radius*1.0e-6 ## converting to micrometer
    P = power

    r = (x**2 + y**2)**0.5
    F = np.where(beam_radius - r < 0, 0, 1)
    cm2.write(r'''$$\Gamma(1/k) = $$ '''+f'{math.gamma(1/k):.3f}')
    return F*((A**(1/k)*k*P*eta)/(np.pi*r0**2*math.gamma(1/k)))*(np.exp(-C*(r**2/r0**2)**k))
    # return F*((A**(1/k)*k*P*eta)/(np.pi*r0**2))*(np.exp(-C*(r**2/r0**2)**k))

def plot_gaussian_heat_distribution(A, C, k, power, eta, beam_radius, i):
    # Generate data
    x = np.linspace(-200e-6, 200e-6, 100)
    y = np.linspace(-200e-6, 200e-6, 100)
    x, y = np.meshgrid(x, y)
    z = gaussian_3d(A, C, k, x, y, power, eta, beam_radius)
    
    cmaps = ['balance', 'bluered', 'hsv', 'jet', 'picnic', 'portland', 'rainbow', 'rdylbu_r', 'spectral_r', 'turbo']
    # i=6
    cm1.write('Colormap: '+cmaps[i])
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale=cmaps[i])])

    fig.update_layout(scene=dict(xaxis_title='X-axis', yaxis_title='Y-axis', zaxis_title='Intensity'),
                      width=2500,
                      height=2000) 
    fig.update_coloraxes(colorbar=dict(exponentformat='e', thickness=100))
    # fig.update_coloraxes(colorbar_exponentformat='e', colorbar_thickness=100)


    return fig

cm1.title('super-Gaussian Heat Source')


cm2.title(r'''$$Q =  \frac{A^{1/k}Pk\eta}{\pi r_o^2 \Gamma(1/k)} \exp\left[-C(\frac{r^2}{r_0^2})^{k}\right]   $$''')
cm2.title(r'''$$\Gamma(1/k) =  \int_0^\infty t^{\frac{1}{k}-1} e^{-t} dt$$''')

cm1.header('Parameters')
power = cm1.slider(r'''Power $$(P)$$''', min_value=1, max_value=5000, value=1000, step=1)
eta = cm1.slider(r'''Efficiency $$(\eta)$$ ''', min_value=0.0, max_value=1.0, value=0.9, step=0.001)
beam_radius = cm1.slider(r'''Beam Radius $$(r_0$$ $$\mu m)$$''', min_value=100.0, max_value=500.0, value=122.5, step=0.01)
A = cm1.slider(r'''Constant $$(A)$$''', min_value=0.00001, max_value=5.0, value=2.0, step=0.0001)
C = cm1.slider(r'''Constant $$(C)$$''', min_value=0.0000001, max_value=4.0, value=2.0, step=0.0001)
k = cm1.slider(r'''Constant $$(k)$$''', min_value=0.0000001, max_value=10.0, value=3.0, step=0.0001)
i = cm1.slider('colormap', min_value=0, max_value=9, value=6, step=1)

fig = plot_gaussian_heat_distribution(A, C, k, power, eta, beam_radius, i)

cm1.title(r'''$$Q_{peak}$$'''+ f'''= {A**(1/k)*power*k*eta/(math.gamma(1/k)* np.pi*(beam_radius*1e-6)**2):.2e}'''+r'''$$ W/m^2$$''')

# Show the plot
cm2.plotly_chart(fig, use_container_width=True)


## Inverse Gaussian Ring https://link.springer.com/article/10.1007/s11837-023-06363-8