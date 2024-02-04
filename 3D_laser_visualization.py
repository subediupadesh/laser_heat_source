import streamlit as st
import numpy as np
import plotly.graph_objects as go

############################################################
## Visualization of Heat Source
############################################################
st.set_page_config(layout="wide")

cm1, cm2 = st.columns([0.2,0.8])

def gaussian_3d(A, C, x, y, power, absorptance, beam_radius):
    beam_radius = beam_radius*1.0e-6 ## converting to micrometer
    absorptance = absorptance*1.0e7  # converting to 1.0e7 /m

    r = (x**2 + y**2)**0.5
    F = np.where(beam_radius - r < 0, 0, 1)

    # return power * absorptance * np.exp(-(x**2 + y**2) / (2 * beam_radius**2))
    return F*((A*power*absorptance)/(np.pi*beam_radius**2))*np.exp(-C*(x**2 + y**2)/beam_radius**2)

def plot_gaussian_heat_distribution(A, C, power, absorptance, beam_radius, i):
    # Generate data
    x = np.linspace(-400e-6, 400e-6, 100)
    y = np.linspace(-400e-6, 400e-6, 100)
    x, y = np.meshgrid(x, y)
    z = gaussian_3d(A, C, x, y, power, absorptance, beam_radius)
    
    cmaps = ['balance', 'bluered', 'hsv', 'jet', 'picnic', 'portland', 'rainbow', 'rdylbu_r', 'spectral_r', 'turbo']
    # i=6
    cm1.write('Colormap: '+cmaps[i])
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale=cmaps[i])])

    fig.update_layout(scene=dict(xaxis_title='X-axis', yaxis_title='Y-axis', zaxis_title='Intensity'),
                      width=2500,
                      height=2000) 
    return fig

cm1.title('Gaussian and Flat-top Heat Source')

cm2.title(r'''$$Q =  \frac{AP\alpha}{\pi r_o^2} \exp(\frac{-Cr^2}{r_0^2})   $$''')

cm1.header('Parameters')
power = cm1.slider(r'''Power $$(P)$$''', min_value=1, max_value=5000, value=1000, step=1)
absorptance = cm1.slider(r'''Absorptance $$(\alpha)$$ $$[\times 10^7 /m]$$''', min_value=1.0, max_value=1.0e2, value=8.5, step=1.0)
beam_radius = cm1.slider(r'''Beam Radius $$(r_0$$ $$\mu m)$$''', min_value=100.0, max_value=500.0, value=222.5, step=0.01)
A = cm1.slider(r'''Constant $$(A)$$''', min_value=0.00001, max_value=5.0, value=2.0, step=0.0001)
C = cm1.slider(r'''Constant $$(C)$$''', min_value=0.0000001, max_value=4.0, value=2.0, step=0.0001)
i = cm1.slider('colormap', min_value=0, max_value=9, value=6, step=1)

fig = plot_gaussian_heat_distribution(A, C, power, absorptance, beam_radius, i)



# Show the plot
cm2.plotly_chart(fig, use_container_width=True)

