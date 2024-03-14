import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math

st.set_page_config(layout="wide")

inline_css = """div[data-testid="stExpander"] div[role="button"] p {font-size: 3rem;}"""
st.markdown(f"<style>{inline_css}</style>", unsafe_allow_html=True)

url1 = 'https://link.springer.com/article/10.1007/s11837-023-06363-8'
url2 = 'https://doi.org/10.1080/17445302.2014.937059'
st.write(f'Reference for Equations used in visualization of laser heat soruces: [Super Gaussian]({url1}), [Double Ellipsoide]({url2}), [Ring]({url1}), [Bessel]({url1})')

############################################################
## Super-Gaussian Laser Heat Source with Gamma Function
############################################################

with st.expander('Click for: Super-Gaussian Laser Heat Source with Gamma Function | Flat Top ', expanded=False):
    st.title(f'[Super-Gaussian Heat Source with Gamma Function | Flat Top]({url1})')
    cm1, cm2 = st.columns([0.2,0.8])

    def plot_gaussian_heat_distribution(A, C, k, P, eta, r_0, i):
        r0 = r_0*1.0e-6 ## converting to micrometer
        cmaps = ['balance', 'bluered', 'hsv', 'jet', 'picnic', 'portland', 'rainbow', 'rdylbu_r', 'spectral_r', 'turbo']
        x = np.linspace(-200e-6, 200e-6, 100)
        y = np.linspace(-200e-6, 200e-6, 100)
        x, y = np.meshgrid(x, y)

        r = (x**2 + y**2)**0.5
        F = np.where(r_0 - r < 0, 0, 1)
        cm1.write(r'$\Gamma(1/k) = $ '+f'{math.gamma(1/k):.3f}')
        
        Q = F*((A**(1/k)*k*P*eta)/(np.pi*r0**2*math.gamma(1/k)))*(np.exp(-C*(r**2/r0**2)**k))
        
        cm1.write('Colormap: '+cmaps[i])
        fig = go.Figure(data=[go.Surface(z=Q, x=x, y=y, colorscale=cmaps[i])])
        fig.update_layout(scene=dict(xaxis_title='X-axis', yaxis_title='Y-axis', zaxis_title='Intensity'), width=2500, height=2000) 
        fig.update_coloraxes(colorbar=dict(exponentformat='e', thickness=100))
        return fig, Q

    cm1.header('Parameters')
    power = cm1.slider(r'''Power $$(P)$$''', min_value=1, max_value=5000, value=1000, step=1)
    eta = cm1.slider(r'''Efficiency $$(\eta)$$ ''', min_value=0.0, max_value=1.0, value=0.9, step=0.001)
    beam_radius = cm1.slider(r'''Beam Radius $$(r_0$$ $$\mu m)$$''', min_value=100.0, max_value=500.0, value=122.5, step=0.01)
    A = cm1.slider(r'''Constant $$(A)$$''', min_value=0.00001, max_value=5.0, value=2.0, step=0.0001)
    C = cm1.slider(r'''Constant $$(C)$$''', min_value=0.0000001, max_value=4.0, value=2.0, step=0.0001)
    k = cm1.slider(r'''Constant $$(k)$$''', min_value=0.0000001, max_value=10.0, value=3.0, step=0.0001)
    i = cm1.slider('colormap', min_value=0, max_value=9, value=6, step=1)

    fig, Q = plot_gaussian_heat_distribution(A, C, k, power, eta, beam_radius, i)

    cm2.title(r'$Q =  \frac{A^{1/k}Pk\eta}{\pi r_o^2 \Gamma(1/k)} \exp\left[-C(\frac{r^2}{r_0^2})^{k}\right]$')
    cm2.title(r'$\Gamma(1/k) =  \int_0^\infty t^{\frac{1}{k}-1} e^{-t} dt$')
    cm2.header(r'$Q_{peak} =$  '+f'{np.max(Q):.3e}'+r'  $W/m^2$')

    cm2.plotly_chart(fig, use_container_width=True)
    st.divider()



############################################################################
## Visualization of Double Ellipsoide with Super Gamma Function Heat Source
############################################################################


with st.expander('Click for: Double Ellipsoide Laser Heat Source with Gamma Function'):
    st.title(f'Double Ellipsoide Heat Source with Gamma Function')

    cm3, cm4 = st.columns([0.2,0.8])

    def plot_double_ellipsoide_heat_distribution(P, eta, a_f_DE, a_r_DE, b_DE, c_DE, f_f, f_r, A, C, k, i):
        a_f, a_r, b, c = a_f_DE*1e-6, a_r_DE*1e-6, b_DE*1e-6, c_DE*1e-6  # scaling unit to micro meter

        cmaps = ['balance', 'bluered', 'hsv', 'jet', 'picnic', 'portland', 'rainbow', 'rdylbu_r', 'spectral_r', 'turbo']
        cm3.write('Colormap: '+cmaps[i])

        x = np.linspace(-400e-6, 400e-6, 100)
        y = np.linspace(-200e-6, 200e-6, 100)
        x, y = np.meshgrid(x, y)

        r = (x**2 + y**2)**0.5
        F = np.where((a_f_DE+a_r_DE) - r < 0, 0, 1)
        Qf  =  F*((A**(1/k)*k*f_f*eta*P)/(a_f*b*math.gamma(1/k)*(np.pi)**1.5))*np.exp(-C*(x**2/a_f**2)**k-C*(y**2/b**2)**k)
        Qr  =  F*((A**(1/k)*k*f_r*eta*P)/(a_r*b*math.gamma(1/k)*(np.pi)**1.5))*np.exp(-C*(x**2/a_r**2)**k-C*(y**2/b**2)**k)
        
        Q = Qf+Qr
        fig = go.Figure(data=[go.Surface(z=Q, x=x, y=y, colorscale=cmaps[i])])
        fig.update_layout(scene=dict(xaxis_title='X-axis', yaxis_title='Y-axis', zaxis_title='Intensity'), width=2500, height=2000) 
        fig.update_coloraxes(colorbar=dict(exponentformat='e', thickness=100))
        return Q, fig 


    cm3.header('Parameters')

    P_DE = cm3.slider(r'$Power \, \, (P)$', min_value=1, max_value=5000, value=1000, step=1)
    eta_DE = cm3.slider(r'$Efficiency \, \, (\eta)$', min_value=0.0, max_value=1.0, value=0.9, step=0.001)
    a_f_DE = cm3.slider(r'$a_f \, \, (\mu m)$', min_value=10.0, max_value=500.0, value=250.0, step=0.01)
    a_r_DE = cm3.slider(r'$a_r \, \, (\mu m)$', min_value=10.0, max_value=500.0, value=100.0, step=0.01)
    b_DE = cm3.slider(r'$b \,\, (\mu m)$', min_value=10.0, max_value=500.0, value=150.0, step=0.01)
    # c_DE = cm3.slider(r'''c $$(c$$ $$\mu m)$$''', min_value=10.0, max_value=500.0, value=70.0, step=0.01)
    c_DE = 70.0
    A_DE = cm3.slider(r'$A$', min_value=0.00001, max_value=50.0, value=6*3**0.5, step=0.0001)
    C_DE = cm3.slider(r'$C$', min_value=0.0, max_value=10.0, value=3.0, step=0.0001)
    k_DE = cm3.slider(r'$k$', min_value=0.00001, max_value=10.0, value=1.0, step=0.0001)
    i_DE = cm3.slider('colormap', min_value=0, max_value=9, value=4, step=1)
    f_f_DE = 2*a_f_DE/(a_f_DE+a_r_DE)
    f_r_DE = 2*a_r_DE/(a_f_DE+a_r_DE)

    Q_DE, fig_DE = plot_double_ellipsoide_heat_distribution(P_DE, eta_DE, a_f_DE, a_r_DE, b_DE, c_DE, f_f_DE, f_r_DE, A_DE, C_DE, k_DE, i_DE)

    cm4.title(r'$Q = Q_f + Q_r $')
    cm4.title(r'''$$Q_f =  \frac{A^{1/k}Pk\eta f_f}{a_f b \pi \sqrt{\pi} \Gamma(1/k)} \exp\left[-C\left(\frac{x^2}{a_f^2}\right)^{k}-C\left(\frac{y^2}{b^2}\right)^{k}\right]   $$''')
    cm4.title(r'''$$Q_r =  \frac{A^{1/k}Pk\eta f_r}{a_r b \pi \sqrt{\pi} \Gamma(1/k)} \exp\left[-C\left(\frac{x^2}{a_r^2}\right)^{k}-C\left(\frac{y^2}{b^2}\right)^{k}\right]   $$''')
    cm4.header(r'$f_{f|r} = \frac{2a_{f|r}}{a_f + a_r}$')
    cm4.title(r'$Q_{max} =$ '+f'{np.max(Q_DE):.3e} ' +r'$W/m^2$')

    cm4.plotly_chart(fig_DE, use_container_width=True)
    st.divider()



############################################################################
## Visualization of Double Ellipsoide in 2D
############################################################################

with st.expander('Click for: Pure Double Ellipsoide Laser Heat Source in 2D'):

    st.title(f'[Double Ellipsoide Heat Source in 2D]({url2})')
    cm5, cm6 = st.columns([0.2,0.8])

    def plot_double_ellipsoide_super_gaussian_heat_distribution(P, eta, a_f_DEsG, a_r_DEsG, b_DEsG, f_f, f_r, A, C, i):
        a_f, a_r, b = a_f_DEsG*1e-6, a_r_DEsG*1e-6, b_DEsG*1e-6,  # scaling unit to micro meter

        cmaps = ['balance', 'bluered', 'hsv', 'jet', 'picnic', 'portland', 'rainbow', 'rdylbu_r', 'spectral_r', 'turbo']
        cm5.write('Cmap: '+cmaps[i])

        x = np.linspace(-400e-6, 400e-6, 100)
        y = np.linspace(-200e-6, 200e-6, 100)
        x, y= np.meshgrid(x, y)

        r = (x**2 + y**2)**0.5
        F = np.where((a_f_DEsG+a_r_DEsG) - r < 0, 0, 1)
        Qf  =  F*((A*f_f*eta*P)/(a_f*b*(np.pi)**1.5))*np.exp(-C*(x**2/a_f**2)-C*(y**2/b**2)) # For pure Double Ellipsoide
        Qr  =  F*((A*f_r*eta*P)/(a_r*b*(np.pi)**1.5))*np.exp(-C*(x**2/a_r**2)-C*(y**2/b**2)) # For pure Double Ellipsoide
        
        Q = Qf+Qr
        fig = go.Figure(data=[go.Surface(z=Q, x=x, y=y, colorscale=cmaps[i])])
        fig.update_layout(scene=dict(xaxis_title='X-axis', yaxis_title='Y-axis', zaxis_title='Intensity'), width=2500, height=2000) 
        fig.update_coloraxes(colorbar=dict(exponentformat='e', thickness=100))
        return fig, Q

    cm5.header('Parameters')

    P_DEsG = cm5.slider(r'$Power \, \, [P]$', min_value=1, max_value=5000, value=1000, step=1)
    eta_DEsG = cm5.slider(r'$Efficiency \, \, [\eta]$', min_value=0.0, max_value=1.0, value=0.9, step=0.0001)
    a_f_DEsG = cm5.slider(r'$a_f \, \, [\mu m]$', min_value=10.0, max_value=500.0, value=200.0, step=0.01)
    a_r_DEsG = cm5.slider(r'$a_r \, \, [\mu m]$', min_value=10.0, max_value=500.0, value=75.0, step=0.01)
    b_DEsG = cm5.slider(r'$b \,\, [\mu m]$', min_value=10.0, max_value=500.0, value=125.0, step=0.01)
    # c_DEsG = cm5.slider(r'$c \, \, [\mu m]$', min_value=10.0, max_value=500.0, value=70.0, step=0.01)
    # c_DEsG = 70.0
    A_DEsG = 6.0*np.sqrt(3)
    C_DEsG = 3.0
    i_DEsG = cm5.slider('Cmap', min_value=0, max_value=9, value=2, step=1)
    f_f_DEsG = 2*a_f_DEsG/(a_f_DEsG+a_r_DEsG)
    f_r_DEsG = 2*a_r_DEsG/(a_f_DEsG+a_r_DEsG)

    fig_sG, Q_sG = plot_double_ellipsoide_super_gaussian_heat_distribution(P_DEsG, eta_DEsG, a_f_DEsG, a_r_DEsG, b_DEsG, f_f_DEsG, f_r_DEsG, A_DEsG, C_DEsG, i_DEsG)

    cm6.title(r'$Q = Q_f + Q_r $')
    cm6.header(r'$Q_f =  \frac{6\sqrt{3}P\eta f_f}{a_f b \pi \sqrt{\pi} } \exp\left[-3\left(\frac{x^2}{a_f^2}\right)-3\left(\frac{y^2}{b^2}\right)\right]$')
    cm6.header(r'$Q_r =  \frac{6\sqrt{3}P\eta f_r}{a_r b \pi \sqrt{\pi} } \exp\left[-3\left(\frac{x^2}{a_r^2}\right)-3\left(\frac{y^2}{b^2}\right)\right]$')
    cm6.header(r'$f_{f|r} = \frac{2a_{f|r}}{a_f + a_r}$')

    cm6.title(r'$Q_{max} =$ '+f'{np.max(Q_sG):.3e} ' +r'$W/m^2$')

    cm6.plotly_chart(fig_sG, use_container_width=True)


############################################################################
## Visualization of Ring Laser Heat Source
############################################################################


with st.expander('Click for: Ring Laser Heat Source'):
    st.title(f'[Ring Heat Source]({url1})')
    cm7, cm8 = st.columns([0.2,0.8])

    def plot_ring_heat_distribution(P, eta, r_0, r_t, A, C, i ):
        r0, rt = r_0*1.0e-6, r_t*1.0e-6  # scaling unit to micro meter

        cmaps = ['balance', 'bluered', 'hsv', 'jet', 'picnic', 'portland', 'rainbow', 'rdylbu_r', 'spectral_r', 'turbo']
        cm7.write('Cmap: '+cmaps[i])

        x = np.linspace(-400e-6, 400e-6, 200)
        y = np.linspace(-400e-6, 400e-6, 200)
        x, y= np.meshgrid(x, y)


        r = (x**2 + y**2)**0.5
        F = np.where(r_0 - r < 0, 0, 1)
        Y = np.exp(-r0**2/(2*rt**2)) + (r0/rt)*(np.pi/2)**0.5 * math.erfc(-r0/(rt*2**0.5))

        Q = F*((A*P*eta)/((2*np.pi**3)**0.5*rt**2 * Y)) * (np.exp(-C*((r-r0)**2/(2*rt**2))))

        fig = go.Figure(data=[go.Surface(z=Q, x=x, y=y, colorscale=cmaps[i])])
        fig.update_layout(scene=dict(xaxis_title='X-axis', yaxis_title='Y-axis', zaxis_title='Intensity'), width=2500, height=2000) 
        fig.update_coloraxes(colorbar=dict(exponentformat='e', thickness=100))
        return fig, Q


    cm7.header('Parameters')

    P_Ring = cm7.slider(r'$Power \, \, [P ]$', min_value=1, max_value=5000, value=1000, step=1)
    eta_Ring = cm7.slider(r'$Efficiency \, \, [\eta ]$', min_value=0.0, max_value=1.0, value=0.9, step=0.0001)
    ring_radius = cm7.slider(r'''Beam Radius $$(r_0 $$ $$\mu m)$$''', min_value=100.0, max_value=500.0, value=122.5, step=0.01)
    ring_thickness = cm7.slider(r'''Beam ring thickness $$(r_t$$ $$\mu m)$$''', min_value=1.0, max_value=100.0, value=25.0, step=0.01)

    # c_Ring = cm7.slider(r'$c \, \, [\mu m]$', min_value=10.0, max_value=500.0, value=70.0, step=0.01)
    # c_Ring = 70.0

    A_Ring = cm7.slider(r'''Constant $$(A)$$''', min_value=0.00001, max_value=5.0, value=1.0, step=0.0001)
    C_Ring = cm7.slider(r'''Constant $$(C)$$''', min_value=0.0000001, max_value=4.0, value=1.0, step=0.0001)
    i_Ring = cm7.slider('Cmap ', min_value=0, max_value=9, value=5, step=1)

    fig_Ring, Q_Ring = plot_ring_heat_distribution(P_Ring, eta_Ring, ring_radius, ring_thickness, A_Ring, C_Ring, i_Ring )

    cm8.title(r'$Q =  \frac{AP\eta}{\sqrt{2\pi ^3} r_s^2 \text{Y}(r_0,r_t)} \exp\left[-C\left(\frac{(r-r_0)^2}{2r_t^2}\right)\right]$')
    cm8.title(r'$\text{Y}(r_0,r_t) =  \exp\left(\frac{-r_0^2}{2r_t^2}\right) +\frac{r_0}{r_t}\sqrt{\frac{\pi}{2}}\, \text{erfc}(\frac{-r_0}{\sqrt{2}r_t}) $')
    cm8.header(r'$Q_{peak} =$  '+f'{np.max(Q_Ring):.3e}'+r'  $W/m^2$')
    cm8.plotly_chart(fig_Ring, use_container_width=True)


############################################################################
## Visualization of Bessel Laser Heat Source
############################################################################


with st.expander('Click for: Bessel Laser Heat Source', expanded=True):
    st.title(f'[Bessel Heat Source]({url1})')
    cm9, cm10 = st.columns([0.2,0.8])

    def plot_bessel_heat_distribution(P1, P2, eta, r_0, r_1, r_2, A, C1, C2, i ):
        r0, r1, r2 = r_0*1.0e-6, r_1*1.0e-6, r_2*1.0e-6  # scaling unit to micro meter

        cmaps = ['balance', 'bluered', 'hsv', 'jet', 'picnic', 'portland', 'rainbow', 'rdylbu_r', 'spectral_r', 'turbo']
        cm9.write('Cmap: '+cmaps[i])

        x = np.linspace(-400e-6, 400e-6, 200)
        y = np.linspace(-400e-6, 400e-6, 200)
        x, y= np.meshgrid(x, y)


        r = (x**2 + y**2)**0.5
        F = np.where(r_0 - r < 0, 0, 1)
        Y = np.exp(-r0**2/(2*r2**2)) + (r0/r2)*(np.pi/2)**0.5 * math.erfc(-r0/(r2*2**0.5))

        Q1 = F*((A*P1*eta)/((2*np.pi**3)**0.5*r1**2)) * (np.exp(-C1*((r)**2/(2*r1**2))))
        Q2 = F*((A*P2*eta)/((2*np.pi**3)**0.5*r2**2 * Y)) * (np.exp(-C2*((r-r0)**2/(2*r2**2))))
        Q = Q1 + Q2

        fig = go.Figure(data=[go.Surface(z=Q, x=x, y=y, colorscale=cmaps[i])])
        fig.update_layout(scene=dict(xaxis_title='X-axis', yaxis_title='Y-axis', zaxis_title='Intensity'), width=2500, height=2000) 
        fig.update_coloraxes(colorbar=dict(exponentformat='e', thickness=100))
        return fig, Q


    cm9.header('Parameters')

    P1_Bessel = cm9.slider(r'$Power Inside \, \, [P1]$', min_value=1, max_value=5000, value=1000, step=1)
    P2_Bessel = cm9.slider(r'$Power Ring \, \, [P2]$', min_value=1, max_value=5000, value=1000, step=1)
    eta_Bessel = cm9.slider(r'$Efficiency \, \, [\eta  ]$', min_value=0.0, max_value=1.0, value=0.9, step=0.0001)
    Bessel_radius = cm9.slider(r'''Beam Radius $$(r_0  $$ $$\mu m)$$''', min_value=100.0, max_value=500.0, value=222.5, step=0.01)
    Bessel_in_thickness = cm9.slider(r'''Beam ring thickness $$(r_1$$ $$\mu m)$$''', min_value=1.0, max_value=100.0, value=100.0, step=0.01)
    Bessel_out_thickness = cm9.slider(r'''Beam ring thickness $$(r_2$$ $$\mu m)$$''', min_value=1.0, max_value=100.0, value=20.0, step=0.01)

    # c_Ring = cm9.slider(r'$c \, \, [\mu m]$', min_value=10.0, max_value=500.0, value=70.0, step=0.01)
    # c_Ring = 70.0

    A_Bessel = cm9.slider(r'''Constant $$(A )$$''', min_value=0.00001, max_value=5.0, value=1.0, step=0.0001)
    # B_Bessel = cm9.slider(r'''Constant $$(B)$$''', min_value=0.00001, max_value=5.0, value=1.0, step=0.0001)
    C1_Bessel = cm9.slider(r'''Constant $$(C_1)$$''', min_value=0.0000001, max_value=4.0, value=1.0, step=0.0001)
    C2_Bessel = cm9.slider(r'''Constant $$(C_2)$$''', min_value=0.0000001, max_value=4.0, value=1.0, step=0.0001)
    i_Bessel = cm9.slider('Cmap  ', min_value=0, max_value=9, value=7, step=1)

    fig_Bessel, Q_Bessel = plot_bessel_heat_distribution(P1_Bessel, P2_Bessel, eta_Bessel, Bessel_radius, Bessel_in_thickness, Bessel_out_thickness, A_Bessel, C1_Bessel, C2_Bessel, i_Bessel )

    cm10.title(r'$Q_1 =  \frac{AP_1\eta}{\sqrt{2\pi ^3} r_1^2} \exp\left[-C_1\left(\frac{r^2}{2r_1^2}\right)\right]$')
    cm10.title(r'$Q_2 =  \frac{AP_2\eta}{\sqrt{2\pi ^3} r_2^2 \text{Y}(r_0,r_2)} \exp\left[-C_2\left(\frac{(r-r_0)^2}{2r_2^2}\right)\right]$')
    cm10.title(r'$\text{Y}(r_0,r_2) =  \exp\left(\frac{-r_0^2}{2r_2^2}\right) +\frac{r_0}{r_2}\sqrt{\frac{\pi}{2}}\, \text{erfc}(\frac{-r_0}{\sqrt{2}r_2}) $')
    cm10.header(r'$Q_{peak} =  Q_1 + Q_2  =$  '+f'{np.max(Q_Bessel):.3e}'+r'  $W/m^2$')
    cm10.plotly_chart(fig_Bessel, use_container_width=True)
    st.divider()

## Double Ellipsoide https://www.tandfonline.com/doi/epdf/10.1080/17445302.2014.937059?needAccess=true

## Inverse Gaussian Ring, Bessel https://link.springer.com/article/10.1007/s11837-023-06363-8
    
