"""
Created: 2017-12-10
Author: Filip Perczyński
------------------------------------------------------------------------------
Program główny, w którym zawierają sie:
- funkcja transit() zwraca: - wektory pozycji i predkosci planet, rozwiazanie 
    problemu Lamberta
- funkcja transit_optimal() - zwraca zoptymalizowana date konca misji i 
    zmiane predkosci
- funkcja start_date_optimal() - zwraca zoptymalizowana date startu, konca misji, 
    najmniejsza zmiane predkosci, zuzycie materialu pednego i dane do wykresu 
- funkcja planets2plot() - zwraca etykiete planety i kolor do narysowania wykresu
- funkcja mission_plot() - zwraca wykresy 2D i 3D trajektorii lotu
- funkcja opt_plot() - zwraca wykresy zmiany predkosci i masy materialu pednego 
    w funkcji czasu dla analizy 2
------------------------------------------------------------------------------
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel

from poliastro import iod
from poliastro.bodies import *
from poliastro.twobody import Orbit
from poliastro.plotting import OrbitPlotter
from poliastro.util import time_range

import plotly.graph_objs as go
from plotly.offline import plot 

import input_data
import myplot

def transit(date, date_arrival, planet1, planet2):
    #czas trwania misji
    tof = date_arrival - date       
    N = 50 
    tof.to(u.h)
    times_vector = time_range(date, end=date_arrival, periods=N)      

# okreslenie pozycji planet w przedziale czasu: od startu do końca misji
    rr_planet1, vv_planet1 = get_body_barycentric_posvel(planet1, times_vector)
    rr_planet2, vv_planet2 = get_body_barycentric_posvel(planet2, times_vector)
        
    r0 = rr_planet1[0].xyz #wektor pozycji Ziemi w momencie startu
    v0 = vv_planet1[0].xyz #wektor predkosci Ziemi w momencie startu
    rf = rr_planet2[-1].xyz #wektor pozycji planety docelowej w momencie końca misji
    vf = vv_planet2[-1].xyz #wektor predkosci planety docelowej w momencie końca misji  

# rozwiazanie problemu Lamberta    
    (va, vb), = iod.lambert(Sun.k, r0, rf, tof, numiter=1000)

    return(r0, v0, rf, vf, va, vb, rr_planet1, rr_planet2, times_vector)    


def transit_optimal(date, transit_min, transit_max, planet1, planet2, vs0, step):
    #wartosci poczatkowe
    date_out = date + transit_min 
    date_max = date + transit_max            
    date_out_final = date_out
    dv_final = 0 * u.km / u.s
    
    step_one = True  #sprawdzenie pierwszego kroku    

    while date_out < date_max:  #poszukiwanie optymalnej daty konca misji   
             
        r0, v0, rf, vf, va, vb, rr_planet1, rr_planet2, times_vector = transit(date, date_out, planet1, planet2)
        
        dv_vector = va - (vs0 + (v0 / (24*3600) * u.day / u.s)) #wektor wymaganej zmiany predkosci    
        dv = np.linalg.norm(dv_vector) * u.km / u.s  #zmiana predkosci   
        
        if step_one:        
            dv_final = dv
            step_one = False
        else:
            if dv < dv_final: #wybór koniguracji z najmniejsza zmniana predkosci     
                dv_final = dv
                date_out_final = date_out

        date_out += step
    
    return dv_final, date_out_final


def start_date_optimal(H, date0, date1, m, Isp, step):
    #wartosci poczatkowe   
    delta_v = 0 * u.km / u.s
    dv_final = 0 * u.km / u.s
    m_prop = 0 * u.kg
    date_launch_final = date0
    date_arrival_final = date0
    
    step_one0 = True  #sprawdzenie pierwszego kroku         

    print('Data startu, Czas lotu [dni], Zmiana predkosci dV [km/s], Material pedny [kg]')
    
    # inicjalizacja zmiennych do tworzenia wykresu
    plot_date = np.array([])
    plot_mprop = np.array([])
    plot_dv = np.array([])

    while date0 < date1: #poszukiwanie optymalnej daty poczatku misji
        epoch0 = date0.jyear_str
        ss0 = Orbit.circular(Earth, H, epoch=epoch0) #poczatkowa orbita kolowa       
        vsE = ss0.rv()[1] #predkosc na orbicie kolowej                     
        
        dv_final, date_out_final = transit_optimal(date0, transit_min, transit_max, 'earth', planet, vsE, step)
    
        m_p_tot = m * (np.exp(dv_final/ Isp)-1) # wymagana masa materialu pednego
        
        # zmiana formatu danych do przedstawienia na wykresie
        x = str(date0.iso[0:10])
        y = m_p_tot.value
        y2 = dv_final.value
        # macierze zmiennych do stworzenia wykresu
        plot_date = np.append(plot_date,x)
        plot_mprop = np.append(plot_mprop,y)
        plot_dv = np.append(plot_dv,y2)
        
        print(date0.iso[0:10], ', %i dni, %.3f km/s, %i kg' % (int((date_out_final - date0).jd),
                                                               float(dv_final / u.km * u.s),
                                                               int(m_p_tot / u.kg)))
        if step_one0:      
            delta_v = dv_final
            m_prop = m_p_tot
            date_arrival_final = date_out_final
            step_one0 = False
        else:
            if dv_final < delta_v: #wybór koniguracji z najmniejsza zmniana predkosci       
                delta_v = dv_final
                m_prop = m_p_tot
                date_launch_final = date0 #optymalna data startu
                date_arrival_final = date_out_final #optymalana data konca
                
        date0 += step

    return delta_v, date_launch_final, date_arrival_final, m_prop, plot_date, plot_mprop, plot_dv 


def planets2plot(planet):
    if planet == 'mercury':
        pl_planet = Mercury
        color_planet = 'sandybrown'
    if planet == 'venus':
        pl_planet = Venus
        color_planet = 'blueviolet'
    if planet == 'earth':
        pl_planet = Earth
        color_planet = 'royalblue'
    if planet == 'mars':
        pl_planet = Mars
        color_planet = 'indianred'
    if planet == 'jupiter':
        pl_planet = Jupiter
        color_planet = 'coral'
    if planet == 'saturn':
        pl_planet = Saturn
        color_planet = 'springgreen'
    if planet == 'uranus':
        pl_planet = Uranus
        color_planet = 'skyblue'
    if planet == 'neptune':
        pl_planet = Neptune
        color_planet = 'navy'

    return (pl_planet, color_planet)


def mission_plot(date_launch, date_arrival, planet):

    color_planet_e = 'royalblue'
    color_trans = 'dimgrey'
    color_orbit_trans = 'orchid'
    
    r0, v0, rf, vf, va, vb, rr_planet1, rr_planet2, times_vector = transit(date_launch, date_arrival, 'earth', planet)
    
    # Obliczenie orbit transferowych oraz orbit planet
    ss0_trans = Orbit.from_vectors(Sun, r0, va, date_launch)
    ssf_trans = Orbit.from_vectors(Sun, rf, vb, date_arrival)
    ss_e= Orbit.from_vectors(Sun, r0, v0, date_launch)
    ss_p = Orbit.from_vectors(Sun, rf, vf, date_arrival)

    pl_planet, color_planet = planets2plot(planet)
    
    #wykres orbit 2D
    orb = OrbitPlotter()
    orb.plot(ss_p, label= pl_planet, color=color_planet)
    orb.plot(ss_e, label= Earth, color=color_planet_e)
    orb.plot(ssf_trans, label='Orbita transferowa', color=color_orbit_trans)
    
    #wykres orbit i trajektorii lotu 3D
    frame = myplot.OrbitPlotter3D()
    frame.set_attractor(Sun)
    frame.plot_trajectory(rr_planet1, label=Earth, color=color_planet_e)
    frame.plot(ss_e, label= Earth, color=color_planet_e)
    frame.plot(ss_p, label= pl_planet, color=color_planet)
    frame.plot_trajectory(rr_planet2, label=pl_planet, color=color_planet)
    frame.plot_trajectory(ss0_trans.sample(times_vector), label="Mission trajectory", color=color_trans)
    frame.set_view(30 * u.deg, 260 * u.deg, distance=3* u.km)
    frame.show(title="Mission to Solar System Planet")
    
   
def opt_plot(x,y1,y2): 
    trace1 = go.Scatter(
        x = plot_date,
        y = plot_mprop,
        name='Masa materiału pędnego',
        line = dict(
            color = ('rgb(93,75,144)')
        )
    )
    trace2 = go.Scatter(
        x = plot_date,
        y = plot_dv,
        name='Zmiana prędkosci',
        yaxis='y2',
        line = dict(
            color = ('rgb(110,164,193)')
        )
    )
    data =[trace1, trace2]
    
    layout = go.Layout(

        title='Wykres zmiany prędkosci oraz masy materiału pędnego',
        yaxis=dict(
            title='M materiału pędnego [kg]'
        ),
        xaxis=dict(
            title='Data startu misji'
        ),
        yaxis2=dict(
            title='Delta V [km/s]',
            overlaying='y',
            side='right'
            )  
    )
    fig = go.Figure(data=data, layout=layout)
    plot_url = plot(fig, filename='parametry.html')
            
    return()
"""
------------------------------------------------------------------------------
"""

print ('='*80)
print('Analiza misji do planet Układu Słonecznego')
print ('='*80)
print()    
 
nr, m_spacecraft, I_sp, H, planet, date_launch, date_arrival, transit_min, transit_max = input_data.inputf()

solar_system_ephemeris.set("jpl")

if nr in [2]:
    step1 = 20 * u.day #krok analizy optymalizacyjnej  
    step2 = 1 * u.day   
    
    print()
    print('='*80)
    print('Wyniki analizy wstepnej')
    print('='*80)
    # analiza wstępna dla kroku 1
    delta_v, date_launch_final, date_arrival_final, m_prop, plot_date, plot_mprop, plot_dv = start_date_optimal(H, date_launch, date_arrival, m_spacecraft, I_sp, step1)
    
    print('-'*80)
    print('Koniec analizy wstepnej')
    print('Optymalna data startu:', date_launch_final.iso[0:10])
    print('-'*80)
    
    print('='*80)
    print('Czy przeprowadzić analizę szczegółową?')
    an = input('Wpisz "tak" lub "nie": ')    
    an = str(an)
    check = True
    while check:
        if an in ['tak','nie']:
            check = False
            check2 = True
            while check2:
                if an == 'tak':
                    print('='*80)
                    print('Wyniki analizy szczegolowej')
                    print('='*80)
                    date0_prec = date_launch_final - 20 * u.day #zawezenie czasu analizy
                    date1_prec = date_arrival_final + 20 * u.day
                    # analiza szczegolowa
                    delta_v, date_launch_final, date_arrival_final, m_prop, plot_date, plot_mprop, plot_dv = start_date_optimal(H, date0_prec, date1_prec, m_spacecraft, I_sp, step2)
                    
                    print()
                    print('Koniec analizy szczegolowej')
                    print('-'*80)
                    check2 = False
                else:
                    check2 = False
        else:
            print('Wprowadzone słowo jest niepoprawne')
    #prezentacja wynikow        
    print()
    print('='*80)
    print('Wyniki analizy optymalizacyjnej')
    print('='*80)
    print()
    print('Optymalna data startu:', date_launch_final.iso[0:10])
    print('Data dolotu do planety:', date_arrival_final.iso[0:10])
    print('Czas lotu:', int((date_arrival_final - date_launch_final).jd),'dni')
    print('Wymagana zmiana predkosci: %.3f km/s' % float(delta_v / u.km * u.s))
    print('Masa zuzytego materialu pednego: %i kg' % int(m_prop / u.kg))
    print('-'*80)
    
    opt_plot( plot_date, plot_mprop, plot_dv)
    
mission_plot(date_launch_final, date_arrival_final, planet)
