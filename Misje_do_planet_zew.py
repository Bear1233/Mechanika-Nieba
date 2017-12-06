"""
Created on Tue Dec  2 18:57:13 2017

@author: Filip Perczynski
"""

import math
import numpy as np
from astropy import units as u
from astropy import time
from poliastro.bodies import Earth, Sun
from poliastro.twobody import Orbit
from poliastro import ephem
from poliastro.ephem import get_body_ephem
from poliastro import iod

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Wersja wstępna

# Dane wejciowe 
def input():
    # Start z powierzchni Ziemi na rakiecie
    # aktualnie założenie:
    H = 500 * u.km
    # Parametry statku kosmicznego:
    m_veh = 20000 * u.kg
    I_sp = 10 * u.km / u.s
    # Zakładana data startu i dotarcia do celu 
    date_0 = '2018-12-01 12:00'
    date_1 = '2019-01-01 12:00'
    date_0 = time.Time(date_0, format='iso', scale='utc')   # format ISO
    date_1 = time.Time(date_1, format='iso', scale='utc')
    date_0 = time.Time(date_0.jd, format='jd', scale='utc')     # format JD
    date_1 = time.Time(date_1.jd, format='jd', scale='utc')

    return m_veh, I_sp, H, date_0, date_1

# Optymalizacja tranzytu
# Krok iteracji:
step1 = 20 * u.day    
step2 = 10  * u.day 
transit_min= 100 * u.day
transit_max = 400 * u.day

def optimal_transit_mars(date, transit_min, transit_max, vs0, step):

    date_arrival = date + transit_min       
    date_max = date + transit_max           
    date_arrival_final = date_arrival

    vs2_ = 0 * u.km / u.s
    dv_final = 0 * u.km / u.s
    step_one = True

    while date_arrival < date_max:      
        tof = date_arrival - date       
        date_iso = time.Time(str(date.iso), format='iso', scale='utc')      
        date_arrival_iso = time.Time(str(date_arrival.iso), format='iso', scale='utc')      

        r1, vp1 = get_body_ephem("earth", date_iso)     
        r2, vp2 = get_body_ephem("mars", date_arrival_iso) 
        # Rozwiązanie zagadnienia Lamberta
        (vs1, vs2), = iod.lambert(Sun.k, r1, r2, tof, numiter=1000)        

        dv_vector = vs1 - (vs0 + (vp1 / (24 * 3600) * u.day / u.s))     
        dv = np.linalg.norm(dv_vector) * u.km / u.s     

        if step_one:        
            dv_final = dv
            vs2_ = vs2

            step_one = False
        else:
            if dv < dv_final:       
                dv_final = dv
                date_arrival_final = date_arrival
                vs2_ = vs2

        date_arrival += step

    return dv_final, date_arrival_final, vs2_

# Optymalizacja daty startu ze wzgledu na najmniejszy koszt transferu
def start_date_optimal(H, date0, date1, m, Isp, step):
    delta_v = 0 * u.km / u.s
    v_out = 0 * u.km / u.s
    m_prop = 0 * u.kg
    date_in = date0
    date_out = date0

    step_one0 = True      

    while date0 < date1:        
        epoch0 = date0.jyear_str
        ss0 = Orbit.circular(Earth, H, epoch=epoch0)        
        vsE = ss0.rv()[1]       
        # Optymalizacja  manewru
        dv_tot, date_arrivalM, vsM = optimal_transit_mars(date0, transit_min, transit_max, vsE, step)
        # Koszt manewru 
        m_p_tot = m * (math.exp(dv_tot / Isp))

        if step_one0:       
            delta_v = dv_tot
            m_prop = m_p_tot
            v_out = vsM
            date_out = date_arrivalM

            step_one0 = False
        else:
            if dv_tot < delta_v:      
                delta_v = dv_tot
                m_prop = m_p_tot
                v_out = vsM
                date_in = date0
                date_out = date_arrivalM

        date0 += step

    return delta_v, v_out, date_in, date_out, m_prop


m_veh, I_sp, H, date0, date1 = input()

delta_v, v_out, date_in, date_out, m_prop = start_date_optimal(H, date0, date1, m_veh, I_sp, step1)

date0_prec = date_in - 20 * u.day
date1_prec = date_in + 20 * u.day

delta_v, v_out, date_in, date_out, m_prop = start_date_optimal(H, date0_prec, date1_prec, m_veh, I_sp, step2)

# Wyniki
print()
print('Wyniki optymalizacji')
print('Optymalna data startu:', date_in.iso[0:10])
print('Data dolotu do Marsa:', date_out.iso[0:10])
print('Czas lotu: %i dni', int((date_out - date_in).jd))
print('Wymagana zmiana predkosci: %.3f km/s' % float(delta_v / u.km * u.s))
print('Masa zuzytego materialu pednego: %i kg' % int(m_prop / u.kg))
print()

