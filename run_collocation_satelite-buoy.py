'''
Description
----------- 
    Extract satellite along track wind  and wave dataset 
    and make comparison plots with buoys data
    
Usage
-----
    python run_collocation_satellite-buoy.py

Input
-----
    satellite data and buoy data or model data 

Output
------
    figures of time series, scatter plots and maps of the study area
    
 By B.Sc., M.Sc. Rodney Eduardo Mora-Escalante
    Creation day August 27, 2022

'''

##
##---I-M-P-O-R-T---M-O-D-U-L-E-S---
##

import numpy as np
import netCDF4 as nc
from netCDF4 import date2num, num2date
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, LogLocator
import glob
import os
import datetime as dt
from math import radians, degrees, sin, cos, asin, acos, sqrt
from mpl_toolkits.axes_grid.inset_locator import inset_axes, InsetPosition, mark_inset
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from scipy import stats as st
from scipy.optimize import curve_fit
#from scipy import stats
from scipy.stats.stats import pearsonr

##
##---G-L-O-B-A-L---P-A-R-A-M-E-T-E-R-S---
##

OS = os.popen('uname -s').read()[:-1]
PATH = os.getcwd()

BLOCK = False
PAUSE = 1

## Root netCDF file
if OS == 'Darwin':
    ROOTNCFILE = '/Users/rodneymoes/Desktop/ProjectOnPython/Ojive-Turbulence/dataset'
else:
    ROOTNCFILE = '/media/student/BackupPlus/BOMM1-PER1/level1'

## name of file
dirfile = []
for name in sorted(glob.glob(ROOTNCFILE+'/*.nc')):
    dirfile.append(name)

## Quality figure
if OS == 'Darwin':
    plt.style.use(os.path.join('/Users/rodneymoes', 'my_plot_style.txt'))
else:
    plt.style.use(os.path.join('/home/student', 'my_plot_style.txt'))

# Width as measured in inkscape
# One column.
##width = 3.487
##height = width / 1.618
# Two columns. Multiply by 2
#width = 2*3.487

##
##---F-U-N-C-T-I-O-N-S---B-E-L-O-W---
##
def linear_trend(xi, yi, alpha = 0.05):
    '''
    Description
    -----------
        Do a linear regression with confidents intervals
        an correlation coefficient
    Arguments
    ---------
    Return
    ------
    Reference:
    https://rpubs.com/hllinas/R_Regresion_Lineal_toc
    '''
    
    from scipy.stats import t

    xm = np.nanmean(xi)
    ym = np.nanmean(yi)
    xdev = np.nanstd(xi)
    ydev = np.nanstd(yi)

    N = len(xi)
    Sxx = sum(xi**2) - sum(xi)**2 / N # Variance
    Syy = sum(yi**2) - sum(yi)**2 / N # Variance
    Sxy = sum(xi * yi) - (sum(xi) * sum(yi)) / N #Covariance

    Beta_hat = Sxy / Sxx # Compute Slope
    delta_hat =  ym - Beta_hat * xm # Intercept

    ## Best Estimation, also know as expected values
    E_x_y = Beta_hat * xi + delta_hat

    ## Pierson correlation coefficient
    cov_xy = sum((xi - xm) * (yi - ym)) * (1 / (N - 1))
    rho_xy = cov_xy / (xdev * ydev)

    ## Estimation of variance of error, slope and intercept
    SSR = Beta_hat * Sxy
    SSE = Syy - SSR
    S_eps2 = SSE / (N - 2)
    S_eps = np.sqrt(S_eps2)
    # 
    Beta_eps2 = S_eps2 / Sxx
    Beta_eps = np.sqrt(Beta_eps2)
    # 
    delta_eps2 = (S_eps2 * sum(xi**2)) / (N * Sxx)
    delta_eps = np.sqrt(delta_eps2)

    ## Confidence intervals (95%) with N -2 degrees of freedom
    df = N - 2
    t_alpha_half = t.ppf(1 - alpha / 2, df) # t-Student
    delta_ci_low = delta_hat - t_alpha_half * delta_eps
    delta_ci_upp = delta_hat + t_alpha_half * delta_eps
    delta_ci = [delta_ci_low, delta_ci_upp]
    Beta_ci_low = Beta_hat - t_alpha_half * Beta_eps
    Beta_ci_upp = Beta_hat + t_alpha_half * Beta_eps
    Beta_ci = [Beta_ci_low, Beta_ci_upp]

    ## Hypothesis null and alternative
    Beta_zero = 0
    t_value = Beta_hat - Beta_zero / Beta_eps
    if (t_value > -t_alpha_half) and (t_value < t_alpha_half):
        Beta = 'Accept Ho: m = 0'
    else:
        Beta = 'Accept Ha: m is different to zero'

    return Beta_hat, delta_hat, E_x_y, rho_xy, delta_ci, Beta_ci, Beta


def get_prediction_interval(prediction, y_test, test_predictions, pi = .95):
    '''
    Get a prediction interval for a linear regression.
    INPUTS:
    - Single prediction,
    - y_test
    - All test set predictions,
    - Prediction interval threshold (default = .95)
    OUTPUT:
    - Prediction interval for single prediction
    Reference:
    https://medium.com/swlh/ds001-linear-regression-and-confidence-interval-a-hands-on-tutorial-760658632d99
    '''
    
    ##get standard deviation of y_test
    sum_errs = np.sum((y_test - test_predictions)**2)
    stdev = np.sqrt(1 / (len(y_test) - 2) * sum_errs)
    
    ##get interval from standard deviation
    one_minus_pi = 1 - pi
    ppf_lookup = 1 - (one_minus_pi / 2)
    z_score = stats.norm.ppf(ppf_lookup)
    interval = z_score * stdev
    
    ##generate prediction interval lower and upper bound cs_24
    lower, upper = prediction - interval, prediction + interval
    
    return lower, prediction, upper


def near(x, x0 , n = 1):
    '''
    NEAR  finds the indices of x that are closest to the point x0.

    x is an array, 
    x0 is a point, 
    n is the number of closest points to get
    (in order of increasing distance).  Distance is the abs(x-x0)
    '''


    mindist = np.abs(x - x0)
    distance = np.sort(mindist)
    index = np.argsort(mindist)

    distance = distance[0:n]
    index = index[0:n]

    return index, distance



def great_circle(lon1, lat1, lon2, lat2):
    '''
    The great-circle distance or orthodromic distance is the shortest
    distance between two points on the surface of a sphere,
    measured along the surface of the sphere.
    Reference:
    Calculate distance of two locations on Earth using Python
    Author Pete Houston
    https://medium.com/@petehouston/calculate-distance-of-two-locations-on-earth-using-python-1501b1944d97
    See also https://www.movable-type.co.uk/scripts/latlong.html
    Chris Veness (C) 2002-2022
    '''
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    R_Earth = 6371 # Km
    
    dist = R_Earth * (np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)))
    
    return dist


def haversine(lon1, lat1, lon2, lat2):
    '''
    Reference:
    Calculate distance of two locations on Earth using Python
    Author Pete Houston
    https://medium.com/@petehouston/calculate-distance-of-two-locations-on-earth-using-python-1501b1944d97
    '''

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    R_Earth = 6371 # Km
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = R_Earth * c#2 * np.arcsin(np.sqrt(a))
    
    return dist

def linear(X, m, b):
    '''
    Define a straight line
    '''
    Y = m * X + b

    return Y


## Main program
if __name__ == '__main__':

    ## Which plot to make
    TIMSER1 = True
    TIMSER2 = True
    MAP = True

    units1 = 'days since 1985-01-01 00:00:00 UTC'
    units2 = 'seconds since 1970-01-01 00:00:00.000 UTC'

    ## Read satellite dataset
    filecsv = '/Users/rodneymoes/PROJECTS-WORKING/IMOS-RADWAVES/altimeterData-GoM.csv'
    lat, lon, hs, sat_ser, ws = np.loadtxt(filecsv, skiprows = 1, unpack = True)
    lon = lon - 360
    ## Note that time convention: days since 1985
    sat_for = num2date(sat_ser, units = units1,
        only_use_cftime_datetimes = False, only_use_python_datetimes = True)

    ## Read Buoy dataset
    filenc = '/Users/rodneymoes/DOCTORADO/BOMM-DATABASE/level2/bomm1_per1_level2_10min.nc'
    nc_bomm = nc.Dataset(filenc, 'r')
    ## bomm_for[25056] --> Out[0]: real_datetime(2019, 1, 1, 0, 0)
    tE = 25056
    Hm0 = np.array(nc_bomm.variables['Hm0'][:tE])
    Wspd = np.array(nc_bomm.variables['Wspd'][:])
    Ua = np.array(nc_bomm.variables['Ua'][:tE])
    Va = np.array(nc_bomm.variables['Va'][:tE])
    U10N = np.array(nc_bomm.variables['U10N'][:tE])
    U = np.sqrt(Ua**2 + Va**2)
    ## Note that time convention: seconds since 1970
    bomm_ser = np.array(nc_bomm.variables['time'][:tE])
    bomm_for = num2date(bomm_ser, units = units2,
        only_use_cftime_datetimes = False, only_use_python_datetimes = True)

    BOMM1 = [-96.62464, 24.59861]

    ## Read coastline data
    file_coast ='/Users/rodneymoes/DOCTORADO/BOMM-DATABASE/analysis/Coastline_WholeWorld/Coastline_GoM/gom.coast.intermediate.dat'
    coastline = np.loadtxt(file_coast, delimiter = ',')

    lon_min = np.nanmin(-coastline[:,0])
    lon_max = np.nanmax(-coastline[:,0])
    lat_min = np.nanmin(coastline[:,1])
    lat_max = np.nanmax(coastline[:,1])


    ##
    ##---S-T-A-R-T---C-O-M-P-U-T-A-T-I-O-N-S---
    ##

    ## Inline function
    min_time = lambda T, T0: np.abs(T - T0)

    ## Compute distance
    dist = haversine(BOMM1[0], BOMM1[1], lon, lat)
    ## Criterion to match 50 km (radius of 25 km)
    indx_50km = np.where(dist <= 25)[0]

    ## Extract just 50 km criterion
    ws_50km = ws[indx_50km]
    hs_50km = hs[indx_50km]
    sat_time_50km = sat_for[indx_50km]
    indx_sort_time = np.argsort(sat_time_50km)

    ## Select just the time from July to December 2018
    sat_for_Y18 = np.sort(sat_time_50km[indx_sort_time])[560:595]
    ws_Y18 = ws_50km[indx_sort_time][560:595]
    hs_Y18 = hs_50km[indx_sort_time][560:595]

    ## Transform sat time from convention of days since 1985 to seconds since 1970
    sat_ser_Y18 = date2num(sat_for_Y18, units = units2)
    sat_for_Y18 = num2date(sat_ser_Y18, units = units2,
        only_use_cftime_datetimes = False, only_use_python_datetimes = True)

    ## Match the closer time between buoy and satellite
    ## For this dataset, I choose 10 minutes or 600 seconds
    dt_crit = 600
    indx_match = []
    hs_sat_buoy = []
    ws_sat_buoy = []
    for ti in range(0, tE):
        dt = min_time(sat_ser_Y18, bomm_ser[ti])
        if sum(dt < dt_crit) != 0:
            indx_match.append(ti)
            if sum(dt < dt_crit) > 1:
                hs_sat_buoy.append(np.mean(hs_Y18[dt < dt_crit]))
                ws_sat_buoy.append(np.mean(ws_Y18[dt < dt_crit]))
            else:
                hs_sat_buoy.append(hs_Y18[dt < dt_crit])
                ws_sat_buoy.append(ws_Y18[dt < dt_crit])

    ## Convert list to numpy array
    hs_sat_buoy = np.array(hs_sat_buoy)
    ws_sat_buoy = np.array(ws_sat_buoy)
    ## Resolve issue with nans, remove them
    nonans = np.where(~np.isnan(Hm0[indx_match]))[0]

    ## Compute linear fit with curve_fit function
    hs_pred, hs_cov = curve_fit(linear, hs_sat_buoy[nonans], Hm0[indx_match][nonans])
    U10N_pred, U10N_cov = curve_fit(linear, ws_sat_buoy[nonans], U10N[indx_match][nonans])
    ## To compute one standard deviation errors on the parameters: m and b
    st_hs = np.sqrt(np.diag(hs_cov)) 
    st_U10N = np.sqrt(np.diag(U10N_cov))

    print('m =', hs_pred[0], '+/-', hs_cov[0, 0]**0.5)
    print('b =', hs_pred[1], '+/-', hs_cov[1, 1]**0.5)

    ## Second option to compute linear fit with linear_trend function
    m1, B1, y_pred1, R1, B_ci1, m_ci1, _ = linear_trend(hs_sat_buoy[nonans], Hm0[indx_match][nonans], alpha = 0.05)
    m2, B2, y_pred2, R2, B_ci2, m_ci2, _ = linear_trend(ws_sat_buoy[nonans], U10N[indx_match][nonans], alpha = 0.05)

    ## Evaluation metrics
    MAE1 = metrics.mean_absolute_error(hs_sat_buoy[nonans], Hm0[indx_match][nonans])
    MSE1 = metrics.mean_squared_error(hs_sat_buoy[nonans], Hm0[indx_match][nonans])
    RMSE1 = np.sqrt(metrics.mean_squared_error(hs_sat_buoy[nonans], Hm0[indx_match][nonans]))
    MAE2 = metrics.mean_absolute_error(ws_sat_buoy[nonans], U10N[indx_match][nonans])
    MSE2 = metrics.mean_squared_error(ws_sat_buoy[nonans], U10N[indx_match][nonans])
    RMSE2 = np.sqrt(metrics.mean_squared_error(ws_sat_buoy[nonans], U10N[indx_match][nonans]))

    ###
    ###---M-A-K-E---P-L-O-T-S---
    ###

    plt.close('all')

    ## Time series figure
    if TIMSER1 == True:

        fig, ax = plt.subplots(figsize = (10, 6), facecolor = 'w',
            edgecolor = 'k', constrained_layout = True)

        ax.plot(sat_time_50km[indx_sort_time], hs_50km[indx_sort_time],
            linestyle = '-', alpha = 1., marker = '', markersize = 6,\
            markerfacecolor = '#14213D', markeredgecolor = 'black',\
            linewidth = 2, color = '#14213D', zorder = 20, label = r'Hs')

        ax.set_xlabel('Time [days]', fontweight = 'bold')
        ax.set_ylabel('Wave height [m]', fontweight = 'bold')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax2 = ax.twinx()
        ax2.plot(sat_time_50km[indx_sort_time], ws_50km[indx_sort_time],
            linestyle = '-', alpha = 0.8, marker = '', markersize = 6,\
            markerfacecolor = '#14213D', markeredgecolor = 'black',\
            linewidth = 2, color = '#FCA311', zorder = 10, label = r'U10')
        ax2.set_ylabel('Wind speed [m/s]', fontweight = 'bold')
        ax2.yaxis.set_minor_locator(AutoMinorLocator())

        ## Label for legend
        ax.plot([],[], linestyle = '-', alpha = 1., marker = '', markersize = 6,\
            markerfacecolor = '#14213D', markeredgecolor = 'black',\
            linewidth = 2, color = '#FCA311', label = r'U10')

        #ax.axis([0, 16, 0, 16])
        ## Legend and grids
        ax.tick_params(which = 'both', direction = 'in', bottom = True, top = True, left = True, right = True)
        ax.legend(loc = 'upper right' , ncol = 1, prop = dict(weight = 'bold'))
        ax.grid(color = 'black', alpha = .5, axis = 'both', which = 'both', linestyle = '--', linewidth = .5)
        plt.savefig('Hs-Wspd_Buoy-Satellite_timeseries.png', dpi = 300)
        plt.show()


    if TIMSER2 == True:

        fig, ax = plt.subplots(nrows = 1, ncols = 2,figsize = (10, 6), facecolor = 'w',
            edgecolor = 'k', constrained_layout = True)

        ax[0].plot(hs_sat_buoy, Hm0[indx_match], 
            linestyle = '', alpha = 1., marker = 'o', markersize = 6,\
            markerfacecolor = 'green', markeredgecolor = 'black',\
            linewidth = 2, color = '#FCA311')

        ax[0].plot(hs_sat_buoy[nonans], linear(hs_sat_buoy[nonans], *hs_pred),
            linestyle = '--', alpha = 1., marker = '', markersize = 6,\
            markerfacecolor = '#14213D', markeredgecolor = 'black',\
            linewidth = 2, color = 'black', 
            label = 'y = %5.3f x + %5.3f, R = %4.2f\nMAE = %5.3f m\nMSE = %5.3f m$^2$\nRMSE = %5.3f m' % (hs_pred[0], hs_pred[1], R1, MAE1, MSE1, RMSE1))

        ax[0].set_xlabel('Wave height - Satellite [m]', fontweight = 'bold')
        ax[0].set_ylabel('Wave height -Buoy [m]', fontweight = 'bold')

        ax[1].plot(ws_sat_buoy, U10N[indx_match], 
            linestyle = '', alpha = 1., marker = 'o', markersize = 6,\
            markerfacecolor = 'red', markeredgecolor = 'black',\
            linewidth = 2, color = '#FCA311',label = r'U$_{10N}$')

        ax[1].plot(ws_sat_buoy, U[indx_match], 
            linestyle = '', alpha = 1., marker = 's', markersize = 6,\
            markerfacecolor = 'b', markeredgecolor = 'black',\
            linewidth = 2, color = '#FCA311', label = r'U$_{6.5m}$')

        ax[1].plot(ws_sat_buoy, Wspd[indx_match],
            linestyle = '', alpha = 1., marker = '^', markersize = 6,\
            markerfacecolor = 'm', markeredgecolor = 'black',\
            linewidth = 2, color = '#FCA311',label = r'U$_{5.5m}$')

        ax[1].plot(ws_sat_buoy[nonans], linear(ws_sat_buoy[nonans], *U10N_pred),
            linestyle = '--', alpha = 1., marker = '', markersize = 6,\
            markerfacecolor = '#14213D', markeredgecolor = 'black',\
            linewidth = 2, color = 'black', 
            label = 'y = %5.3f x + %5.3f, R = %4.2f' % (U10N_pred[0], U10N_pred[1], R2))

        ax[1].set_xlabel('Wind speed - Satellite [m/s]', fontweight = 'bold')
        ax[1].set_ylabel('Wind speed - Buoy [m/s]', fontweight = 'bold')
        
        ## Legend and grids
        for i in range(2):
            ax[i].xaxis.set_minor_locator(AutoMinorLocator())
            ax[i].yaxis.set_minor_locator(AutoMinorLocator())
            ax[i].tick_params(which = 'both', direction = 'in', bottom = True, top = True, left = True, right = True)
            ax[i].legend(loc = 'upper left' , ncol = 1, prop = dict(weight = 'bold'))
            ax[i].grid(color = 'black', alpha = .5, axis = 'both', which = 'both', linestyle = '--', linewidth = .5)
        
        plt.savefig('Hs-Wspd_Buoy-Satellite_scatter.png', dpi = 300)
        plt.show()


    ## Map figure
    if MAP == True:

        fig, ax = plt.subplots(figsize = (10, 6), facecolor = 'w',
            edgecolor = 'k', constrained_layout = True)

        ax.plot(BOMM1[0], BOMM1[1],'sg', markersize = 12, label = 'BOMM1')
        ax.plot(lon[indx_50km], lat[indx_50km], 'ko', label = 'Satellite')
        ax.plot(-coastline[:,0], coastline[:, 1], '-k', linewidth = 2)

        ax.axhspan(lat_min, lat_max, facecolor = 'gray', alpha = 0.5)

        ax.set_xlabel('Longitude [Degree]', fontweight = 'bold')
        ax.set_ylabel('Latitude [Degree]', fontweight = 'bold')

        ax.axis([lon_min, lon_max, lat_min, lat_max])

        ax2 = plt.axes([0,0,1,1])
        # Manually set the position and relative size of the inset axes within ax1
        ip = InsetPosition(ax, [0.4,0.2,0.5,0.5])
        ax2.set_axes_locator(ip)
        ax2.plot(BOMM1[0], BOMM1[1],'sg', markersize = 12)
        ax2.plot(lon[indx_50km], lat[indx_50km], 'ko')
        ax2.tick_params(which = 'both', direction = 'in', bottom = True, top = True, left = True, right = True)             
        
        ## Legend and grid
        ax.legend(loc = 'upper right' , ncol = 1, prop = dict(weight = 'bold'))
        ax.grid(color = 'black', alpha = .5, axis = 'both', which = 'both', linestyle = '--', linewidth = .5)
        plt.savefig('Map_GoM_Buoy-Satellite.png', dpi = 300)
        plt.show()




