import os                              # a library that allows us access to basic operating system commands
import glob                            # a library that helps us search for files
import datetime                        # a library that allows us to work with dates and times
import warnings                        # a library that supports managing warning messages

import xarray as xr                    # a library that helps us work efficiently with multi-dimensional arrays
import numpy as np                     # a library that lets us work with arrays; we import this with a new name "np"
warnings.filterwarnings("ignore")


def get_chlblend_phytotype(ds_l2,OC4_mask,G2B_mask,MEDGlint_mask):

    ## MERIS (OC4E) coefficients from NOMAD:
    A_0 = 0.3255
    A_1 = -2.7677
    A_2 = 2.4409
    A_3 = -1.1288
    A_4 = -0.4990
    
    a,b = OC4_mask.shape
    mratio =  np.ndarray([a,b,3]) * np.nan    
    mratio[:,:,0] = (ds_l2["Oa03_reflectance"]/ds_l2["Oa06_reflectance"]).values
    mratio[:,:,1] = (ds_l2["Oa04_reflectance"]/ds_l2["Oa06_reflectance"]).values
    mratio[:,:,2] = (ds_l2["Oa05_reflectance"]/ds_l2["Oa06_reflectance"]).values
    x = np.log10(mratio.max(axis=2))
    zzz = mratio.max(axis=2)   
    
    ocv4 = 10**(A_0 + (A_1*x) + (A_2*(x**2)) + (A_3*(x**3)) + (A_4*(x**4)));         
    
    OC4 = ocv4.real       
    OC4[OC4 > 100] = 100
    OC4[OC4 < 0] = 0.009
    OC4_masked = OC4 * OC4_mask
    
    ## Define the test chla algorithms:
    g2band = ((35.75*((ds_l2["Oa11_reflectance"]/ds_l2["Oa08_reflectance"]).values))-19.3)**1.124
       
    g2b = g2band.real
    g2b [g2b < 0] = 0.009
    g2b [g2b > 300] = 300
    g2b_masked = g2b * G2B_mask
    
    mrub = (ds_l2["Oa07_reflectance"]/ds_l2["Oa06_reflectance"]).values
    ratio2 = (ds_l2["Oa11_reflectance"]/ds_l2["Oa08_reflectance"]).values
    
    ## EXCEPTION FOR WHEN RATIO MAY FAIL:
    ext_bad = (ratio2>0.75) & (((ds_l2["Oa08_reflectance"]).values/np.pi)>0.01)
    baddata = (ratio2 > 1.2) & (((ds_l2["Oa08_reflectance"]).values/np.pi)< 0.0005)
    baddata2 = ((((ds_l2["Oa08_reflectance"]).values)/np.pi)< 0)
    baddata3 = (np.isnan(MEDGlint_mask)==1) & (ratio2>0.75)  ## <-------------------------------------new
    baddata4 = (zzz>1) & (ratio2>0.75) & (mrub<0.8)  ## <-------------------------------------new
    tester = np.zeros((a,b))
    tester[baddata]=1
    tester[baddata2]=1
    tester[baddata3]=1   ## <-------------------------------------new
    tester[baddata4]=1   ## <-------------------------------------new
    tester[ext_bad]=1
    
    ## APPLY THE RATIO -BASED BLENDING PROCEDURE:
    ratio_llim = 0.75
    ratio_ulim = 1.15
    ratio2[ratio2 < ratio_llim] = ratio_llim
    ratio2[ratio2 > ratio_ulim] = ratio_ulim
    red_weight = ((ratio2 - ratio_llim)/(ratio_ulim - ratio_llim))-tester   #weighting for the red-band algo       
    red_weight[red_weight<0]=0        ## <-------------------------------------new
    red_weight[red_weight>1]=1        ## <-------------------------------------new
    
    blue_weight = np.absolute(red_weight - 1)   
    
    chl_blend = np.nansum([(blue_weight*OC4_masked),(red_weight*g2b_masked)],axis=0) * G2B_mask
    
    R665 = (ds_l2["Oa08_reflectance"]).values * G2B_mask
    R681 = (ds_l2["Oa10_reflectance"]).values* G2B_mask
    R709 = (ds_l2["Oa11_reflectance"]).values * G2B_mask
    R753 = (ds_l2["Oa12_reflectance"]).values * G2B_mask
    
    ## TURBIDITY FLAG:
    integral = (R753 * 88.75) + (0.5*88.75*(R665 - R753))
        
    ## SETUP THE DINOFLAGELLATE, PSEUDO-NITZSCHIA:   
    flh = (R681 - (1.005*(R665 + ((R753 - R665)*((681.25-665)/(753.75-665))))))
    mci = (R709 - (1.005*(R665 + ((R753 - R665)*((708.75-665)/(753.75-665))))))
        
    ## RED TIDE INDEX:
    rti = mci/flh
    
    peak = np.copy(flh)
    peak[rti>=1]=mci[rti>=1]
    
    peakwave = np.ones((a,b))*681.25
    peakwave[rti>=1]=708.75
    
    mesocond   = (peak>=0.0016) & (integral<0.5) & (mrub>1) & ((ds_l2["Oa06_reflectance"]).values>0)
    mixcond    = (rti<0.6)  & (peak>=0.0016) & (integral<0.5)
    mixhicond  = (rti<0.6)  & (peak>=0.0022) & (integral<0.5)
    psncond    = (rti<0.6) & (peak>0.003)  & (integral<0.5)
    dinolocond = (rti>0.6)  & (peak>0.0016)  & (integral<0.5)
    dinohicond = (rti>0.6)  & (peak>0.003)   & (peakwave>682) & (R709>R665)
    
    phytoflag = np.zeros((a,b))
    phytoflag[mixcond]=1
    phytoflag[mixhicond]=2
    phytoflag[psncond]=3
    phytoflag[dinolocond]=5
    phytoflag[dinohicond]=6
    phytoflag[mesocond]=4
    all_phyto = np.copy(phytoflag) * G2B_mask

    return all_phyto, chl_blend
