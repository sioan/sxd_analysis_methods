import h5py
import numpy as np
from scipy.optimize import curve_fit
#from scipy.stats import binned_statistic_dd
from scipy.stats import binned_statistic_dd
import pickle
from scipy import stats
from scipy.stats.mstats import theilslopes
from scipy.signal import medfilt
from scipy.signal import savgol_filter
import os

import time, sys
from IPython.display import clear_output

def update_progress(progress):
	bar_length = 20
	if isinstance(progress, int):
		progress = float(progress)
	if not isinstance(progress, float):
		progress = 0
	if progress < 0:
		progress = 0
	if progress >= 1:
		progress = 1

	block = int(round(bar_length * progress))
	clear_output(wait = True)
	text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
	print(text)

def get_edge_position(background_subtracted,savgol_win_size = 51,fit_range=10):
	recalculated_atm = []
	recalculated_atm_cov = []
	def peakFunction(x,a,x0,offset):
		return a*(x-x0)**2+offset
	for i in np.arange(len(background_subtracted)):

		update_progress(i / background_subtracted.shape[0])

		filtered_signal = savgol_filter(background_subtracted[i],savgol_win_size,2,1)      #calculates the rolling slope over 25 points
		win_c = np.argmax(np.abs(filtered_signal))
		initial_guess = [1,win_c,filtered_signal[win_c]]
		try:
        #fitting peak with a parabola
			popt,pcov = curve_fit(peakFunction,np.arange(win_c-fit_range,win_c+fit_range+1),np.abs(filtered_signal[win_c-fit_range:win_c+fit_range+1]), p0=initial_guess)


			recalculated_atm.append(popt[1])
			recalculated_atm_cov.append(pcov[1,1])
		except (RuntimeError,ValueError):
			recalculated_atm.append(-999.0)
			recalculated_atm_cov.append(-999.0)

	update_progress(1)
	return np.array(recalculated_atm), np.array(recalculated_atm_cov)


def get_background_coefficients(X,dropped_shot_mask,signal_of_interest_mask,svd_size=15):

	print("getting background coefficients")
	atm_backgrounds = X[dropped_shot_mask.astype(bool)]
	print("dropped shots selected")
	print(str(atm_backgrounds.shape))

	
	#singular value decomposition on background shots. variable v contains the eigen backgrounds
	#u,s,v = np.linalg.svd(atm_backgrounds) 
	to_eig = np.dot(atm_backgrounds.transpose(),atm_backgrounds)
	print("transpose dot product")

	eig_val,eig_vec = np.linalg.eig(to_eig)
	print("eigen values calculated")
	v = eig_vec.transpose()


	#background_subtracted = my_dict[time_camera] - dot(dot(my_dict[time_camera][:,my_mask],pinv(v[:svd_size][:,my_mask])),v[:svd_size])
	my_mask = signal_of_interest_mask

	return v,np.dot(X[:,my_mask],np.linalg.pinv(v[:svd_size][:,my_mask]))

def subtract_background(X,dropped_shot_mask,signal_of_interest_mask,svd_size=15):

	print("subtracting background")

	v,coefs = get_background_coefficients(X,dropped_shot_mask,signal_of_interest_mask,svd_size=15)

	background_subtracted = X - np.dot(coefs,v[:svd_size])

	return background_subtracted
