import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def make_data_arrays(resispaths,skypaths):

	data1 = np.loadtxt(resispaths[0],delimiter=',',skiprows=2)

	freq_array = data1[:,1]

	resisaud = np.zeros(data1[:,2].size)
	skyaud = np.zeros(data1[:,2].size)

	for i in resispaths:
		tmpdata = np.loadtxt(i,delimiter=',',skiprows=2)
		resisaud += tmpdata[:,2]
	resisaud /= len(resispaths)

	for i in skypaths:
		tmpdata = np.loadtxt(i,delimiter=',',skiprows=2)
		skyaud += tmpdata[:,2]
	skyaud /= len(skypaths)
	n = 1#30

	smooth_sky = np.convolve(skyaud,np.ones(n)/n,mode='valid') 
	smooth_resis = np.convolve(resisaud,np.ones(n)/n,mode='valid') 
	smooth_freq = np.convolve(freq_array,np.ones(n)/n,mode='valid')

	return smooth_freq,smooth_resis,smooth_sky


resisfiles = glob.glob('take_50ohm_data_4/*.csv')
skyfiles = glob.glob('take_21cm_data_4/*.csv')
smooth_freq_4,smooth_resis_4,smooth_sky_4 = make_data_arrays(resisfiles,skyfiles)
ratio_4 = smooth_sky_4/smooth_resis_4

resisfiles = glob.glob('take_50ohm_data_3/*.csv')
skyfiles = glob.glob('take_21cm_data_3/*.csv')
smooth_freq_3,smooth_resis_3,smooth_sky_3 = make_data_arrays(resisfiles,skyfiles)
ratio_3 = smooth_sky_3/smooth_resis_3

resisfiles = glob.glob('take_50ohm_data/*.csv')
skyfiles = glob.glob('take_21cm_data/*.csv')
smooth_freq_1,smooth_resis_1,smooth_sky_1 = make_data_arrays(resisfiles,skyfiles)
ratio_1 = smooth_sky_1/smooth_resis_1

resisfiles = glob.glob('take_50ohm_data_5/*.csv')
skyfiles = glob.glob('take_21cm_data_5/*.csv')
smooth_freq_5,smooth_resis_5,smooth_sky_5 = make_data_arrays(resisfiles,skyfiles)
ratio_5 = smooth_sky_5/smooth_resis_5

resisfiles = glob.glob('take_50ohm_data_6/*.csv')
skyfiles = glob.glob('take_21cm_data_6/*.csv')
smooth_freq_6,smooth_resis_6,smooth_sky_6 = make_data_arrays(resisfiles,skyfiles)
ratio_6 = smooth_sky_6/smooth_resis_6

resisfiles = glob.glob('take_50ohm_data_7/*.csv')
skyfiles = glob.glob('take_21cm_data_7/*.csv')
smooth_freq_7,smooth_resis_7,smooth_sky_7 = make_data_arrays(resisfiles,skyfiles)
ratio_7 = smooth_sky_7/smooth_resis_7

plt.figure()
plt.plot(smooth_freq_1,ratio_1-np.mean(ratio_1),label='f_center = 1420.4 MHz')
plt.plot(smooth_freq_3,ratio_3-np.mean(ratio_3),label='f_center = 1420.7 MHz')
# plt.plot(smooth_freq_4,ratio_4-np.mean(ratio_4))
# plt.plot(smooth_freq_5,ratio_5-np.mean(ratio_5))
# plt.plot(smooth_freq_6,ratio_6-np.mean(ratio_6))
# plt.plot(smooth_freq_7,ratio_7-np.mean(ratio_7))

plt.xlabel('Frequency (MHz)')
plt.ylabel(r'$P_{sky}/P_{50 \Omega}$')

plt.legend()

def fit_linear_plus_periodic(x, y, fit_mask, nfreq=2):
	x_fit = x[fit_mask]
	y_fit = y[fit_mask]
	x0 = np.mean(x_fit)
	xs = x - x0

	lin_coef = np.polyfit(x_fit, y_fit, 1)
	lin_fit = np.polyval(lin_coef, x)
	y_detrended = y - lin_fit
	y_detrended_fit = y_detrended[fit_mask]

	dx = np.median(np.diff(x_fit))
	fft_freq = np.fft.rfftfreq(y_detrended_fit.size, d=dx)
	fft_amp = np.abs(np.fft.rfft(y_detrended_fit))
	fft_amp[0] = 0.0
	n_pick = min(nfreq, fft_amp.size - 1)
	peak_idx = np.argpartition(fft_amp, -n_pick)[-n_pick:]
	peak_idx = peak_idx[np.argsort(fft_amp[peak_idx])[::-1]]
	nu_peaks = fft_freq[peak_idx]

	# Allow each FFT component amplitude/phase to vary linearly across band.
	cols = [np.ones_like(x)]
	for nu in nu_peaks:
		omega = 2.0 * np.pi * nu
		s = np.sin(omega * x)
		c = np.cos(omega * x)
		cols.append(s)
		cols.append(c)
		cols.append(xs * s)
		cols.append(xs * c)
	design = np.column_stack(cols)
	design_fit = design[fit_mask]
	coef, _, _, _ = np.linalg.lstsq(design_fit, y_detrended_fit, rcond=None)
	periodic_fit = design @ coef
	y_resid = y_detrended - periodic_fit
	return lin_fit, periodic_fit, y_detrended, y_resid, nu_peaks

def fit_linear_periodic_plus_gaussian_joint(
	x,
	y,
	fit_mask,
	gauss_center_bounds=(1420.3, 1420.41),
	gauss_sigma_bounds=(0.05, 0.1),
	nfreq=3,
	n_center=220,
	n_sigma=140
):
	x_fit = x[fit_mask]
	y_fit = y[fit_mask]
	x0 = np.mean(x_fit)
	xs = x - x0

	# Seed periodic frequencies from detrended masked data.
	lin_coef = np.polyfit(x_fit, y_fit, 1)
	lin_seed = np.polyval(lin_coef, x)
	y_detrended_seed = y - lin_seed
	y_detrended_fit = y_detrended_seed[fit_mask]

	dx = np.median(np.diff(x_fit))
	fft_freq = np.fft.rfftfreq(y_detrended_fit.size, d=dx)
	fft_amp = np.abs(np.fft.rfft(y_detrended_fit))
	fft_amp[0] = 0.0
	n_pick = min(nfreq, fft_amp.size - 1)
	peak_idx = np.argpartition(fft_amp, -n_pick)[-n_pick:]
	peak_idx = peak_idx[np.argsort(fft_amp[peak_idx])[::-1]]
	nu_peaks = fft_freq[peak_idx]

	# Build x-dependent columns that are linear in coefficients.
	base_cols = [np.ones_like(x), x]
	for nu in nu_peaks:
		omega = 2.0 * np.pi * nu
		s = np.sin(omega * x)
		c = np.cos(omega * x)
		base_cols.append(s)
		base_cols.append(c)
		base_cols.append(xs * s)
		base_cols.append(xs * c)
	base = np.column_stack(base_cols)

	centers = np.linspace(gauss_center_bounds[0], gauss_center_bounds[1], n_center)
	sigmas = np.linspace(gauss_sigma_bounds[0], gauss_sigma_bounds[1], n_sigma)

	best_rss = np.inf
	best_coef = None
	best_center = centers[len(centers) // 2]
	best_sigma = sigmas[len(sigmas) // 2]
	best_gauss = np.zeros_like(x)

	for c0 in centers:
		dx0 = x - c0
		for s0 in sigmas:
			gauss_col = np.exp(-0.5 * (dx0 / s0) ** 2)
			design = np.column_stack([base, gauss_col])
			design_fit = design[fit_mask]
			coef, _, _, _ = np.linalg.lstsq(design_fit, y_fit, rcond=None)
			resid = y_fit - design_fit @ coef
			rss = np.dot(resid, resid)
			if rss < best_rss:
				best_rss = rss
				best_coef = coef
				best_center = c0
				best_sigma = s0
				best_gauss = gauss_col

	model = np.column_stack([base, best_gauss]) @ best_coef
	lin_fit = best_coef[0] + best_coef[1] * x
	periodic_fit = model - lin_fit - best_coef[-1] * best_gauss
	gaussian_fit = best_coef[-1] * best_gauss
	y_resid = y - model
	return lin_fit, periodic_fit, gaussian_fit, y_resid, nu_peaks, best_center, best_sigma, best_coef[-1]

def fit_gaussian_bruteforce(x, y, fit_mask, center_bounds, sigma_bounds, n_center=220, n_sigma=140):
	centers = np.linspace(center_bounds[0], center_bounds[1], n_center)
	sigmas = np.linspace(sigma_bounds[0], sigma_bounds[1], n_sigma)

	best_rss = np.inf
	best_center = centers[len(centers) // 2]
	best_sigma = sigmas[len(sigmas) // 2]
	best_amp = 0.0

	x_fit = x[fit_mask]
	y_fit = y[fit_mask]

	for c in centers:
		dx = x_fit - c
		for s in sigmas:
			g = np.exp(-0.5 * (dx / s) ** 2)
			den = np.dot(g, g)
			if den <= 0:
				continue
			amp = np.dot(y_fit, g) / den
			resid = y_fit - amp * g
			rss = np.dot(resid, resid)
			if rss < best_rss:
				best_rss = rss
				best_center = c
				best_sigma = s
				best_amp = amp

	gauss_full = best_amp * np.exp(-0.5 * ((x - best_center) / best_sigma) ** 2)
	return gauss_full, best_center, best_sigma, best_amp


def process_series(freq, ratio, label,plot_diag=False):
	ratio_centered = ratio #- np.mean(ratio)
	fmin = np.min(freq)
	fmax = np.max(freq)
	edge_mask = (freq >= (fmin + 0.25)) & (freq <= (fmax - 0.25))
	hi_mask = ~((freq >= 1420.1) & (freq <= 1420.7))
	fit_mask = edge_mask & hi_mask

	lin_fit, periodic_fit, y_detrended, y_resid, nu_peaks = fit_linear_plus_periodic(
		freq, ratio_centered, fit_mask, nfreq=3
	)
	gauss_fit_mask = (freq >= 1419.95) & (freq <= 1420.85)
	gaussian_fit, g_center, g_sigma, g_amp = fit_gaussian_bruteforce(
		freq,
		y_resid,
		gauss_fit_mask,
		center_bounds=(1420.05, 1420.75),
		sigma_bounds=(0.005, 0.2),
	)
	y_final = y_resid - gaussian_fit
	print(
		f'{label}: Gaussian fit -> center={g_center:.6f} MHz, sigma={g_sigma:.6f} MHz, amplitude={g_amp:.6e}'
	)
	if plot_diag:
		plt.figure()
		plt.plot(freq, ratio_centered, label=f'{label}: centered ratio')
		plt.plot(freq, lin_fit, label='Linear fit', linewidth=2)
		plt.plot(freq[~fit_mask], ratio_centered[~fit_mask], '.', markersize=2, label='Masked (excluded from fit)')
		plt.xlabel('Frequency (MHz)')
		plt.ylabel(r'$P_{sky}/P_{50 \Omega}$ (centered)')
		plt.xlim(1420.4-0.75,1420.4+0.75)
		plt.legend()

		plt.figure()
		plt.plot(freq, y_detrended, label=f'{label}: detrended (linear removed)')
		plt.xlabel('Frequency (MHz)')
		plt.ylabel('Detrended')
		plt.xlim(1420.4-0.75,1420.4+0.75)
		plt.legend()

		plt.figure()
		plt.plot(freq, y_detrended, label=f'{label}: detrended')
		plt.plot(
			freq,
			periodic_fit,
			label=f'Periodic fit ({nu_peaks.size} FFT terms)',
			linewidth=2
		)
		plt.xlabel('Frequency (MHz)')
		plt.ylabel('Detrended')
		plt.xlim(1420.4-0.75,1420.4+0.75)

		plt.legend()

		plt.figure()
		plt.plot(freq, y_resid, label=f'{label}: residual')
		plt.plot(freq, gaussian_fit, label='Gaussian fit', linewidth=2)
		plt.xlabel('Frequency (MHz)')
		plt.ylabel('Residual')
		plt.xlim(1420.4-0.75,1420.4+0.75)

		plt.legend()

		plt.figure()
		plt.plot(freq, y_final, label=f'{label}: residual after Gaussian subtraction')
		plt.xlabel('Frequency (MHz)')
		plt.ylabel('Residual')
		plt.xlim(1420.4-0.75,1420.4+0.75)
		plt.legend()
		plt.show()

	return y_final

def build_total_model_and_residual(freq, ratio):
	ratio_data = ratio
	fmin = np.min(freq)
	fmax = np.max(freq)
	edge_mask = (freq >= (fmin + 0.25)) & (freq <= (fmax - 0.25))
	hi_mask = ~((freq >= 1420.1) & (freq <= 1420.7))
	fit_mask = edge_mask & hi_mask

	lin_fit, periodic_fit, _, y_resid, _ = fit_linear_plus_periodic(
		freq, ratio_data, fit_mask, nfreq=3
	)
	gauss_fit_mask = (freq >= 1419.95) & (freq <= 1420.85)
	gaussian_fit, _, _, _ = fit_gaussian_bruteforce(
		freq,
		y_resid,
		gauss_fit_mask,
		center_bounds=(1420.05, 1420.75),
		sigma_bounds=(0.005, 0.2),
	)
	total_model = lin_fit + periodic_fit + gaussian_fit
	residual = ratio_data - total_model
	return total_model, residual, gaussian_fit


fit_data_1 = process_series(smooth_freq_1, ratio_1, 'f_center = 1420.4 MHz')
fit_data_3 = process_series(smooth_freq_3, ratio_3, 'f_center = 1420.7 MHz',plot_diag=True)

fit_data_4 = process_series(smooth_freq_4, ratio_4, 'f_center = 1420.4 MHz')
fit_data_5 = process_series(smooth_freq_5, ratio_5, 'f_center = 1420.7 MHz')
fit_data_6 = process_series(smooth_freq_6, ratio_6, 'f_center = 1420.7 MHz')
fit_data_7 = process_series(smooth_freq_7, ratio_7, 'f_center = 1420.7 MHz')

plt.figure()

plt.plot(smooth_freq_1,fit_data_1,label='f_center = 1420.4 MHz')
plt.plot(smooth_freq_3,fit_data_3,label='f_center = 1420.7 MHz')

# plt.plot(smooth_freq_4,fit_data_4,label='f_center = 1420.4 MHz')
# plt.plot(smooth_freq_5,fit_data_5,label='f_center = 1420.7 MHz')
# plt.plot(smooth_freq_6,fit_data_6,label='f_center = 1420.7 MHz')
# plt.plot(smooth_freq_7,fit_data_7,label='f_center = 1420.7 MHz')

plt.xlabel('Frequency (MHz)')
plt.ylabel(r'Residual')

plt.show()

model_1, residual_1, gauss_1 = build_total_model_and_residual(smooth_freq_1, ratio_1)
model_3, residual_3, gauss_3 = build_total_model_and_residual(smooth_freq_3, ratio_3)

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

axs[0].plot(smooth_freq_1, ratio_1, label='Raw data: f_center = 1420.4 MHz', alpha=0.75)
axs[0].plot(smooth_freq_1, model_1, label='Model: linear + periodic + gaussian', linewidth=2)
axs[0].plot(smooth_freq_3, ratio_3, label='Raw data: f_center = 1420.7 MHz', alpha=0.75)
axs[0].plot(smooth_freq_3, model_3, label='Model: linear + periodic + gaussian (1420.7)', linewidth=2)
axs[0].set_ylabel(r'$P_{sky}/P_{50 \Omega}$')
axs[0].set_xlim(1420.4 - 0.75, 1420.4 + 0.75)
axs[0].legend()

axs[1].plot(smooth_freq_1, residual_1, label='Residual: f_center = 1420.4 MHz')
axs[1].plot(smooth_freq_3, residual_3, label='Residual: f_center = 1420.7 MHz')
axs[1].set_xlabel('Frequency (MHz)')
axs[1].set_ylabel('Residual')
axs[1].set_xlim(1420.4 - 0.75, 1420.4 + 0.75)
axs[1].legend()

axs[2].plot(smooth_freq_1, gauss_1, label='Gaussian fit: f_center = 1420.4 MHz')
axs[2].plot(smooth_freq_3, gauss_3, label='Gaussian fit: f_center = 1420.7 MHz')
axs[2].set_xlabel('Frequency (MHz)')
axs[2].set_ylabel('Gaussian')
axs[2].set_xlim(1420.4 - 0.75, 1420.4 + 0.75)
axs[2].legend()

plt.tight_layout()
plt.show()
