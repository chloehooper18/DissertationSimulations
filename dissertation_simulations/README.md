# Dissertation Simulations functions
This folder contains a list of function files that have been created. The aims of the contents of the functions contained within each file are listed below:

## eeg_data
- Load EEG data, select desired electrodes, compute PSDs, average PSDs together, plot Spectral model
- Sweep across user-defined electrode groups and estimate the exponent from PSDs averaged across each selected group.
- Load EEG data, select desired electrodes, compute PSDs, plot individual Spectral Models, average the resulting aperiodic exponents
- Sweep across user-defined electrode groups and estimate the exponent from PSDs averaged across each selected group.

## electrode_signal_simulations
- Simulating electrode signal that contains aperiodic noise only
- Simulating electrode signal that contains aperiodic and periodic noise

## exponents
- Generate random exponents within a given range
- Generate random exponents for simulated electrodes and specify the electrode distribution type
- Generate a 2D matrix of exponents for electrode simulations
- Generate a spatially varying exponent matrix

## generate_electrodes
- Generate multiple aperiodic signals with different exponents
- Generate multiple "full" signals with different exponents
- Generate a 2D grid of simulated aperiodic (1/f) EEG signals

## loading_summaries
- Load one sweep summary JSON file

## params
- A set of dicts that contain consistent parameters for generating electrode signals

## plotting
- Plot sweep results with optional error bars
- Plot saved exponent distributions as styled box plots
- Add significance brackets to box plots
- Add significant within-panel Tukey brackets and preserve axis space
- Calculate Cohen's D: group 1 mean minus group 2 mean
- Run Tukey HSD and add Cohen's d to every comparison

## psd
- Compute PSDs for multiple signals, plot them, and plot the average PSD. Works when electrodes are generated as a 1D array
- Compute PSDs for a 2D electrode grid, plot them, and plot the average PSD
- Simulate electrode signals, compute and average power spectral densities (PSDs), fit a spectral model to the averaged PSD, and return a distribution of estimated aperiodic exponents across repeated simulations.

## spec_mod
- Fit spectral models to PSDs and compute aperiodic exponents. Works when electrodes are generated as a 1D array
- Simulate electrode signals, compute individual PSDs, fit spectral models to each electrode separately, average the resulting aperiodic exponents, and return a distribution across repeated simulations.

## time_series
- Plot multiple time series as separate subplots. Works when electrodes are generated as a 1D array
- Compute and plot the average time series across multiple signals. Works when electrodes are generated as a 1D array
- Plot each signal from a 2D grid of electrodes as separate time series
-  Compute and plot the average time series across a 2D electrode grid
-  Simulate electrode signals, compute and average the time series, compute the PSD, fit a spectral model to the PSD, and return a distribution of estimated  aperiodic exponents across repeated simulations.
