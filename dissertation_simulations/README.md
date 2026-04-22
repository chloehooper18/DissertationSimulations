# Dissertation Simulations functions
This folder contains a list of function files that have been created. The aims of the contents of the functions contained within each file are listed below:

## electrode_signal_simulations
- Simulating electrode signal that contains aperiodic noise only
- Simulating electrode signal that contains aperiodic and periodic noise

## exponents
- Generate random exponents within a given range

## generate_electrodes
- Generate multiple aperiodic signals with different exponents
- Generate multiple "full" signals with different exponents
- Generate a 2D grid of simulated aperiodic (1/f) EEG signals

## params
- A set of dicts that contain consistent paramaters for generating electrode signals

## psd
- Compute PSDs for multiple signals, plot them, and plot the average PSD. Works when electrodes are generated as a 1D array
- Compute PSDs for a 2D electrode grid, plot them, and plot the average PSD

## spec_mod
- Fit spectral models to PSDs and compute aperiodic exponents. Works when electrodes are generated as a 1D array

## time_series
- Plot multiple time series as separate subplots. Works when electrodes are generated as a 1D array
- Compute and plot the average time series across multiple signals. Works when electrodes are generated as a 1D array
- Plot each signal from a 2D grid of electrodes as separate time series
-  Compute and plot the average time series across a 2D electrode grid
