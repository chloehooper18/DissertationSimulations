# Simulations worksheet
These notebooks walk through a series of simulations of time series and their power spectra, and their spectral models by focusing on simulating white, pink, and brown noise. These notebooks start with single simulations and models and builds up to generating and fitting multiple signals.

## 01 - SimSetup_TimeSeries
### Simulating and plotting time series
- Simulate the following 3 time series: white noise, pink noise, and brown noise
- Plot the time series for each, and compare

## 02 - SimSetup_PowerSpectra
### Computing and plotting power spectra
- Compute power spectra for each time series
- Plot the power spectra together, how do they relate to each other
- How do the power spectra vary when plotted with log vs linear freqs & powers

## 03 -SimSetup_SpectralModels
### Initialising and fitting spectral models to the power spectra
- Initialise and fit spectral models to the power spectra
- Extract the computed aperiodic exponents. Do these match the simulated values

## 04 - SimExplores_DataLength
### Differing data length
- Explores the effect of different lengths of data on the PSDs & fit models

## 05 - SimExplores_SamplingRates
### Differring sampling rates
- Explores the effect of different sampling rates on the PSDs & fit models

## 06- SimExplores_ModelSettings
### Different model settings
- Explores the different settings you can initialize the model with, how do they affect fitting

## 07 - SimManipulations_AddOscillations
### Adding an oscillation
- Explores what happens / changes if you add an oscillation to the simulated time series?

## 08 -SimManipulations_AddPeaks
### Adding peaks
- Explores how many peaks can you add before the model fitting gets weird
- Explores how this interacts with model settings

## 09 - SimCombinations_AvgTimeSeries
### Averaging together time series
- For the {white, pink, brown} simulations, if you average together the time series, compute a power spectra and fit a spectral model, what is the resultant exponent?
- Explores how this relates to the simulated value(s)

## 10 - SimCombinations_AvgPowerSpectra
### Averaging together power spectra
- If you average together the power spectra and fit a spectral model, what is the resultant exponent?
- Explores how this relates to the simulated value(s)

## 11 - SimCombinations_FittingMultipleSignals
### fitting multiple power spectra to a spectral model
- For a given exponent, if you simulate M versions, and fit them all, what is the distribution of fit values (aka how variable is it).
- Explores how this relates to different simulation & fitting parameters?
