# Simulations - single participant, 1D and 2D electrode lines 
These notebooks simulation both single and multiple electrodes for a single participant. These begin with simulating the Cz electrode which serves as a reference for all following electrode simulations. 

## 01 - electrode_simulation
### Simulating an electrode for a single participant when the signal is made up of:
- Aperiodic noise only
- Aperiodic and periodic noise

## 02 - 1D_electrode_lines
### Simulating 1D lines of electrodes 
- Simulating electrodes where the exponent is -1 for each
- Explore what happens to the output exponent when you average:
    - The time series
    - The PSDs
    - The exponent outputs from the model fit
- Simulate this for both aperiodic and full electrode signals

## 03 - 1D_electrodes_ap
### Simulating 1D lines of electrodes 
- Simulating electrodes where the exponent is a random value between -1 and -2
- Average the PSD and plot the Spectral Model
- Simulate this for aperiodic signals only 

## 04 - 2D_electrodes_lines
### Simulating 2D lines of electrodes 
- Simulating electrodes where the exponent is -1 for each
- Explore what happens to the output exponent when you average:
    - The time series
    - The PSDs
    - The exponent outputs from the model fit
- Simulate this for both aperiodic and full electrode signals

## 05 - 2D_electrodes_ap
### Simulating 2D lines of electrodes 
- Simulating electrode clusters where the exponent is a random value between -1 and -2
- Average the PSD and plot the Spectral Model
- Simulate this for aperiodic signals only
