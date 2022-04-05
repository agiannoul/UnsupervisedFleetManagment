# TranAD 

Provided in (https://github.com/imperial-qore/TranAD.git)
TranAD model with modifications to fit in streaming data.

# Expiriments
Use fleetTranAD.py to run expirements for Turbofan or Bus Dataset.

## PARAMETERS 

In main section you can use test_forBussed() or test_for_f000(filename,factor) to test expiriments.

filename: "vehicles" or "f0001","f0002","f0003","f0004"  filename of dataset to use.

shiftdays : days of shift for time Window (shiftdays=window/2)
window : days of Time Window
factor: factor based on which the threshold is calculated

In test_forBussed() factor parameter is missing, because we test several factors with one pass.


## readRuslts

This file is used in order to plot the best cost achived by the method.
