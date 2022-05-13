# TranAD 

Provided in (https://github.com/imperial-qore/TranAD.git)
TranAD model with modifications to fit in fleet streaming data.

Plus an extention:
1) Use clustering before training TranAD model
2) Calculate smart threshold.

## Expiriments
Use fleetTranAD.py to run expirements for Turbofan or Bus Dataset.

## PARAMETERS 

In main section you can use test_forBussed() or test_for_f000(filename,factor) to test expiriments.

filename: "vehicles" or "f0001","f0002","f0003","f0004"  filename of dataset to use.

shiftdays : days of shift for time Window (shiftdays=window/2)

window : days of Time Window

factor: factor based on which the threshold is calculated

In test_forBussed() factor parameter is missing, because we test several factors with one pass.

For every test the hyperparameters and the resulted cost is stored in txt file (diferent for each dataset).

Form in txt file: filename | shiftdays | window | factor | [[costs for PH=15], [costs for PH=23], [costs for PH=30]] | 

## readRuslts.py

This script is used in order to plot the best cost achived by the method. Read the result from produced txt files.
