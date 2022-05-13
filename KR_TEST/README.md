# DOD method

A distance based method using k and R to determine anomalies

## Expiriments

Use krTestCode.py to run expirements for Turbofan or Bus Dataset.

## PARAMETERS 



filename: "vehicles" or "f0001","f0002","f0003","f0004"  filename of dataset to use.

shiftdays: The shift of the window in days

window: Window length in days

k: the number of neighboars which required for a sample to consider inlier.

R: The Radius.

For more details about the parameters see the original package of Grand. In our expiriments we use constant w_mart=15 and use different values in non_k,metric and Reference_window

If the filename is "vehicles" use the runsenarioCostBus() function to run expiriments.
Otherwise for Turbofan dataset use the runsenarioCost() function.

For every test the hyperparameters and the resulted cost is stored in txt file (diferent for each dataset). 

Form in txt file: filename | shift | window | k | R | normalize | [[costs for PH=15], [costs for PH=23], [costs for PH=30]] | 

## readRuslts.py

This file is used in order to plot the best cost achived by the method. It reads the result from produced txt files.
