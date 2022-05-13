## ClusterJoint method

Implementation of (https://link.springer.com/article/10.1007/s13349-016-0160-0) with modifications in order to fit in streaming data.

## Expiriments

Use ClusterJoint.py to run expirements for Turbofan or Bus Dataset.

## PARAMETERS 



filename: "vehicles" or "f0001","f0002","f0003","f0004"  filename of dataset to use.

step: Time step (equals to one)

window: Window length in days

SDfactor: the Facotr for multiplication of standard deviation in 2T thresholding technique.

If the filename is "vehicles" use the runsenarioCostBus() function to run experiments.
Otherwise for Turbofan dataset use the runsenarioCost() function.

For every test the hyperparameters and the resulted cost is stored in txt file (different for each dataset). 
Form in txt file: filename | step | window | SDfactor | [[costs for PH=15], [costs for PH=23], [costs for PH=30]] | 


## readRuslts.py

This script is used in order to plot the best cost achived by the method. Read the result from produced txt files.
