# ClusterJoint method

Implementation of (link to paper) with modifications in order to fit in streaming data.

#expiriments
Use ClusterJoint.py to run expirements for Turbofan or Bus Dataset.

# PARAMETERS 



filename: "vehicles" or "f0001","f0002","f0003","f0004"  filename of dataset to use.
step: Time step (equals to one)
window: Window length in days
SDfactor: the Facotr for multiplication of standard deviation in 2T thresholding technique.

If the filename is "vehicles" use the runsenarioCostBus() function to run expiriments.
Otherwise for Turbofan dataset use the runsenarioCost() function.

For every test the hyperparameters and the resulted cost is stored in txt file (diferent for each dataset). 


#readRuslts

This file is used in order to plot the best cost achived by the method.
