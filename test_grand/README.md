#Grand

Provided in (https://github.com/caisr-hh/group-anomaly-detection)

In order to implement 2-Stage algoprithm some modification are made in grand package.

#expiriments
Use TestGrand.py to run expirements for Turbofan or Bus Dataset.

# PARAMETERS 
filename: "vehicles" or "f0001","f0002","f0003","f0004"  filename of dataset to use.
non_k: k for lof or knn metric
metric: "median","knn","lof"  non-conformity measure used for strangeness
Reference_window: "15days" Peer Group in days
w_mart:  Window size for computing the deviation level

For more details about the parameters see the original package of Grand. In our expiriments we use constant w_mart=15 and use different values in non_k,metric and Reference_window

If the filename is "vehicles" use the senarioCostBuss() function to run expiriments.
Otherwise for Turbofan dataset use the senarioCost() function.

For every test the hyperparameters and the resulted cost is stored in txt file (diferent for each dataset). 


#readRuslts

This file is used in order to plot the best cost achived by the method.
