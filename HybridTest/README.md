# 2Stage 

2Stage is an aproach of Grand method using a post processing step

In order to implement 2-Stage algoprithm some modification are made in grand package.

## Expiriments
Use HybridTestFP.py to run expirements for Turbofan or Bus Dataset.

## PARAMETERS 
filename: "vehicles" or "f0001","f0002","f0003","f0004"  filename of dataset to use.

non_k: k for lof or knn metric

metric: "median","knn","lof"  non-conformity measure used for strangeness

Reference_window: "15days" Peer Group in days

w_mart:  Window size for computing the deviation level

R: the distance limit for a sample to consider neighbor in the post-processing-step.

th: Threshold used in deviation level of Grand method to produce anomalies. We test multiple thresholds at each run.

There two addition thrshold for post-processing-step Tin and Tout. We test several of those for each run of the method in order to test multple parameters at once.


If the filename is "vehicles" use the senarioCostBuss() function to run expiriments.
Otherwise for Turbofan dataset use the senarioCost() function.

For every test the hyperparameters and the resulted cost is stored in txt file (diferent for each dataset).

Form in txt file: filename | non_k | metric | Reference_window | w_mart | normalized | th | Tin | Tout | [[costs for PH=15], [costs for PH=23], [costs for PH=30]] | 


## readRuslts.py

This script is used in order to plot the best cost achived by the method. Read the result from produced txt files.
