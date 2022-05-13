#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 13:17:00 2022

@author: agiannous
"""

# NOTE: run this cell before anything else
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from grand import IndividualAnomalyInductive, IndividualAnomalyTransductive, GroupAnomaly
from grand.datasets import load_artificial, load_vehicles, load_f0001,load_f0002,load_f0003,load_f0004
import re
import grand



def runGrand(filename,non_k,metric,Reference_window,w_mart,normalized):
    print("VERSION:", grand.__version__)
    f002 =[155, 247, 102, 89, 62, 9, 209, 221, 190, 87, 116, 18, 110, 98, 150, 10, 35, 41, 65, 34, 21, 196, 144, 245, 143]
    f001 =[32, 15, 50, 68, 24, 16, 59, 13, 47, 49, 74, 58, 44, 41]
    f003 =[98, 88, 6, 61, 8, 90, 25, 87, 74, 29, 62, 17, 22, 63, 7, 65, 77]
    f004 =[111, 48, 107, 34, 1, 47, 204, 157, 46, 196, 67, 162, 25, 63, 172, 223, 176, 17, 187, 100, 82, 2, 136, 20, 139, 221, 195, 105, 179, 96, 220]
    
    busses=["369","370","371","372","373","374","375","376","377","378","379","380","381","382","383","452","453","454","455"]

    
    if filename=="f0001":
        indexesforgrand=f001
    elif filename=="f0002":
        indexesforgrand=f002
    elif filename=="f0003":
        indexesforgrand=f003
    elif filename=="f0004":
        indexesforgrand=f004
    elif filename=="vehicles":
        indexesforgrand=[0,1,3,4,9,11,12,13,14] #ids of busses where used in our case
    
    if filename=="f0001":
        dataset = load_f0001()
    elif filename=="f0002":
        dataset = load_f0002()
    elif filename=="f0003":
        dataset = load_f0003()
    elif filename=="f0004":
        dataset = load_f0004()
    elif filename=="vehicles":
        dataset = load_vehicles()
    if normalized:
        dataset=dataset.normalize(True,True)
    
   
    
    #dataset.plot()
    #f002 [155, 247, 102, 89, 62, 9, 209, 221, 190, 87, 116, 18, 110, 98, 150, 10, 35, 41, 65, 34, 21, 196, 144, 245, 143]
    #f001 [32, 15, 50, 68, 24, 16, 59, 13, 47, 49, 74, 58, 44, 41]
    #f003 [98, 88, 6, 61, 8, 90, 25, 87, 74, 29, 62, 17, 22, 63, 7, 65, 77]
    #f004 [111, 48, 107, 34, 1, 47, 204, 157, 46, 196, 67, 162, 25, 63, 172, 223, 176, 17, 187, 100, 82, 2, 136, 20, 139, 221, 195, 105, 179, 96, 220]
    
    nb_units = dataset.get_nb_units()  # Number of systems (vehicles)
    ids_target_units = indexesforgrand
    model = GroupAnomaly(nb_units, ids_target_units,w_martingale = w_mart, w_ref_group=Reference_window,dynamic_threshold=False,dynamic_error_presentage=0.000005,historic_errors=2000, non_conformity=metric,k = non_k)
    results=[]
    Dates=[]
    
    for dt, x_units in dataset.stream():
        infos = model.predict(dt, x_units)
        #info is array with :[[DeviationContext(strangeness=0, pvalue=0.5, deviation=0, is_deviating=False),..] ,...]
        Dates.append(dt)
        results.append(infos)
        if str(infos[0])!="DeviationContext(strangeness=0, pvalue=0.5, deviation=0, is_deviating=False)":
            print("Time: {}".format(dt), end="\n", flush=True)
        
    return model,ids_target_units

def plotmodel(model):
    model.plot_deviations(figsize=(13, 5), plots=["deviation","strangeness"])#, "threshold"])
    

def plotmodel2(model,ids_target_units):
    
    fig, axis = plt.subplots(len(ids_target_units))
    c=0
    busses=["369","370","371","372","373","374","375","376","377","378","379","380","381","382","383","452","453","454","455"]
    
    for uid in ids_target_units:
        #busses=["369","370","371","372","373","374","375","376","377","378","379","380","381","382","383","452","453","454","455"]
        axis[c].set_ylabel(busses[uid])
        axis[c].set_ylim(0, 1)
        T, P, M ,thresh,deviatingvalues,deviatingTimes,NumberOfN,NumberOfREf=model.get_information(uid)
        if True:
            axis[c].plot(T, M)
        plotLines(axis[c],uid,"busFailures/")
        c+=1


# used when we plot results for Bus Dataset to plot lines when failure occur
def plotLines(ax,uid,path="busFailures/"):
    busses=["369","370","371","372","373","374","375","376","377","378","379","380","381","382","383","452","453","454","455"]
    bus=busses[uid]
    filepath2 = path+'Blueoutliers/outlier'+bus+'.txt'
    #red outliers
    dateredoutlier=[]  
    with open(path+"redouliers.txt", 'r') as f: 
        Lines = f.readlines() 
        for line in Lines: 
            split = line.split(" ")
            if split[0]==bus :
                dateRed= datetime.strptime(split[1].strip(), '%Y-%m-%d') 
                dateredoutlier.append(dateRed)
    dateredoutlierDash=[]
    
    #DashRed
    with open(path+"rediutliersDASH.txt", 'r') as f: 
        Lines = f.readlines() 
        for line in Lines: 
            split = line.split(" ")
            if split[0]==bus :
                dateRed= datetime.strptime(split[1].strip(), '%Y-%m-%d') 
                dateredoutlierDash.append(dateRed)

    # onlyforcheck contains all failures
    onlyforcheck=[]
    for date in dateredoutlier:
        onlyforcheck.append(date)
    for date in dateredoutlierDash:
        onlyforcheck.append(date)
    #Blueoutliers
    Blueoutlier = [] 
    with open(filepath2, 'r') as f: 
        file = f.read() 
        x = re.findall('\d\d\d\d-\d\d-\d\d', file) 
        for item in x: 
            try: 
                date = datetime.strptime(item, '%Y-%m-%d') 
                Blueoutlier.append(date) 
            except: 
                pass  
    for out in dateredoutlier:
        ax.axvline(out , color='r')
    for out in dateredoutlierDash:
        ax.axvline(out,color='r',dashes=[2, 2])
    for out in Blueoutlier:
        ax.axvline(out ,color='royalblue',dashes=[2, 2])    



# In bus Dataset we have multiple types of failures (which has different cost),
# this function find for each class of failure the false positives and True positives
def redAndBLueOutliers(reported,path,uid,PH,redSolidFnCost,redDashFnCost,BlueDashFnCost,TpCost,FpCost):
    busses=["369","370","371","372","373","374","375","376","377","378","379","380","381","382","383","452","453","454","455"]
    bus=busses[uid]
    filepath2 = path+'Blueoutliers/outlier'+bus+'.txt'
    #red outliers
    dateredoutlier=[]  
    with open(path+"redouliers.txt", 'r') as f: 
        Lines = f.readlines() 
        for line in Lines: 
            split = line.split(" ")
            if split[0]==bus :
                dateRed= datetime.strptime(split[1].strip(), '%Y-%m-%d') 
                dateredoutlier.append(dateRed)
    dateredoutlierDash=[]
    
    #DashRed
    with open(path+"rediutliersDASH.txt", 'r') as f: 
        Lines = f.readlines() 
        for line in Lines: 
            split = line.split(" ")
            if split[0]==bus :
                dateRed= datetime.strptime(split[1].strip(), '%Y-%m-%d') 
                dateredoutlierDash.append(dateRed)

    # onlyforcheck contains all failures
    onlyforcheck=[]
    for date in dateredoutlier:
        onlyforcheck.append(date)
    for date in dateredoutlierDash:
        onlyforcheck.append(date)
    #Blueoutliers
    Blueoutlier = [] 
    with open(filepath2, 'r') as f: 
        file = f.read() 
        x = re.findall('\d\d\d\d-\d\d-\d\d', file) 
        for item in x: 
            try: 
                date = datetime.strptime(item, '%Y-%m-%d') 
                Blueoutlier.append(date) 
            except: 
                pass 
     
    ## count False positi
    fp=0
    for rep in reported:
        nearestRed=min([ abs((rep-bd).days) for bd in onlyforcheck ])
        # if report is made PH befroe or after the failure in't fp (ingored or Trupositive)
        if nearestRed<PH: continue
        #take all day distances from blue outliers, if the min of them is smaller than Then it is TP
        distfromblue=[ (bd-rep).days for bd in onlyforcheck if (bd-rep).days>0]
        if len(distfromblue)>0:
            nearestBlue=min(distfromblue)
            if nearestBlue<PH:continue
        fp+=1
        
    
    #Red solid TP/FN:
    reportedREds=[]
    for redfail in dateredoutlier:
        for rep in reported:
            if (redfail-rep).days<PH and (redfail-rep).days>0:
                reportedREds.append(redfail)
                break
    redtp=len(reportedREds)
    redfn=len(dateredoutlier)-len(reportedREds)
    
    #Red dash TP/FN:
    reportedREds=[]
    for redfail in dateredoutlierDash:
        for rep in reported:
            if (redfail-rep).days<PH and (redfail-rep).days>0:
                reportedREds.append(redfail)
                break
    redDashtp=len(reportedREds)
    redDashfn=len(dateredoutlierDash)-len(reportedREds)
    
    #blue dash TP/FN:
    reportedBlue=[]
    for bluefail in Blueoutlier:
        for rep in reported:
            if (bluefail-rep).days<PH and (bluefail-rep).days>0:
                reportedBlue.append(bluefail)
                break
    BlueDashtp=len(reportedBlue)
    BlueDashfn=len(Blueoutlier)-len(reportedBlue)
    
    #redSolidFnCost,redDashFnCost,BlueDashFnCost,TpCost,FpCost
    cost=redSolidFnCost*redfn+redDashFnCost*redDashfn+BlueDashfn*BlueDashFnCost+fp*FpCost+(redtp+redDashtp+BlueDashtp)
    
    return cost

# Calculate costs for multiple Predctive horizon and FN cost (fleet bus Dataset).
def calculateCostBus(ids_target_units,model,thh):
    fpcost=1
    fncost=10
    tpcost=1

    ######
    #plt.figure(2)
    phcost=[]
    for PH in [15,23,30]:
        Cost=[]
        for fncost in  [4,5,10,20]:
            BlueDashFnCost=fncost
            redDashFnCost=5*fncost
            redSolidFnCost=10*fncost
            TpCost=1
            FpCost=1
            
            cost=0
            for uid in ids_target_units:
                T, P, M ,thresh,deviatingvalues,deviatingTimes,_,_=model.get_information(uid)
                deviatingTimes=[t[0] for t in zip(T,M) if t[1]>thh ]
                costofUid=redAndBLueOutliers(deviatingTimes,"busFailures/",uid,PH,redSolidFnCost,redDashFnCost,BlueDashFnCost,TpCost,FpCost)
                cost+=costofUid
            Cost.append(cost)
        phcost.append(Cost)
    return phcost

def plotgrand(model,ids_target_units,thh):
    fig, axis = plt.subplots(len(ids_target_units))
    c=0
    for uid in ids_target_units:
        #busses=["369","370","371","372","373","374","375","376","377","378","379","380","381","382","383","452","453","454","455"]
        axis[c].set_ylabel(uid)
        axis[c].set_ylim(0, 1)
        T, P, M ,thresh,deviatingvalues,deviatingTimes,_,_=model.get_information(uid)
        deviatingTimes=[t[0] for t in zip(T,M) if t[1]>thh ]
        deviatingValues=[t[1] for t in zip(T,M) if t[1]>thh ]
        if True:
            axis[c].plot(T, M)
            if True:#self.dynamic_threshold:
                axis[c].scatter(deviatingTimes, deviatingValues, color="red")
        if False:
            axis[c].axhline(y=thresh, color='r', linestyle='--', label="Threshold")
        c+=1


# Calculate costs for multiple Predctive horizon and FN cost (fleet Trubofan Dataset).
def calculateCost(ids_target_units,model,thh):
    fpcost=1
    fncost=10
    tpcost=1
    
    #plt.figure(2)
    phcost=[]
    for PH in [15,23,30]:
        Cost=[]
        for fncost in  [5,10,30,50,100]:
            cost=0
            for uid in ids_target_units:
                # P=values
                tp=0
                fp=0
                fn=0
                T, P, M ,thresh,deviatingvalues,deviatingTimes,_,_=model.get_information(uid)
                deviatingTimes=[t[0] for t in zip(T,M) if t[1]>thh ]
                for dd in deviatingTimes:
                    if (T[-2]-dd).days <PH and (T[-2]-dd).days>=0:
                        tp+=1
                    elif (T[-2]-dd).days>=0:
                        fp+=1
                if tp>0:
                    tp=1
                fn=1-tp
                cost+=fn*fncost+tp*tpcost+fp*fpcost
            Cost.append(cost)
        phcost.append(Cost)
    return phcost

# Run Fleet Turbofan dataset
def senarioCost(filename,non_k,metric,Reference_window,w_mart,normalized):
    model,ids_target_units=runGrand(filename,non_k,metric,Reference_window,w_mart,normalized)
    #plotmodel(model)
    for thh in [0.4,0.5,0.6,0.7,0.8]:
        Cost=calculateCost(ids_target_units,model,thh)
        #print(Cost)
        plotgrand(model,ids_target_units,thh)
        towrite=[filename,non_k,metric,Reference_window,w_mart,normalized,thh,Cost]
        with open(f'{filename}_GRAND_RESULTS_Cost.txt', 'a') as f:
            for item in towrite:
                f.write("%s | " % item)
            f.write("\n")
# Run Bus Dataset
def senarioCostBuss(filename,non_k,metric,Reference_window,w_mart,normalized):
    model,ids_target_units=runGrand(filename,non_k,metric,Reference_window,w_mart,normalized)
    #plotmodel2(model,ids_target_units)
    for thh in [0.4, 0.5, 0.6, 0.7, 0.8]:
        Cost=calculateCostBus(ids_target_units,model,thh)
        #print(Cost)
        towrite=[filename,non_k,metric,Reference_window,w_mart,normalized,thh,Cost]
        with open(f'{filename}_GRAND_RESULTS_Cost.txt', 'a') as f:
            for item in towrite:
                f.write("%s | " % item)
            f.write("\n") 



###### PARAMETERS ##############
filename="vehicles" #filename of expirement
non_k=20 # k for lof or knn metric
metric="median" # non-conformity measure (median,knn,lof)
Reference_window="15days" #Peer Group in days
w_mart = 15         # Window size for computing the deviation level
normalized=False
if metric=="median":
    non_k=0
################################

#senarioCost(filename,non_k,metric,Reference_window,w_mart,normalized)
senarioCostBuss(filename,non_k,metric,Reference_window,w_mart,normalized)

