#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 11:36:23 2022

@author: agiannous
"""
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join, dirname
import datetime as dtime
from datetime import datetime
import numpy as np
from sklearn.neighbors import KDTree
from grand.utils import NoRefGroupError
from scipy.stats import norm
from math import sqrt
from sklearn.neighbors import NearestNeighbors


def normalize(dfs, with_mean=True, with_std=True):
        if with_mean:
            dfs = [df - df.mean() for df in dfs]

        if with_std:
            stds = [df.std() for df in dfs]
            for std in stds: std[std==0] = 1
            dfs = [df / std for df, std in zip(dfs, stds)]

        return dfs

def loaddflistBus(filename="vehicles"):
    data_path="../"+filename+"/"
    files_csv = sorted([ join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f)) ])
    file_count = len(files_csv)
    # create empty list
    dataframes_list = []
     
    # append datasets to the list
    names=[]
    for i in range(file_count):
        temp_df = pd.read_csv(files_csv[i],index_col=0)
        
        names.append(files_csv[i])
        temp_df.index=pd.to_datetime(temp_df.index)
        temp_df=temp_df.dropna()
        dataframes_list.append(temp_df)
    print(len(dataframes_list))
    return dataframes_list

def loaddflist(filename="f0001"):
    data_path="../"+filename+"/"
    files_csv = sorted([ join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f)) ])
    file_count = len(files_csv)
    # create empty list
    dataframes_list = []
     
    # append datasets to the list
    names=[]
    for i in range(file_count):
        temp_df = pd.read_csv(files_csv[i])
        names.append(files_csv[i])
        temp_df['Artficial_timestamp']=pd.to_datetime(temp_df['Artficial_timestamp'])
        temp_df=temp_df.set_index('Artficial_timestamp')
        dataframes_list.append(temp_df)
        
    print(len(dataframes_list))
    return dataframes_list
    #dataframes_list[0].head()


# Distance-based outlier detection, using Bus dataset
def detectionBus(window,shiftdays,k,R,dataframes_list):
    
    outliers=[ [] for i in range(len(dataframes_list))]

    
    minlist=[ dfff.index.min() for dfff in dataframes_list]
    #print(minlist)
    maxlist=[ dfff.index.max() for dfff in dataframes_list]
    startingDate=min(minlist)
    daysss=(max(maxlist)-startingDate).days+1
    numberOfwindows=int(daysss/shiftdays)
    for w in range(numberOfwindows):
        #print(w)
        start = startingDate + dtime.timedelta(days=w*shiftdays)
        end = start + dtime.timedelta(days=window)
        windowdf=dataframes_list[0][start:end]
        for i in range(1,len(dataframes_list)):
            windowdf=windowdf.append(dataframes_list[i][start:end])
        X=windowdf.to_numpy()
        knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
        knn.fit(X)
        #print(len(X))
        for i in range(len(dataframes_list)):
            currentindex=dataframes_list[i][start:end]
            for q in range(len(currentindex.index)):
                dists,iindexes=knn.kneighbors([currentindex.iloc[q].to_numpy()], k+1, return_distance=True)
                if R<dists[-1][-1]:
                    #print("ID: "+str(i)+" "+str(currentindex.iloc[q].name))
                    outliers[i].append(currentindex.iloc[q].name)
    return outliers

# Distance-based outlier detection, using Turbofan dataset
def detection(window,shiftdays,k,R,dataframes_list):
    numberOfwindows=int(450/shiftdays)
    #start=
    outliers=[ [] for i in range(len(dataframes_list))]
    
    for w in range(numberOfwindows):
        start = datetime.strptime("2000-01-01 00:00:00", "%Y-%m-%d %H:%M:%S") + dtime.timedelta(days=w*shiftdays)
        end = start + dtime.timedelta(days=window)
        windowdf=dataframes_list[0][start:end]
        for i in range(1,len(dataframes_list)):
            windowdf=windowdf.append(dataframes_list[i][start:end])
        X=windowdf.to_numpy()
        knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
        knn.fit(X)
        #print(len(X))
        for i in range(len(dataframes_list)):
            currentindex=dataframes_list[i][start:end]
            for q in range(len(currentindex.index)):
                dists,iindexes=knn.kneighbors([currentindex.iloc[q].to_numpy()], k+1, return_distance=True)
                if R<dists[-1][-1]:
                    #print("ID: "+str(i)+" "+str(currentindex.iloc[q].name))
                    outliers[i].append(currentindex.iloc[q].name)
    return outliers

def plotResults(outliers,dataframes_list,indexesforgrand):
    ccc=0
    plt.figure(1)
    for i in indexesforgrand:
        # plot
        ccc+=1
        tempdf=dataframes_list[i]
        outs=outliers[i]
        outs.sort()
        x = np.array(outs)
        outsfinal=np.unique(x)
        dubleouts=[]
        for ooo in x:
            if outs.count(ooo)>1:
                dubleouts.append(ooo)
        plt.plot(tempdf.index, [ccc for q in range(len(tempdf.index))],"-b")
        plt.plot(outsfinal, [ccc for q in range(len(outsfinal))],"r.")
        plt.plot(dubleouts, [ccc for q in range(len(dubleouts))],"y.")
        #print(tempdf.index[-1])
        #score


def plotResultsBus(outliers, dataframes_list, indexesforgrand):
    ccc = 0
    fig, axis = plt.subplots(len(indexesforgrand))
    fig.set_dpi(300)
    fig.set_figheight(3)
    fig.set_figwidth(3)
    for i in indexesforgrand:
        # plot
        ccc += 1
        tempdf = dataframes_list[i]
        outs = outliers[i]

        outs.sort()
        x = np.array(outs)
        outsfinal = np.unique(x)
        dubleouts = []
        for ooo in x:
            if outs.count(ooo) > 1:
                dubleouts.append(ooo)
        #print(dubleouts)
        axis[ccc-1].plot(tempdf.index, [0.5 for q in range(len(tempdf.index))], "-b")
        axis[ccc-1].plot(outsfinal, [0.5 for q in range(len(outsfinal))], "r.")
        axis[ccc-1].plot(dubleouts, [0.5 for q in range(len(dubleouts))], "y.")
        plotLines(axis[ccc-1], i, "busFailures/")
    plt.show()
    # print(tempdf.index[-1])
    # score



# used when we plot results for Bus Dataset to plot lines when failure occur
def plotLines(ax, uid, path="busFailures/"):
    busses = ["369", "370", "371", "372", "373", "374", "375", "376", "377", "378", "379", "380", "381", "382", "383",
              "452", "453", "454", "455"]
    bus = busses[uid]
    filepath2 = path + 'Blueoutliers/outlier' + bus + '.txt'
    # red outliers
    dateredoutlier = []
    with open(path + "redouliers.txt", 'r') as f:
        Lines = f.readlines()
        for line in Lines:
            split = line.split(" ")
            if split[0] == bus:
                dateRed = datetime.strptime(split[1].strip(), '%Y-%m-%d')
                dateredoutlier.append(dateRed)
    dateredoutlierDash = []

    # DashRed
    with open(path + "rediutliersDASH.txt", 'r') as f:
        Lines = f.readlines()
        for line in Lines:
            split = line.split(" ")
            if split[0] == bus:
                dateRed = datetime.strptime(split[1].strip(), '%Y-%m-%d')
                dateredoutlierDash.append(dateRed)

    # onlyforcheck contains all failures
    onlyforcheck = []
    for date in dateredoutlier:
        onlyforcheck.append(date)
    for date in dateredoutlierDash:
        onlyforcheck.append(date)
    # Blueoutliers
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
        ax.axvline(out, color='r')
    for out in dateredoutlierDash:
        ax.axvline(out, color='r', dashes=[2, 2])
    for out in Blueoutlier:
        ax.axvline(out, color='royalblue', dashes=[2, 2])




# Calculate costs for multiple Predctive horizon and FN cost (fleet Trubofan Dataset).
def calculateCost(outliers,dataframes_list,indexesforgrand):
    fpcost=1
    fncost=10
    tpcost=1
    
    #plt.figure(2)
    phcost=[]
    for PH in [15,23,30]:
        Cost=[]
        for fncost in  [5,10,30,50,100]:
            cost=0
            for i in indexesforgrand:
                tp=0
                fp=0
                fn=0
                # plot
                tempdf=dataframes_list[i]
                outs=outliers[i]
                outs.sort()
                x = np.array(outs)
                outsfinal=np.unique(x)
                dubleouts=[]
                
                ##### SOS #####
                # PAIRNW TA DIPLA
                for ooo in x:
                    if outs.count(ooo)>1:
                        dubleouts.append(ooo)
                outsfinal=dubleouts
                finalDate=tempdf.index[-2]
                #print(tempdf.index[-1])
                #score
                for oo in outsfinal:
                    if (finalDate-oo).days <PH and (finalDate-oo).days >=0:
                        tp+=1
                    elif (finalDate-oo).days >= 0:
                        fp+=1
                
                if tp>0:
                    tp=1        
                fn=1-tp
                cost+=fn*fncost+tp*tpcost+fp*fpcost
            Cost.append(cost)
        phcost.append(Cost)
    return phcost

# In bus Dataset we have multiple types of failures (which has different cost),
# this function find for each class of failure the false positives and True positives
def redAndBLueOutliers(reported, path, uid, PH, redSolidFnCost, redDashFnCost, BlueDashFnCost, TpCost, FpCost):
    busses = ["369", "370", "371", "372", "373", "374", "375", "376", "377", "378", "379", "380", "381", "382", "383",
              "452", "453", "454", "455"]
    bus = busses[uid]
    filepath2 = path + 'Blueoutliers/outlier' + bus + '.txt'
    # red outliers
    dateredoutlier = []
    with open(path + "redouliers.txt", 'r') as f:
        Lines = f.readlines()
        for line in Lines:
            split = line.split(" ")
            if split[0] == bus:
                dateRed = datetime.strptime(split[1].strip(), '%Y-%m-%d')
                dateredoutlier.append(dateRed)
    dateredoutlierDash = []

    # DashRed
    with open(path + "rediutliersDASH.txt", 'r') as f:
        Lines = f.readlines()
        for line in Lines:
            split = line.split(" ")
            if split[0] == bus:
                dateRed = datetime.strptime(split[1].strip(), '%Y-%m-%d')
                dateredoutlierDash.append(dateRed)

    # onlyforcheck contains all failures
    onlyforcheck = []
    for date in dateredoutlier:
        onlyforcheck.append(date)
    for date in dateredoutlierDash:
        onlyforcheck.append(date)
    # Blueoutliers
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
    fp = 0
    for rep in reported:
        nearestRed = min([abs((rep - bd).days) for bd in onlyforcheck])
        # if report is made PH befroe or after the failure in't fp (ingored or Trupositive)
        if nearestRed < PH: continue
        # take all day distances from blue outliers, if the min of them is smaller than Then it is TP
        distfromblue = [(bd - rep).days for bd in onlyforcheck if (bd - rep).days > 0]
        if len(distfromblue) > 0:
            nearestBlue = min(distfromblue)
            if nearestBlue < PH: continue
        fp += 1

    # Red solid TP/FN:
    reportedREds = []
    for redfail in dateredoutlier:
        for rep in reported:
            if (redfail - rep).days < PH and (redfail - rep).days > 0:
                reportedREds.append(redfail)
                break
    redtp = len(reportedREds)
    redfn = len(dateredoutlier) - len(reportedREds)

    # Red dash TP/FN:
    reportedREds = []
    for redfail in dateredoutlierDash:
        for rep in reported:
            if (redfail - rep).days < PH and (redfail - rep).days > 0:
                reportedREds.append(redfail)
                break
    redDashtp = len(reportedREds)
    redDashfn = len(dateredoutlierDash) - len(reportedREds)

    # blue dash TP/FN:
    reportedBlue = []
    for bluefail in Blueoutlier:
        for rep in reported:
            if (bluefail - rep).days < PH and (bluefail - rep).days > 0:
                reportedBlue.append(bluefail)
                break
    BlueDashtp = len(reportedBlue)
    BlueDashfn = len(Blueoutlier) - len(reportedBlue)

    # redSolidFnCost,redDashFnCost,BlueDashFnCost,TpCost,FpCost
    cost = redSolidFnCost * redfn + redDashFnCost * redDashfn + BlueDashfn * BlueDashFnCost + fp * FpCost + (
                redtp + redDashtp + BlueDashtp)

    return cost




# Calculate costs for multiple Predctive horizon and FN cost (fleet bus Dataset).
def calculateCostBus(outliers,dataframes_list,indexesforgrand):
    fpcost=1
    fncost=10
    tpcost=1
    
    #plt.figure(2)
    phcost=[]
    for PH in [15, 23, 30]:
        Cost = []
        for fncost in [4,5,10,20]:
            BlueDashFnCost = fncost
            redDashFnCost = 5 * fncost
            redSolidFnCost = 10 * fncost
            TpCost = 1
            FpCost = 1
            cost = 0
            for i in indexesforgrand:
                tp=0
                fp=0
                fn=0
                # plot
                tempdf=dataframes_list[i]
                outs=outliers[i]
                outs.sort()
                x = np.array(outs)
                outsfinal=np.unique(x)
                dubleouts=[]
                
                ##### SOS #####
                # PAIRNW TA DIPLA
                for ooo in x:
                    if outs.count(ooo)>1:
                        dubleouts.append(ooo)
                outsfinal=dubleouts
                
                costofUid=redAndBLueOutliers(outsfinal,"busFailures/",i,PH,redSolidFnCost,redDashFnCost,BlueDashFnCost,TpCost,FpCost)
                cost += costofUid
                
            Cost.append(cost)
        phcost.append(Cost)
    return phcost



# Run Fleet Turbofan dataset
def runsenarioCost(filename,shiftdays,window,k,R,normalized):
    f002 =[155, 247, 102, 89, 62, 9, 209, 221, 190, 87, 116, 18, 110, 98, 150, 10, 35, 41, 65, 34, 21, 196, 144, 245, 143]
    f001 =[32, 15, 50, 68, 24, 16, 59, 13, 47, 49, 74, 58, 44, 41]
    f003 =[98, 88, 6, 61, 8, 90, 25, 87, 74, 29, 62, 17, 22, 63, 7, 65, 77]
    f004 =[111, 48, 107, 34, 1, 47, 204, 157, 46, 196, 67, 162, 25, 63, 172, 223, 176, 17, 187, 100, 82, 2, 136, 20, 139, 221, 195, 105, 179, 96, 220]
    if filename=="f0001":
        indexesforgrand=f001
    elif filename=="f0002":
        indexesforgrand=f002
    elif filename=="f0003":
        indexesforgrand=f003
    elif filename=="f0004":
        indexesforgrand=f004

    dataframes_list=loaddflist(filename)

    if normalized:
        dataframes_list=normalize(dataframes_list, with_mean=True, with_std=True)

    outliers=detection(window,shiftdays,k,R,dataframes_list)

    #plotResults(outliers,dataframes_list,indexesforgrand)

    Cost=calculateCost(outliers,dataframes_list,indexesforgrand)
    #print(Cost)
    towrite=[filename,shiftdays,window,k,R,normalized,Cost]
    with open(f'{filename}_KRRESULTS_COST.txt', 'a') as f:
        for item in towrite:
            f.write("%s | " % item)
        f.write("\n")

# Run Bus Dataset
def runsenarioCostBus(filename,shiftdays,window,k,R,normalized):
    indexesforgrand=[0,1,3,4,9,11,12,13,14]

    dataframes_list=loaddflistBus(filename)

    if normalized:
        dataframes_list=normalize(dataframes_list, with_mean=True, with_std=True)

    outliers=detectionBus(window,shiftdays,k,R,dataframes_list)

    #plotResultsBus(outliers,dataframes_list,indexesforgrand)

    Cost=calculateCostBus(outliers,dataframes_list,indexesforgrand)
    #print(Cost)
    towrite=[filename,shiftdays,window,k,R,normalized,Cost]
    with open(f'{filename}_KRRESULTS_COST.txt', 'a') as f:
        for item in towrite:
            f.write("%s | " % item)
        f.write("\n")



######## PARAMETERS #############
filename="vehicles"
shiftdays=20 # shift parameter for time window
window=40   # length of the time window
k=15        # parameter k for neighboars
R=1         # Radius
normalized=False

runsenarioCostBus(filename,shiftdays,window,k,R,normalized)
#runsenarioCost(filename,shiftdays,window,k,R,normalized)





















