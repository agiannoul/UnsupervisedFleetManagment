#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 12:50:02 2022

@author: agiannous
"""
#[filename,non_k,metric,Reference_window,w_mart,normalized,F1,PR,RE]
def conditions(re,metric):
    if re[2]==metric:
        return True
    else:
        return False

def resultsGrand(filename):
    results=[]
    with open(f'{filename}_GRAND_RESULTS_CORRECT.txt') as file:
        lines = file.readlines()
        for line in lines:
            #print(line)
            PhRange=[i* 5 for i in range(1,14)]
            listline=line.split(" | ")[:-1]
            
            filename=listline[0]
            
            listline[1]=int(listline[1])
            non_k=listline[1]
            
            metric=listline[2]
            
            Reference_window=listline[3]
            
            listline[4]=int(listline[4])
            w_mart=listline[4]
            
            if listline[5]=='False':
                listline[5]=False
            else:
                listline[5]=True
            normalization=listline[5]
            
            listline[6]=float(listline[6])
            th=listline[6]
            
            F1=[]
            f1string=listline[7][1:-1].split(',')
            for value in f1string:
                F1.append(float(value))
            listline[7]=F1
            PR=[]
            prstring=listline[8][1:-1].split(',')
            for value in prstring:
                PR.append(float(value))
            listline[8]=PR
            
            RE=[]
            recstring=listline[9][1:-1].split(',')
            for value in recstring:
                RE.append(float(value))
            listline[9]=RE
            results.append(listline)
    return results
        #print(listline)


def resultsCost(filename):
    results=[]
    with open(f'{filename}_GRAND_RESULTS_Cost.txt') as file:
        lines = file.readlines()
        for line in lines:
            #print(line)
            PhRange=[i* 5 for i in range(1,14)]
            listline=line.split(" | ")[:-1]
            
            filename=listline[0]
            
            listline[1]=int(listline[1])
            non_k=listline[1]
            
            metric=listline[2]
            
            Reference_window=listline[3]
            
            listline[4]=int(listline[4])
            w_mart=listline[4]
            
            if listline[5]=='False':
                listline[5]=False
            else:
                listline[5]=True
            normalization=listline[5]
            
            listline[6]=float(listline[6])
            th=listline[6]
            
            
            
            F1=[]
            f1string=listline[7]
            listcost=f1string.split("],")
            listcost=[l.replace("[","").replace("]","") for l in listcost ]
            finalList=[  sub.split(",") for sub in listcost  ]
            for i in range(len(finalList)):
                for j in range(len(finalList[i])):
                    finalList[i][j]=int(finalList[i][j])
            listline[7]=finalList
            
            
            results.append(listline)
    return results


def maxresultsGrand(filename,Sum=True):
    maxf1=-1
    maxExp=-1
    results=resultsGrand(filename)
    c=0
    for re in results:
        if Sum==True:
            maxx=sum(re[7])
        else:
            maxx=max(re[7])
        if maxx>maxf1:
            maxf1=maxx
            maxExp=c
        c+=1
    return results[maxExp],7

def maxresultsGrandWithCondition(filename,Sum=True,metric="median"):
    maxf1=-1
    maxExp=-1
    results=resultsGrand(filename)
    c=0
    for re in results:
        if not conditions(re,metric):
            c+=1
            continue
        if Sum==True:
            maxx=sum(re[7])
        else:
            maxx=max(re[7])
        if maxx>maxf1:
            maxf1=maxx
            maxExp=c
        c+=1
    return results[maxExp],7

def maxresultsCost(filename,PH=30,COST_FN=100):
    mincost=10000000000
    minExp=-1
    results=resultsCost(filename)
    c=0
    Phrange=[15,23,30]
    COSTS=[5,10,30,50,100]
    if filename=="test_grand/vehicles":
        COSTS=[4,5,10,20]
    phpos=Phrange.index(PH)
    costpos=COSTS.index(COST_FN)
    for re in results:
        cost=re[7][phpos][costpos]
        if mincost>cost:
            mincost=cost
            minExp=c
        c+=1
    return mincost,results[minExp],7  

