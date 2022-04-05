#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 12:50:02 2022

@author: agiannous
"""

def resultsKR(filename):
    results=[]
    with open(f'{filename}_TranAD_COST.txt') as file:
        lines = file.readlines()
        for line in lines:
            #print(line)
            PhRange=[i* 5 for i in range(1,14)]
            listline=line.split(" | ")[:-1]
            
            filename=listline[0]
            
            listline[1]=float(listline[1])
            shift=listline[1]
            
            listline[2]=int(listline[2])
            window=listline[2]
            
            
            listline[3]=int(listline[3])
            k=listline[3]
            
            listline[4]=float(listline[4])
            R=listline[4]
            
            if listline[5]=='False':
                listline[5]=False
            else:
                listline[5]=True
            normalization=listline[5]
            F1=[]
            f1string=listline[6][1:-1].split(',')
            for value in f1string:
                F1.append(float(value))
            listline[6]=F1
            PR=[]
            prstring=listline[7][1:-1].split(',')
            for value in prstring:
                PR.append(float(value))
            listline[7]=PR
            
            RE=[]
            recstring=listline[8][1:-1].split(',')
            for value in recstring:
                RE.append(float(value))
            listline[8]=RE
            results.append(listline)
    return results
def resultsCost(filename):
    results=[]
    with open(f'{filename}_TranAD_COST.txt') as file:
        lines = file.readlines()
        for line in lines:
            #print(line)
            PhRange=[i* 5 for i in range(1,14)]
            listline=line.split(" | ")[:-1]
            filename=listline[0]
            
            listline[1]=float(listline[1])
            non_k=listline[1]
            
            metric=listline[2]
            f1string=listline[4]
            
            listcost=f1string.split("],")
            listcost=[l.replace("[","").replace("]","") for l in listcost ]
            finalList=[  sub.split(",") for sub in listcost  ]
            for i in range(len(finalList)):
                for j in range(len(finalList[i])):
                    finalList[i][j]=int(finalList[i][j])
            listline[4]=finalList
            
            
            results.append(listline)
    return results
def maxresultsKR(filename,Sum=True):
    maxf1=-1
    maxExp=-1
    results=resultsKR(filename)
    c=0
    for re in results:
        maxx=max(re[6])
        if maxx>maxf1:
            maxf1=maxx
            maxExp=c
        c+=1
    return results[maxExp],6


def maxresultsCost(filename,PH=30,COST_FN=100):
    mincost=10000000000
    minExp=-1
    results=resultsCost(filename)
    c=0
    Phrange=[15,23,30]
    COSTS=[5,10,30,50,100]
    if filename=="TranAD/vehicles":
        COSTS=[4,5,10,20]
    phpos=Phrange.index(PH)
    costpos=COSTS.index(COST_FN)
    for re in results:
        cost=re[4][phpos][costpos]
        if mincost>cost:
            mincost=cost
            minExp=c
        c+=1
    return mincost,results[minExp],4 
    
    
