#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 12:50:02 2022

@author: agiannous
"""
#[filename,non_k,metric,Reference_window,w_mart,normalized,F1,PR,RE]
def resultsCost(filename):
    results=[]
    with open(f'{filename}_clusterjoint_COST.txt') as file:
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
        #print(listline)

    
def maxresultsCost(filename,PH=30,COST_FN=100):
    mincost=10000000000
    minExp=-1
    results=resultsCost(filename)
    c=0
    Phrange=[15,23,30]
    COSTS=[5,10,30,50,100]
    if filename=="ClusterJoint/vehicles":
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

