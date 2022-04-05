# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 11:08:41 2022

@author: user
"""

import test_grand.readResults as GrandResults
import HybridTest.readResults as HybridResults
import matplotlib.pyplot as plt
from KR_TEST import readResults as krResults
from TranAD import readResults as tranResults
#from HybridDyn import readResults as hybridDyn
from HybridTest import readResults as hybridDyn
folderfor_Hybrid="HybridTest"
from numpy import trapz
from ClusterJoint import readResults as clusterjoint



filename="f0004"
PH=30


Phrange=[15,23,30]
#costkr,kr,kri=krResults.maxresultsCost("KR_TEST/"+filename,PH,20)
#costgrand,grand,grandi=GrandResults.maxresultsCost("test_grand/"+filename,PH,20)
#cotsthybridDyn,hybridyn,hybridyni=hybridDyn.maxresultsCost(f"{folderfor_Hybrid}/"+filename,PH,20)
#costcluster,cluster,clusteri=clusterjoint.maxresultsCost("ClusterJoint/"+filename,PH,20)

#costtran,tran,trani=tranResults.maxresultsCost("TranAD/"+filename,PH,20)
#print(tran)


fig, axs = plt.subplots(3)
fig.suptitle('')
c=0

NoAlarms=[[720, 900, 1800, 3600], [720, 900, 1800, 3600], [720, 900, 1800, 3600]]

jjj=-1
for PH in Phrange:
    jjj+=1
    #Phrange=[15,23,30]
    COSTS=[5,10,30,50,100]
    if filename=="vehicles":
        COSTS=[4,5,10,20]
    
    Grand_cost=[]
    KR_cost=[]
    Tran_cost=[]
    Hybrid_cost=[]
    Cluster_cost=[]
    iii=-1
    for cost in COSTS:
        iii+=1
        
        costkr,kr,kri=krResults.maxresultsCost("KR_TEST/"+filename,PH,cost)
        costgrand,grand,grandi=GrandResults.maxresultsCost("test_grand/"+filename,PH,cost)
        cotsthybridDyn,hybridyn,hybridyni=hybridDyn.maxresultsCost(f"{folderfor_Hybrid}/"+filename,PH,cost)
        costcluster,cluster,clusteri=clusterjoint.maxresultsCost("ClusterJoint/"+filename,PH,cost)

        if filename == "f0001" or filename=="f0003" or filename=="f0002" or filename=="f0004" or filename=="vehicles":
            costtran,tran,trani=tranResults.maxresultsCost("TranAD/"+filename,PH,cost)
            Tran_cost.append(costtran)
        KR_cost.append(costkr)
        Grand_cost.append(costgrand)
        Hybrid_cost.append(cotsthybridDyn)
        Cluster_cost.append(costcluster)
        
        
    axs[c].plot(COSTS,KR_cost,marker='o',label="KR")
    axs[c].plot(COSTS,Hybrid_cost,marker='s',label="2Stage")
    axs[c].plot(COSTS,Cluster_cost,marker='p',label="ClusterJoint")
    axs[c].plot(COSTS,Grand_cost,marker='*',label="Grand")

    if filename == "f0001" or filename=="f0003" or filename=="f0002" or filename=="f0004" or filename=="vehicles":
        axs[c].plot(COSTS,Tran_cost,marker='>',label="TranAD")
    
    if filename=="vehicles":
    	axs[c].plot(COSTS,NoAlarms[jjj],marker='2',color="black",label="NoAlarmsCost")

    axs[c].set(xlabel="Cost of FN", ylabel="Total Cost",title=f"Total Cost vs cost of FN, for PH={PH}")
    axs[c].legend()
    
    c+=1

plt.show()

#costhybrid,hybrid,hybridi=HybridResults.maxresultsCost("HybridTest/"+filename,PH,COST)
#print(hybrid)


#print("========== MIN COST ===========")
#print("KR =",costkr)
#print("Grand =",costgrand)
#print("Hybrid =",costhybrid)

#C
# ph=15
# plt.title(f"Cost for PH={ph}")

# pos=Phrange.index(ph)
# plt.plot(,kr[kri][pos],label="KR")
# plt.plot(COSTS,dyn[dyni][pos],label="DYN_KR")
# plt.plot(COSTS,grand[grandi][pos],label="Grand")
# #plt.plot(COSTS,hybrid[hybridi][pos],label="Hybrid")
# plt.xlabel("Cost of FN")
# plt.ylabel("Total Cost")
# plt.legend()


# plt.title("Cost for FN_cost=50")

# pos=COSTS.index(50)
# plt.plot(Phrange,[kr[kri][i][pos] for i in range(len(Phrange)) ],label="KR")
# plt.plot(Phrange,[dyn[dyni][i][pos] for i in range(len(Phrange)) ],label="DYN_KR")
# plt.plot(Phrange,[grand[grandi][i][pos] for i in range(len(Phrange)) ],label="Grand")
# plt.plot(Phrange,[hybrid[hybridi][i][pos] for i in range(len(Phrange)) ],label="Hybrid")
# plt.xlabel("PH")
# plt.ylabel("Total Cost")
# plt.legend()
