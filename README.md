# A context aware unsupervised predictive maintenance solution for fleet management

## Methods:
TranAD folder : TranAD method. (pytorch library is requiered to run)

HybridTest folder: 2Stage method. In order to run 2Stage method, you will need to use our version of Grand package, which is provided under the grand folder.

ClusterJoint folder: ClusterJoint method in paper.

KR_TEST folder: distance-based outlier detection method (brute force python implementation of MCOD, you can find faster implementation in java here: https://infolab.usc.edu/Luan/Outlier/) reffered as DOD in paper.

test_grand folder: Grand method. Group Anomaly Detection algotrithm used in paper as baseline.

Each folder contains a README file fo further explaination of the parameters and methods.

## Install modified grand package (before run any experiment)

GrandPackage folder contains the modified library of Grand method.

Install the package using "pip install ." or "pip3 install ." inside the folder.


## Fleet dataset construction (Turbofan C-MPAS data)

In CreateFleetDataset folder are the raw data and four notebooks to produce a fleet dataset using them.

## Final cost Results

Use showBestCost.py in order to plot the lowest cost achived by the methods used in expiriements
