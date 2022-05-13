import csv
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn import decomposition
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
import os
from os import listdir
from os.path import isfile, join, dirname
from datetime import datetime
import datetime as dtime
from matplotlib.pyplot import figure
import  re
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
# from beepy import beep

def convert_to_windows(data, model):
    windows = [];
    w_size = model.n_window
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i - w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
        windows.append(w)
    return torch.stack(windows)


def load_dataset(train, test, label):
    loader = [train, test, label]
    # loader[0] is ndarray
    # loader = [i[:, debug:debug+1] for i in loader]
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    labels = DataLoader(loader[2], batch_size=loader[2].shape[0])
    return train_loader, test_loader, labels


def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    folder = f'checkpoints/{"TranAD"}_{"ignore"}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)


def load_model(modelname, dims):
    import src.models
    model_class = getattr(src.models, modelname)
    model = model_class(dims).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'checkpoints/{"TranAD"}_{"ignore"}/model.ckpt'

    if True:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1;
        accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
    l = nn.MSELoss(reduction='mean' if training else 'none')
    feats = dataO.shape[1]
    if 'TranAD' in model.name:
        l = nn.MSELoss(reduction='none')
        data_x = torch.DoubleTensor(data);
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1;
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in dataloader:
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
                if isinstance(z, tuple): z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            for d, _ in dataloader:
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, bs, feats)
                z = model(window, elem)
                if isinstance(z, tuple): z = z[1]
            loss = l(z, elem)[0]
            return loss.detach().numpy(), z.detach().numpy()[0]
    else:
        y_pred = model(data)
        loss = l(y_pred, data)
        if training:
            tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            return loss.detach().numpy(), y_pred.detach().numpy()


def loaddflist(filename="f0001"):
    data_path = "./datafleet/" + filename + "/"
    files_csv = sorted([join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))])
    file_count = len(files_csv)
    # create empty list
    dataframes_list = []

    # append datasets to the list
    names = []
    for i in range(file_count):
        temp_df = pd.read_csv(files_csv[i])
        names.append(files_csv[i])
        temp_df['Artficial_timestamp'] = pd.to_datetime(temp_df['Artficial_timestamp'])
        temp_df = temp_df.set_index('Artficial_timestamp')
        if filename == "f0002" or filename == "f0004":
            temp_df = temp_df.drop(['s_1', 's_5', 's_10', 's_16', 's_18', 's_19'], axis=1)
        dataframes_list.append(temp_df)
    # print(len(dataframes_list))
    return dataframes_list
    # dataframes_list[0].head()


def loaddbusses(filename="vehicles"):
    data_path = "./datafleet/" + filename + "/"
    files_csv = sorted([join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))])
    file_count = len(files_csv)
    # create empty list
    dataframes_list = []

    # append datasets to the list
    names = []
    for i in range(file_count):
        temp_df = pd.read_csv(files_csv[i], index_col=0)

        names.append(files_csv[i])
        temp_df.index = pd.to_datetime(temp_df.index)
        temp_df = temp_df.dropna()
        dataframes_list.append(temp_df)
    print(len(dataframes_list))
    return dataframes_list

def factorcalculation(distances,factor):
    distancesTest = distances

    meanofTrain = statistics.mean(distancesTest)
    stdofTrain = statistics.stdev(distancesTest)
    thmean = meanofTrain + factor * stdofTrain
    tempdist2 = []
    for d in distancesTest:
        if d <= thmean:
            tempdist2.append(d)
    meanofTrain = statistics.mean(tempdist2)
    stdofTrain = statistics.stdev(tempdist2)
    thmean = meanofTrain + factor * stdofTrain
    return thmean

def addoutliers(listoToadd,distancesTarget,datess,thmean):
    ccc = 0
    for d in distancesTarget:
        if d > thmean:
            listoToadd.append(datess.index[ccc])
        ccc += 1
    return listoToadd

#run proposed TranAd anomaly detection in fleet data using sliding winodw
# We test multiple factors for threshold at one run.
def detectionBus(window, shiftdays, dataframes_list, indexes, factor,num_epochs):
    outliersfactor2 = [[] for i in range(len(dataframes_list))]
    outliersfactor3 = [[] for i in range(len(dataframes_list))]
    outliersfactor4 = [[] for i in range(len(dataframes_list))]
    outliersfactor5 = [[] for i in range(len(dataframes_list))]
    outliersfactor6 = [[] for i in range(len(dataframes_list))]
    outliersfactor7 = [[] for i in range(len(dataframes_list))]



    minlist = [dfff.index.min() for dfff in dataframes_list]
    print(minlist)
    maxlist = [dfff.index.max() for dfff in dataframes_list]
    startingDate = min(minlist)
    daysss = (max(maxlist) - startingDate).days + 1
    numberOfwindows = int(daysss / shiftdays)


    for w in range(numberOfwindows):
        print("ITERATION:", w)
        start = startingDate + dtime.timedelta(days=w * shiftdays)
        middle = start + dtime.timedelta(days=shiftdays)
        end = start + dtime.timedelta(days=window)
        # dataframes_list[0][start:end]
        for uid in indexes:
            if uid == 0:
                windowref = dataframes_list[1][start:middle]
                windowref2 = dataframes_list[1][middle:end]
                startIndex = 2
            else:
                windowref = dataframes_list[0][start:middle]
                windowref2 = dataframes_list[0][middle:end]

                startIndex = 1
            for j in range(startIndex, len(dataframes_list)):
                if j != uid:
                    windowref = pd.concat([windowref, dataframes_list[j][start:middle]])
                    windowref2 = pd.concat([windowref2, dataframes_list[j][middle:end]])

            TrainX = windowref.to_numpy()

            TestX = windowref2.to_numpy()
            Target = dataframes_list[uid][start:end].to_numpy()
            # this was used to plot data after pca
            if False and uid==0 and len(Target)>0:
                pca = decomposition.PCA(n_components=10)

                pca.fit(TrainX)
                ToPlot =pca.transform(TrainX)
                ToplotTarget=pca.transform(Target)
                print(ToPlot.shape)
                print(ToplotTarget.shape)
                sumpca=0
                for ratio in pca.explained_variance_ratio_:
                    sumpca+=ratio
                print(sumpca)
                figure(figsize=(3, 3), dpi=160)
                plt.plot(ToPlot[:, 0], ToPlot[:, 1], "ro")
                plt.plot(ToplotTarget[:,0],ToplotTarget[:,1], "bo")
                plt.show()
            #plot hierarhical clustering dendrogram
            if False:
                hierarhical(TrainX)
            if len(Target) == 0:
                continue

            scaler = MinMaxScaler()
            scaler.fit(TrainX)
            TrainX = scaler.transform(TrainX)
            TestX = scaler.transform(TestX)
            Target = scaler.transform(Target)

            pcaBool = True
            if pcaBool == True:
                pca = decomposition.PCA(n_components=10)

                pca.fit(TrainX)
                TrainX = pca.transform(TrainX)
                TestX = pca.transform(TestX)
                Target = pca.transform(Target)


            train_loader, test_loader, labels = load_dataset(TrainX, TestX, Target)
            mymodel = "TranAD"
            model, optimizer, scheduler, epoch, accuracy_list = load_model(mymodel, len(TrainX[0]))
            ## Prepare data
            trainD, testD, targetD = next(iter(train_loader)), next(iter(test_loader)), next(iter(labels))
            trainO, testO, target0 = trainD, testD, targetD

            trainD, testD, targetD = convert_to_windows(trainD, model), convert_to_windows(testD,
                                                                                           model), convert_to_windows(
                targetD, model)
            ###### TRAINING
            # print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')

            e = epoch + 1;
            for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
                lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
                accuracy_list.append((lossT, lr))
            # print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
            save_model(model, optimizer, scheduler, e, accuracy_list)
            # plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')
            torch.zero_grad = True
            model.eval()
            # print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
            loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
            lossTarget, y_predTarget = backprop(0, model, targetD, target0, optimizer, scheduler, training=False)
            distances = []
            for pred, y_test in zip(y_pred, testO):
                sum = 0
                for i in range(len(pred)):
                    dist = abs(pred[i] - y_test[i])
                    sum += abs(dist)
                distances.append(float(sum / len(pred)))

            c = -1
            distancesTarget = []
            for pred, ori in zip(y_predTarget, target0):
                c += 1
                sum = 0
                for i in range(len(pred)):
                    dist = abs(pred[i] - ori[i])
                    sum += abs(dist)
                distancesTarget.append(abs(float(sum) / len(ori)))

            distancesTest = distances

            thmean = factorcalculation(distancesTest, 2)
            outliersfactor2[uid]=addoutliers(outliersfactor2[uid], distancesTarget, dataframes_list[uid][start:end],
                                             thmean)

            thmean = factorcalculation(distancesTest, 3)
            outliersfactor3[uid] = addoutliers(outliersfactor3[uid], distancesTarget, dataframes_list[uid][start:end],
                                               thmean)
            thmean = factorcalculation(distancesTest, 4)
            outliersfactor4[uid] = addoutliers(outliersfactor4[uid], distancesTarget, dataframes_list[uid][start:end],
                                               thmean)
            thmean = factorcalculation(distancesTest, 5)
            outliersfactor5[uid] = addoutliers(outliersfactor5[uid], distancesTarget, dataframes_list[uid][start:end],
                                               thmean)

            thmean = factorcalculation(distancesTest, 6)
            outliersfactor6[uid] = addoutliers(outliersfactor6[uid], distancesTarget, dataframes_list[uid][start:end],
                                               thmean)

            thmean = factorcalculation(distancesTest, 7)
            outliersfactor7[uid] = addoutliers(outliersfactor7[uid], distancesTarget, dataframes_list[uid][start:end],
                                               thmean)
            # figure(figsize=(3, 3), dpi=160)
            # plt.plot([c for c in range(len(distancesTest))], distancesTest, "y-")
            # plt.plot([c for c in range(testlen,testlen+len(distances))], distances, "b-")
            # plt.plot([c for c in range(0,testlen+len(distances))], [thmean for c in range(0,testlen+len(distances))], "r-")
            # plt.ylim((0, 1))
            # plt.show()
    return outliersfactor2,outliersfactor3,outliersfactor4,outliersfactor5,outliersfactor6,outliersfactor7


# caculate the distance from a data-point (sample)
# to all clusters
# return the cluster with the nearest with the nearest point to sample.
def distforTest(sample,clusters):

    minimum=9999999999999999999999
    minimumpos=-1
    cccc=0
    for clust in clusters:
        dists1=cdist(np.array([sample]), np.array(clust), 'euclidean')
        if minimumpos==-1 or minimum>min(dists1[0,:]):
            minimumpos = cccc
            minimum=min(dists1[0,:])
        cccc+=1
    return minimumpos
    #dists2=cdist(np.array([sample]), np.array(cluster2), 'euclidean')
    #print(dists2.shape)


# Trains TranAd model using TrainX data
# Calculate threshold using TestX data
# Calculate reconstruction of Target Data
# returns threshold, reconstructed Target Data and original Target Data
def traintestpredict(TrainX,TestX,Target,factor):
    scaler = MinMaxScaler()
    scaler.fit(TrainX)
    TrainX = scaler.transform(TrainX)
    TestX = scaler.transform(TestX)
    Target = scaler.transform(Target)

    train_loader, test_loader, labels = load_dataset(TrainX, TestX, Target)
    mymodel = "TranAD"
    model, optimizer, scheduler, epoch, accuracy_list = load_model(mymodel, len(TrainX[0]))
    ## Prepare data
    trainD, testD, targetD = next(iter(train_loader)), next(iter(test_loader)), next(iter(labels))
    trainO, testO, target0 = trainD, testD, targetD

    trainD, testD, targetD = convert_to_windows(trainD, model), convert_to_windows(testD,
                                                                                   model), convert_to_windows(
        targetD, model)
    ###### TRAINING
    # print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
    num_epochs = 5;
    e = epoch + 1;
    for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
        lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
        accuracy_list.append((lossT, lr))
    # print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
    save_model(model, optimizer, scheduler, e, accuracy_list)
    # plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')
    torch.zero_grad = True
    model.eval()
    # print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
    loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

    distances = []
    for pred, y_test in zip(y_pred, testO):
        sum = 0
        for i in range(len(pred)):
            dist = abs(pred[i] - y_test[i])
            sum += abs(dist)
        distances.append(float(sum / len(pred)))
    testlen = len(distances)
    distancesTest = distances
    startind = len(TrainX)
    trainsize = len(TrainX)

    meanofTrain = statistics.mean(distances)
    stdofTrain = statistics.stdev(distances)
    thmean = meanofTrain + factor * stdofTrain
    tempdist2 = []
    for d in distancesTest:
        if d <= meanofTrain + factor * stdofTrain:
            tempdist2.append(d)
    meanofTrain = statistics.mean(tempdist2)
    stdofTrain = statistics.stdev(tempdist2)
    thmean = meanofTrain + factor * stdofTrain
    loss, y_pred = backprop(0, model, targetD, target0, optimizer, scheduler, training=False)
    return  thmean,y_pred,target0

# TranAD anomaly detection using sliding window.
# clusterToUse parameter define how many clusters will bew used for clustering before training phase of TranAd
def detectionulticluster(window, shiftdays, dataframes_list, indexes, factor,clusterToUse):
    numberOfwindows = int(450 / shiftdays)
    # start=
    outliers = [[] for i in range(len(dataframes_list))]

    for w in range(numberOfwindows):
        print("ITERATION:", w)
        start = datetime.strptime("2000-01-01 00:00:00", "%Y-%m-%d %H:%M:%S") + dtime.timedelta(days=w * shiftdays)
        middle = start + dtime.timedelta(days=shiftdays)
        end = start + dtime.timedelta(days=window)
        # dataframes_list[0][start:end]
        for uid in indexes:
            if uid == 0:
                windowref = dataframes_list[1][start:middle]
                windowref2 = dataframes_list[1][middle:end]
                startIndex = 2
            else:
                windowref = dataframes_list[0][start:middle]
                windowref2 = dataframes_list[0][middle:end]

                startIndex = 1
            for j in range(startIndex, len(dataframes_list)):
                if j != uid:
                    windowref = pd.concat([windowref, dataframes_list[j][start:middle]])
                    windowref2 = pd.concat([windowref2, dataframes_list[j][middle:end]])

            TrainX = windowref.to_numpy()

            TestX = windowref2.to_numpy()
            Target = dataframes_list[uid][start:end].to_numpy()

            clustering = AgglomerativeClustering(n_clusters=clusterToUse).fit(TrainX)
            clusters=[ [] for _ in np.unique(clustering.labels_)]
            Testclusters=[ [] for _ in np.unique(clustering.labels_)]
            Targetclusters=[ [] for _ in np.unique(clustering.labels_)]
            TargetclustersPos=[ [] for _ in np.unique(clustering.labels_)]
            for sample,cl in zip(TrainX,clustering.labels_):
                clusters[cl].append(sample)
            for sample in TestX:
                Testclusters[distforTest(sample,clusters)].append(sample)
            counterpostarget=0
            for sample in Target:
                clss=distforTest(sample,clusters)
                Targetclusters[clss].append(sample)
                TargetclustersPos[clss].append(counterpostarget)
                counterpostarget+=1

            if False:
                hierarhical(TrainX)

            for tr,tes,targ,targetclustersPos in zip(clusters,Testclusters,Targetclusters,TargetclustersPos):
                if len(targ) == 0:
                    continue
                thmean,y_pred,target0 = traintestpredict(np.array(tr),np.array(tes),np.array(targ),factor)
                mmmeans = []
                outlier = []
                indecout = []
                c = -1
                distances = []
                for pred, ori in zip(y_pred, target0):
                    c += 1
                    sum = 0
                    for i in range(len(pred)):
                        dist = abs(pred[i] - ori[i])
                        sum += abs(dist)
                    distances.append(abs(float(sum) / len(ori)))
                    # print(distances)
                    # mnow = statistics.mean(distances[-min(trainsize,len(distances)):])
                    # mmmeans.append(mnow)
                    # if mnow >= thmean:
                    #    outliers[uid].append(dataframes_list[uid][start:end].index[c])
                    # print("OUTLIER:")
                    # print(dataframes_list[uid][start:end].index[c])
                # for f0001 and f0002
                ccc = 0
                for d in distances:
                    if d > thmean:
                        outliers[uid].append(dataframes_list[uid][start:end].index[targetclustersPos[ccc]])
                    ccc += 1

            # figure(figsize=(3, 3), dpi=160)
            # plt.plot([c for c in range(len(distancesTest))], distancesTest, "y-")
            # plt.plot([c for c in range(testlen,testlen+len(distances))], distances, "b-")
            # plt.plot([c for c in range(0,testlen+len(distances))], [thmean for c in range(0,testlen+len(distances))], "r-")
            # plt.ylim((0, 1))
            # plt.show()
    return outliers


def plotResults(outliers, dataframes_list, indexesforgrand):
    ccc = 0
    figure(figsize=(3, 3), dpi=160)
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
        print(dubleouts)
        plt.plot(tempdf.index, [ccc for q in range(len(tempdf.index))], "-b")
        plt.plot(outsfinal, [ccc for q in range(len(outsfinal))], "r.")
        plt.plot(dubleouts, [ccc for q in range(len(dubleouts))], "y.")
    plt.show()
    # print(tempdf.index[-1])
    # score

def plotResultsBus(outliers, dataframes_list, indexesforgrand):
    ccc = 0
    fig, axis = plt.subplots(len(indexesforgrand))
    fig.set_dpi(200)
    fig.set_figheight(3)
    fig.set_figwidth(3)
    plt.rcParams.update({'font.size': 1})
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
        print(dubleouts)
        #axis[ccc-1].plot(tempdf.index, [0.5 for q in range(len(tempdf.index))], "-b")
        axis[ccc-1].plot(outsfinal, [0.5 for q in range(len(outsfinal))], "r.",markersize=2)
        axis[ccc-1].plot(dubleouts, [0.5 for q in range(len(dubleouts))], "r.",markersize=2)
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
        ax.axvline(out, color='r', marker="P")
        ax.plot([out], [0.5], color='y', marker="P")
    for out in dateredoutlierDash:
        ax.axvline(out, color='r')
    for out in Blueoutlier:
        ax.axvline(out, color='royalblue', dashes=[2, 2])


# Calculate costs for multiple Predctive horizon and FN cost (fleet Trubofan Dataset).
def calculateCost(outliers, dataframes_list, indexesforgrand):
    fpcost = 1
    fncost = 10
    tpcost = 1

    # plt.figure(2)
    phcost = []
    for PH in [15, 23, 30]:
        Cost = []
        for fncost in [5, 10, 30, 50, 100]:
            cost = 0
            for i in indexesforgrand:
                tp = 0
                fp = 0
                fn = 0
                # plot
                tempdf = dataframes_list[i]
                outs = outliers[i]
                outs.sort()
                x = np.array(outs)
                outsfinal = np.unique(x)
                dubleouts = []

                ##### SOS #####
                # PAIRNW TA DIPLA
                for ooo in x:
                    if outs.count(ooo) > 1:
                        dubleouts.append(ooo)
                outsfinal = dubleouts
                finalDate = tempdf.index[-2]
                # print(tempdf.index[-1])
                # score
                for oo in outsfinal:
                    if (finalDate - oo).days < PH and (finalDate - oo).days >= 0:
                        tp += 1
                    elif (finalDate - oo).days >= 0:
                        fp += 1

                if tp > 0:
                    tp = 1
                fn = 1 - tp
                cost += fn * fncost + tp * tpcost + fp * fpcost
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



#plot dendogram of hierarhical clustering results
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

#run hierarchical clustering and show dendogram results
def hierarhical(X):


    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    model = model.fit(X)
    plt.rcParams.update({'font.size': 4})
    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.tick_params(axis='x', which='major', labelsize=2)
    plt.tick_params(axis='x', which='minor', labelsize=2)

    plt.show()

# Calculate costs for multiple Predctive horizon and FN cost (fleet bus Dataset).
def calculateCostBus(outliers, dataframes_list, indexesforgrand):
    fpcost = 1
    fncost = 10
    tpcost = 1

    # plt.figure(2)
    phcost = []
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
                tempdf = dataframes_list[i]
                outs = outliers[i]
                outs.sort()
                x = np.array(outs)
                outsfinal = np.unique(x)
                # dubleouts = []
                #
                # ##### SOS #####
                # # PAIRNW TA DIPLA
                # for ooo in x:
                #     if outs.count(ooo) > 1:
                #         dubleouts.append(ooo)
                # outsfinal = dubleouts
                costofUid=redAndBLueOutliers(outsfinal,"busFailures/",i,PH,redSolidFnCost,redDashFnCost,BlueDashFnCost,TpCost,FpCost)
                cost += costofUid
            Cost.append(cost)
        phcost.append(Cost)
    return phcost

# Used to run TranAd in turbofan fleet Dataset
def test_for_f000(filename="f0001",factor=2):
    f001 = [32, 15, 50, 68, 24, 16, 59, 13, 47, 49, 74, 58, 44, 41]
    f002 = [155, 247, 102, 89, 62, 9, 209, 221, 190, 87, 116, 18, 110, 98, 150, 10, 35, 41, 65, 34, 21, 196, 144, 245,
            143]
    f003 = [98, 88, 6, 61, 8, 90, 25, 87, 74, 29, 62, 17, 22, 63, 7, 65, 77]
    f004 = [111, 48, 107, 34, 1, 47, 204, 157, 46, 196, 67, 162, 25, 63, 172, 223, 176, 17, 187, 100, 82, 2, 136, 20,
            139, 221, 195, 105, 179, 96, 220]

    # parameters
    # filename = "f0003"
    clusterToUse=1
    if filename == "f0001":
        indexesforgrand = f001
        clusterToUse=1
    elif filename == "f0002":
        indexesforgrand = f002
        clusterToUse=6
    elif filename == "f0003":
        indexesforgrand = f003
        clusterToUse=1
    elif filename == "f0004":
        indexesforgrand = f004
        clusterToUse=6

    shiftdays = 20
    window = 40
    #[[249, 269, 349, 429, 629], [145, 165, 245, 325, 525], [93, 113, 193, 273, 473]]
    #[[169, 184, 244, 304, 454], [95, 110, 170, 230, 380], [63, 78, 138, 198, 348]]f0003_TranAD_COST.txt
    dataframes_list = loaddflist(filename)
    outliers = detectionulticluster(window, shiftdays, dataframes_list, indexesforgrand, factor,clusterToUse)
    #plotResults(outliers, dataframes_list, indexesforgrand)

    Cost = calculateCost(outliers, dataframes_list, indexesforgrand)
    print(Cost)
    towrite = [filename, shiftdays, window, factor, Cost]
    with open(f'{filename}_TranAD_COST.txt', 'a') as f:
        for item in towrite:
            f.write("%s | " % item)
        f.write("\n")

# Used to run TranAd in Bus fleet Dataset
def test_forBussed():
    filename = "vehicles"
    indexesforgrand = [0, 1, 3, 4, 9, 11, 12, 13, 14]  # ids of busses where used in our case


    shiftdays = 10
    window = 20
    num_epochs=8
    dataframes_list = loaddbusses(filename)
    outliersfactor2, outliersfactor3, outliersfactor4, outliersfactor5, outliersfactor6, outliersfactor7 = detectionBus(window, shiftdays, dataframes_list, indexesforgrand, 2,num_epochs)

    plotResultsBus(outliersfactor5, dataframes_list, indexesforgrand)
    Cost = calculateCostBus(outliersfactor5, dataframes_list, indexesforgrand)
    print(Cost)
    for outtt,fact in zip([outliersfactor2,outliersfactor3,outliersfactor4,outliersfactor5,outliersfactor6,outliersfactor7],[2,3,4,5,6,7]):
        Cost=calculateCostBus(outtt, dataframes_list, indexesforgrand)
        towrite = [filename, shiftdays, window, fact, Cost]
        with open(f'{filename}_TranAD_COST.txt', 'a') as f:
            for item in towrite:
                f.write("%s | " % item)
            f.write("\n")
if __name__ == '__main__':
    #factorr=3
    #test_for_f000("f0001",factorr)
    test_forBussed()
