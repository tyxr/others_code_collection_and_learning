import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
global temp,Feature


# get useful format of the data
def recursion(i,temp,Feature):
    if Feature[i] not in temp:
        temp.append(Feature[i])
        return
    else:
        #print(Feature[i])
        Feature[i] = Feature[i]+'5'
        recursion(i,temp,Feature)
def Data_Pre_Process(filename):
    data = pd.read_csv(filename, sep='\t', header=0)
    columns = data.columns
    p = list(filter(lambda x: x.find('FALSE') != -1, columns))   # p = ['POS','POS1',...]
    n = list(filter(lambda x: x.find('TRUE') != -1, columns))   # n = ['NEG','NEG1',...]
    Feature = data['gene']                                      # Feature = ['1990_at','10002_at',...]
    temp = []

    for i in range(len(Feature)):
        
        recursion(i,temp,Feature)




    data = data.set_index(['gene'])

    # POS & NEG
    data = data.ix[:, p].join(data.ix[:, n])                    # same lable has been merged together
    p_data = data.ix[:, p].T                                      # label is p
    n_data = data.ix[:, n].T                                      # label is n

    return p,n,Feature,p_data,n_data,data

# for each row calculate the pvalue
def get_Pvalue(p_data,n_data):
    Pvalues = []
    for row in range(len(p_data)):
        pos = p_data.ix[row]
        neg = n_data.ix[row]
        tvalue, pvalue = stats.ttest_ind(pos,neg)
        Pvalues.append(pvalue)
    return Pvalues

# after a t_test, get a sorted_feature
def get_sorted_feature(Pvalues,Feature):
    PvalueDict = dict(zip(Pvalues, Feature))
    Pvalues.sort()
    sortFeature = map(PvalueDict.get, Pvalues)
    return sortFeature

# 4 group features for top1,top10.top100,bottom100
def feature_set(sortFeature):
    sortFeature = list(sortFeature)
    g1 = [sortFeature[0]]
    g2 = []
    g3 = []
    g4 = []

    for i in range(10):
       g2.append(sortFeature[i])
    for i in range(100):
       g3.append(sortFeature[i])
    for i in range(100):
       g4.append(sortFeature[len(sortFeature)-1-i])

    G = [g1, g2, g3, g4]
    return G

# evaluate each classifier
def SN_SP_ACC_AVC_MCC(Test_y,Predict):
    # Test_y is actual lable,Predict is predict lable
    TP = 0.;
    TN = 0.;
    FP = 0.;
    FN = 0.;
    for i in range(len(Test_y)):
        if Test_y[i] == 'P' and Predict[i] == 'P':
            TP += 1
        if Test_y[i] == 'P' and Predict[i] == 'N':
            FN += 1

        if Test_y[i] == 'N' and Predict[i] == 'P':
            FP += 1
        if Test_y[i] == 'N' and Predict[i] == 'N':
            TN += 1
        i = i + 1

    SN = float(TP/ (TP + FN))  # Sensitivity = TP/P  and P = TP + FN
    SP = float(TN/ (FP + TN))  # Specificity = TN/N  and N = TN + FP
    ACC = float((TP + TN)/ (TP + TN + FP + FN))# Accuracy
    AVC = float((SN + SP) / 2) # Average
    temp = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
    if temp == 0:
        return SN,SP,ACC,AVC,0
    MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return SN,SP,ACC,AVC,MCC

# Take K-Fold and get the mean value of each Model
def Mean_Score(classifier,p_data, n_data,p,n,seed_number):
    SN = []
    SP = []
    ACC = []
    AVC = []
    MCC = []

    p_kf = KFold(p_data.shape[0],n_folds=5,shuffle=True,random_state=seed_number)   #p_kf is just a index p_kf= [([1,2,3],[3,4]),([1,2,4],[3,5])] k=2
    n_kf = KFold(n_data.shape[0],n_folds=5,shuffle=True, random_state=seed_number)

    k = 5
    for i in range(k):
        p_train_data = []
        p_test_data = []
        n_train_data = []
        n_test_data = []

        temp = []
        p_train = p_data.iloc[list(p_kf)[i][0],:]
        p_test = p_data.iloc[list(p_kf)[i][1],:]

        n_train = n_data.iloc[list(n_kf)[i][0],:]
        n_test = n_data.iloc[list(n_kf)[i][1],:]

        temp = p_train.T
        columns = temp.columns

        for i in columns:
            p_train_data.append(list(p_train.ix[i]))
        y_train = ['P' for i in range(len(p_train_data))]

        temp = n_train.T
        columns = temp.columns
        for i in columns:
            n_train_data.append(list(n_train.ix[i]))

        # merge p_train_data & n_train_data
        for i in n_train_data:
            p_train_data.append(i)
        X_train = p_train_data

        temp = p_test.T
        columns = temp.columns
        for i in columns:
            p_test_data.append(list(p_test.ix[i]))
        y_test = ['P' for i in range(len(p_test_data))]

        temp = n_test.T
        columns = temp.columns
        for i in columns:
            n_test_data.append(list(n_test.ix[i]))

        for i in n_test_data:
            p_test_data.append(i)
        X_test = p_test_data

        n = ['N' for i in range(len(n_train_data))]
        for t in n:
            y_train.append(t)

        n = ['N' for i in range(len(n_test_data))]
        for t in n:
            y_test.append(t)
        #print len(y_test),len(X_test),len(y_train),len(X_train)

        Model = classifier.fit(X_train, y_train)
        Predict = Model.predict(X_test)
        sn, sp, acc, avc, mcc = SN_SP_ACC_AVC_MCC(y_test, Predict)

        SN.append(sn)
        SP.append(sp)
        ACC.append(acc)
        AVC.append(avc)
        MCC.append(mcc)

    return np.mean(SN),np.mean(SP),np.mean(ACC),np.mean(AVC),np.mean(MCC)

# the group of features
def Perform_Classifier(G,p_data,n_data,p,n,seed_number):

    performance = []

    for i in range(len(G)):
        svc = []
        nbayes = []
        knn= []
        pfm = []
        # Naive Bayes Classifier
        classifier = GaussianNB()
        sn, sp, acc, avc, mcc = Mean_Score(classifier, p_data[G[i]], n_data[G[i]], p, n, seed_number)
        nbayes.append(sn)
        nbayes.append(sp)
        nbayes.append(acc)
        nbayes.append(avc)
        nbayes.append(mcc)

        # KNN Classifier
        classifier = KNeighborsClassifier()
        sn, sp, acc, avc, mcc = Mean_Score(classifier, p_data[G[i]], n_data[G[i]], p, n, seed_number)
        knn.append(sn)
        knn.append(sp)
        knn.append(acc)
        knn.append(avc)
        knn.append(mcc)

        # SVM Classifier
        classifier = SVC()
        sn, sp, acc, avc, mcc = Mean_Score(classifier, p_data[G[i]], n_data[G[i]], p, n, seed_number)
        svc.append(sn)
        svc.append(sp)
        svc.append(acc)
        svc.append(avc)
        svc.append(mcc)

        pfm.append(tuple(nbayes))
        pfm.append(tuple(knn))
        pfm.append(tuple(svc))

        performance.append(pfm)

    return performance

def polt_picture(performance):

    # plot 2 sub-picture
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))
    # draw a bar
    bar_xtick = ['SN', 'SP', 'ACC', 'AVC', 'MCC']
    lngds = ['NBayes', 'KNN', 'SVM']
    colors = ['r', 'green', 'blue']
    titles = ['TOP1', 'TOP10', 'TOP100', 'BOTTOM100']

    for i in range(4):
        pfm = performance[i]
        Y_bar = np.array(pfm)
        width = 0.15
        ind = np.arange(len(bar_xtick))
        plt.title(titles[i])
        rects = []
        t = 1
        for j in range(len(lngds)):      # NOTICE: put the same color in
            rects.append(axes.flat[i].bar(ind + j * width, Y_bar[j], width=width, color=colors[j])[0])

        axes.flat[i].set_xticks(ind + width)
        axes.flat[i].set_xticklabels(bar_xtick)
        axes.flat[i].legend(rects, lngds, loc='upper right')
        axes.flat[i].set_title(titles[i])
    return plt

# main function
p,n,Feature,p_data,n_data,data = Data_Pre_Process('CNS2.txt')
Pvalues = get_Pvalue(p_data.T,n_data.T)
sortFeature = get_sorted_feature(Pvalues,Feature)
G = feature_set(sortFeature)
performance =  Perform_Classifier(G,p_data,n_data,p,n,0)
#print performance
plt = polt_picture(performance)
plt.show()

