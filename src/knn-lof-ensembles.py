'''
Created on Dec 5, 2017

@author: Prasanna
'''
from sklearn.neighbors import LocalOutlierFactor,KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np
from anomdet.neighborhood import KNN,LOF
from anomdet.utils import normalize_scores, regularize_scores
from anomdet.ensemble import combine_scores
import scipy.io

# 1 - diabetes, 2 - breast cancer
dataset = 2

def getDataX(dd):
    if(dataset == 1):
        return dd[:,:8]
    else:
        return dd[:,:30]

def getDatay(dd):
    if(dataset == 1):
        return dd[:,8]
    else:
        return dd[:,30]
    
#Load datasets
if(dataset == 1):    
    file = open("../pima-indians-diabetes.csv")
    data = np.loadtxt(fname = file,delimiter = ",")
    #data_X = data[:,:8]
    #data_y = data[:,8]
    data_X = getDataX(data)
    data_y = getDatay(data)
else:
   bc = scipy.io.loadmat('../wbc_2.mat')
   data = np.concatenate((bc['X'],bc['y']),axis=1)
   print data.shape
   data_X = getDataX(data)
   data_y = getDatay(data)
   
if(dataset == 2):
    for i in range(data_y.__len__()):
        if data_y[i] == 1.0:
            data_y[i] = 0
        else:
            data_y[i] = 1

#select pure and impure classes
if(dataset == 1):
    impure_data,pure_data = np.ndarray(shape=(0,9)),np.ndarray(shape=(0,9))   
    for x in data:
        if (x[8] == 0):
            pure_data = np.concatenate((pure_data,[x]),axis=0)
        else:
            impure_data = np.concatenate((impure_data,[x]),axis=0)
else:
    impure_data,pure_data = np.ndarray(shape=(0,31)),np.ndarray(shape=(0,31))
    for x in data:
        if (x[30] == 0):
            pure_data = np.concatenate((pure_data,[x]),axis=0)
        else:
            impure_data = np.concatenate((impure_data,[x]),axis=0)    

print "Shape of dataset: ", data.shape        
print "Shape of pure Classes: ", pure_data.shape
print "Shape of impure Classes: ", impure_data.shape


'''
run 3 different classifiers on same data and ensemble results knn=15,20 and 25
method -1
'''

clf_auc_scores = []
for i in range(0,25):
    clf_ensemble_scores = []
    for i in [15,20,25]:
        clf = KNN(k=i)
        clf_result = clf.predict(data_X)
        clf_ensemble_scores.append(normalize_scores(clf_result))
   
    for i in [60,65,70]:
        clf = LOF(k=i)
        clf_result = clf.predict(data_X)
        clf_ensemble_scores.append(normalize_scores(regularize_scores(clf_result,1)))
    
    #print clf_ensemble_scores
    clf_ensemble_scores = np.array(clf_ensemble_scores)
    
    clf_combined_scores = combine_scores(clf_ensemble_scores)
    clf_ensemle_auc_score =  roc_auc_score(data_y, clf_combined_scores)
    clf_auc_scores.append(clf_ensemle_auc_score)
print "\nKnn-LOF Ensembles on whole dataset:"
print "Knn-LOF Ensemble method with k=15,20 and 25:"
print "Knn-LOF-Ensemble Classifier AUC Score: (Avg)", np.mean(clf_auc_scores)
#print "Knn-Ensemble Classifier AUC Score: (Std)", np.std(clf_auc_scores)

#Sampling - split into 50 near equal data
clf_auc_scores = []
for i in range(0,25):
    data_samples = np.array_split(data[np.random.permutation(data.shape[0])],20)
    data_samples_y = []
    clf_ensemble_scores =[]
    for data_sample in data_samples:
        #data_sample_X = data_sample[:,:8]
        #data_sample_y = data_sample[:,8]
        data_sample_X = getDataX(data_sample)
        data_sample_y = getDatay(data_sample)
        data_samples_y = np.concatenate((data_samples_y,data_sample_y),axis=0)
        clf_intermediate_results = []
        clf = KNN(k=3)
        clf_result = clf.predict(data_sample_X)
        clf_result = normalize_scores(clf_result)
        clf_intermediate_results.append(clf_result)
        
        clf = LOF(k=5)
        clf_result = clf.predict(data_sample_X)
        clf_result = normalize_scores(regularize_scores(clf_result,1))
        clf_intermediate_results.append(clf_result)
        #combine the intermediate result (KNN + LOF) for one sample before adding to whole
        clf_intermediate_ensemble = combine_scores(clf_intermediate_results)
        clf_ensemble_scores = np.concatenate((clf_ensemble_scores,clf_intermediate_ensemble),axis=0)
    clf_ensemble_scores = np.array(clf_ensemble_scores)
    clf_ensemle_auc_score =  roc_auc_score(data_samples_y, clf_ensemble_scores)
    clf_auc_scores.append(clf_ensemle_auc_score)
print "\nKnn-LOF Ensembles on (10) subsamples dataset with k=3:"
print "Knn-LOF Ensemble Classifier AUC Score: (Avg)", np.mean(clf_auc_scores)
print "Knn-LOF Ensemble Classifier AUC Score: (Std)", np.std(clf_auc_scores)

'''
knn ensemble method3
Sampling - split dataset into 50 near equal sub datasets
Run 3 different knn classifiers with k=3,5 and 10 on each subsample
combine scores - unweighted average
concatenate results
compare with original class labels to get the roc_auc score
'''

clf_auc_scores = []
for j in range(0,25):
    data_samples = np.array_split(data[np.random.permutation(data.shape[0])],20)
    #print len(data_samples)
    data_samples_y = []
    
    clf_total_combined_scores = []
    
    for data_sample in data_samples:
        clf_ensemble_scores = []
        #data_sample_X = data_sample[:,:8]
        #data_sample_y = data_sample[:,8]
        data_sample_X = getDataX(data_sample)
        data_sample_y = getDatay(data_sample)
        clf_ensemble_scores =[]
        data_samples_y = np.concatenate((data_samples_y,data_sample_y),axis=0)
        for i in [3,4,5]:
            clf = KNN(k=i)
            clf_result = clf.predict(data_sample_X)
            clf_ensemble_scores.append(normalize_scores(clf_result))
        for i in [4,5]:
            clf = LOF(k=i)
            clf_result = clf.predict(data_sample_X)
            clf_ensemble_scores.append(normalize_scores(regularize_scores(clf_result,1)))
        #print clf_ensemble_scores
        clf_combined_scores = combine_scores(clf_ensemble_scores)
        #print clf_combined_scores
        clf_total_combined_scores = np.concatenate((clf_total_combined_scores,clf_combined_scores),axis=0)
    
    clf_ensemble_auc_score =  roc_auc_score(data_samples_y, clf_total_combined_scores)
    clf_auc_scores.append(clf_ensemle_auc_score)
print "\nKnn-LOF Ensembles on (50) subsamples dataset with k=3,4,5 on each subsample:"
print "Knn-LOF Ensemble Classifier AUC Score: (Avg)", np.mean(clf_auc_scores)
print "Knn-LOF Ensemble Classifier AUC Score: (Std)", np.std(clf_auc_scores)


'''
knn ensemble method4
Sampling - split dataset into **50 near equal sub datasets
           sample such a way such that each subset contains about 3% outlier class
Run 3 different knn classifiers with k=3,5 and 10 on each subsample
combine scores - unweighted average
concatenate results
compare with original class labels to get the roc_auc score
'''
clf_auc_scores = []
for j in range(0,25):
    data_samples_impure = np.array_split(impure_data[np.random.permutation(impure_data.shape[0])],50)
    data_samples = np.array_split(pure_data[np.random.permutation(pure_data.shape[0])],50)
    #mix about 3% of outlier to each sample
    for i in range(0,25):
        data_samples[i] = np.concatenate((data_samples[i],data_samples_impure[i]),axis=0)
        
    data_samples_y = []
    
    clf_total_combined_scores = []
    
    for data_sample in data_samples:
        clf_ensemble_scores = []
        #data_sample_X = data_sample[:,:8]
        #data_sample_y = data_sample[:,8]
        data_sample_X = getDataX(data_sample)
        data_sample_y = getDatay(data_sample)
        clf_ensemble_scores =[]
        data_samples_y = np.concatenate((data_samples_y,data_sample_y),axis=0)
        for i in [3,4,5]:
            clf = KNN(k=i)
            clf_result = clf.predict(data_sample_X)
            clf_ensemble_scores.append(normalize_scores(clf_result))
        for i in [4,5]:
            clf = LOF(k=i)
            clf_result = clf.predict(data_sample_X)
            clf_ensemble_scores.append(normalize_scores(regularize_scores(clf_result,1)))
        #print clf_ensemble_scores
        clf_combined_scores = combine_scores(clf_ensemble_scores)
        #print clf_combined_scores
        clf_total_combined_scores = np.concatenate((clf_total_combined_scores,clf_combined_scores),axis=0)
    
    lof_ensemble_auc_score =  roc_auc_score(data_samples_y, clf_total_combined_scores)
    clf_auc_scores.append(clf_ensemle_auc_score)
print "\nKnn-LOF Ensembles on (50 subsamples - outliers equally distributed) subsamples dataset with k=3,5,10 on each subsample:"
print "Knn-LOF Ensemble Classifier AUC Score: (Avg)", np.mean(clf_auc_scores)
print "Knn-LOF Ensemble Classifier AUC Score: (Std)", np.std(clf_auc_scores)