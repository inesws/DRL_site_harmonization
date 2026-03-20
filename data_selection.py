
import os, numpy as np
import scipy as sp, scipy.io
import pandas as pd

import tensorflow as tf
from keras.layers import BatchNormalization, Input, Dense, MaxPooling1D, UpSampling1D, Cropping1D, Conv2D, Conv1DTranspose, Conv1D, Concatenate, MaxPooling2D, MaxPooling3D, UpSampling2D, SpatialDropout2D, Reshape, Dropout, ReLU
import sklearn
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, mean_absolute_error, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from keras.models import Model
from tensorflow.keras.models import save_model

#from keras.optimizers import Adadelta
import matplotlib.pyplot as plt
from keras.regularizers import l2, l1
from keras.callbacks import EarlyStopping

from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

data_folder="/mnt/datafast/ines/pronia_fc/FC_matrices/"

info_data="/mnt/datafast/ines/pronia_fc/pronia_dataset_new.mat"
diag_labels="/mnt/datafast/ines/pronia_fc/diag_dummy.mat"
diag_labels=sp.io.loadmat(diag_labels)
info_data=sp.io.loadmat(info_data)

covariates=info_data['demo_vars']
labels=pd.DataFrame(info_data['diag_dummy'])
envir=info_data['ENV']
ITV_measure=info_data['ITV_measure']
ID_cases=info_data['SUBJ_CODES']

# Eliminate subject 368 


covariates=np.delete(covariates,368,axis=0)
labels.drop(368,axis=0,inplace=True)
labels.reset_index(inplace=True, drop=True)
envir=np.delete(envir,368,axis=0)
ITV_measure=np.delete(ITV_measure,368,axis=0)
ID_cases=np.delete(ID_cases,368,axis=0)

BS_withnan=np.where(envir[:,10:]==5, np.nan, envir[:,10:])

num_subj=len(covariates)
num_ROIs=160

FC_matrices=[pd.read_csv(os.path.join(data_folder, subj), delimiter=' ', header=None).to_numpy() for subj in os.listdir(data_folder)]

stacked_FC_matrices=np.zeros((num_subj, num_ROIs, num_ROIs))

for i in range(len(FC_matrices)):
    stacked_FC_matrices[i,:,:]=FC_matrices[i]