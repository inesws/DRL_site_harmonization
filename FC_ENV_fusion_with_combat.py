# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:38:04 2024

@author: admin
"""



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

import sys
sys.path.insert(0,'C:\\Users\\admin\\Desktop\\Inês\\neurocombat_pyClasse\\combat model')
import Confounder_Correction_Classes
from Confounder_Correction_Classes import ComBatHarmonization,StandardScalerDict, BiocovariatesRegression


data_folder="C:\\Users\\admin\\Desktop\\Inês\\MIICAI\\PRONIA_data_new\\FC_matrices\\"

info_data="C:\\Users\\admin\\Desktop\\Inês\\MIICAI\\PRONIA_data_new\\pronia_dataset_new.mat"
diag_labels="C:\\Users\\admin\\Desktop\\Inês\\MIICAI\\PRONIA_data_new\\diag_dummy.mat"
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
FC_dataset = np.zeros((num_subj, 12720))


for i in range(len(FC_matrices)):
    stacked_FC_matrices[i,:,:]=FC_matrices[i]
    ind=np.flatnonzero(np.triu(FC_matrices[i],k=1))
    FC_dataset[i,:] = np.triu(FC_matrices[i],k=1).flatten()[ind]
    
sites_array = pd.DataFrame(covariates[:,3:]).idxmax(axis=1)

# Name the covariates columns
cov_matrix = pd.DataFrame(covariates[:,0:3],columns=['Age', 'Sex', 'Education'])
cov_matrix = pd.concat([cov_matrix, sites_array],axis=1).rename({0: 'batch'},axis=1)
cov_matrix = pd.concat([cov_matrix, labels.idxmax(axis=1)],axis=1).rename({0: 'Diagn'},axis=1) #.rename({0: 'HC', 1: 'CHR', 2: 'ROD', 3: 'ROP'},axis=1)


# random_state = 42 and 24 -> 2x repeated 5-fold CV
save_results = "C:\\Users\\admin\\Desktop\\Inês\\MIICAI\\new_submission\\FC_ENV_fusion_model_results\\with_combat\\"

fold_results = pd.DataFrame(np.zeros((11,42)), columns=['epochs','train_loss', 'val_loss', 'val_re_rmse',
                                                        'train_sym_rmse', 'val_sym_rmse',
                                                        'z_site_auc_roc', 'z_site_f1',
                                                        'z_Sex_auc_roc', 'z_Sex_f1', 
                                                        'z_Diag_auc_roc', 'z_Diag_f1',
                                                        'z_P_Diag_auc_roc','z_P_Diag_f1',
                                                        'z_Age_rmse', 'z_ERS_rmse',
                                                        'fc_z_site_auc_roc','fc_z_site_f1',
                                                        'fc_z_Sex_auc_roc', 'fc_z_Sex_f1', 
                                                        'fc_z_Diag_auc_roc', 'fc_z_Diag_f1',
                                                        'fc_z_P_Diag_auc_roc','fc_z_P_Diag_f1',
                                                        'fc_z_Age_rmse',
                                                        'env_z_site_auc_roc','env_z_site_f1',
                                                        'env_z_Sex_auc_roc', 'env_z_Sex_f1', 
                                                        'env_z_Diag_auc_roc', 'env_z_Diag_f1',
                                                        'env_z_P_Diag_auc_roc','env_z_P_Diag_f1',
                                                        'env_z_Age_rmse',
                                                        'env_site_auc_roc','env_site_f1',
                                                        'env_Sex_auc_roc', 'env_Sex_f1', 
                                                        'env_Diag_auc_roc', 'env_Diag_f1',
                                                        'env_P_Diag_auc_roc','env_P_Diag_f1'])


random_state=[42,24]
start=[0,5]

for r in range(0,2):
    #random_state = 42 and 24 -> 2x repeated 5-fold CV
    #r=0
    skf = StratifiedKFold(n_splits=5, random_state=random_state[r], shuffle=True)
    print(r)
    
    #i=0
    #(train_ind, val_ind)= list(skf.split(stacked_FC_matrices,labels.iloc[:,0].values))[0]
    for i, (train_ind, val_ind) in enumerate(skf.split(stacked_FC_matrices,labels.iloc[:,0].values)):
        
        i= start[r] + i
        
        print(i)
        
        X_train = FC_dataset[train_ind,:]
        X_val = FC_dataset[val_ind,:]
        
        cov_train=cov_matrix.iloc[train_ind,:].reset_index()
        cov_val=cov_matrix.iloc[val_ind,:].reset_index()
    
        env_train=envir[train_ind,:]
        env_val=envir[val_ind,:]
        
        labels_train = labels.iloc[train_ind,:].reset_index(drop=True)
        labels_val = labels.iloc[val_ind,:].reset_index(drop=True)
        
        BS_withnan_train=BS_withnan[train_ind,:]
        BS_withnan_val = BS_withnan[val_ind,:]
        
        # Re-order the sites ID - needed for combat
        
        # Training set
        cov_train = cov_train.sort_values('batch')
        train_combat_order = list(cov_train.index) # The list of original indexes in the new sorted order
        y_train_sites = cov_train['batch'].values
        cov_train.reset_index(inplace=True)
        env_train=env_train[train_combat_order, :]
        labels_train = labels_train.iloc[train_combat_order, :].reset_index(drop=True)
        BS_withnan_train = BS_withnan[train_combat_order,:]
        
        X_train = X_train[train_combat_order, :]
        
        # Validation set
        cov_val = cov_val.sort_values('batch')
        val_combat_order = list(cov_val.index) # The list of original indexes in the new sorted order
        y_val_sites = cov_val['batch'].values
        cov_val.reset_index(inplace=True)
        env_val = env_val[val_combat_order, :]
        labels_val = labels_val.iloc[val_combat_order, :].reset_index(drop=True)
        BS_withnan_val=BS_withnan[val_combat_order,:]
        
        X_val = X_val[val_combat_order, :]
        
        
        #### Compute environmental risk factor
    
        # replace Nan values with column means
    
        col_mean = np.nanmean(BS_withnan_train, axis=0)
    
        inds = np.where(np.isnan(BS_withnan_train))
    
        env_train[:,10:][inds] = np.take(col_mean,inds[1])
    
        inds = np.where(np.isnan(BS_withnan_val))
        env_val[:,10:][inds] = np.take(col_mean,inds[1])
    
        max_score_per_subject = 6 * 9 + 4*13
    
        train_env_risk=np.sum(env_train,axis=1)/ max_score_per_subject
    
        val_env_risk=np.sum(env_val,axis=1)/ max_score_per_subject
        
        ## SITE CLASSIFICATION

        y_train_multiclass = pd.get_dummies(cov_train, columns=['batch'], drop_first=False, dtype=int).iloc[:,6:]
        y_val_multiclass = pd.get_dummies(cov_val, columns=['batch'], drop_first=False, dtype=int).iloc[:,6:]

        num_classes=y_train_multiclass.shape[1]
        
        
        ## COMBAT HARMONIZATION
        num_feat = X_train.shape[1]
        features = np.arange(0, num_feat)
        
        feat_to_harm = {'fc': {'id': features, 'categorical': ['Sex'],
                               'continuous': ['Age']}}
        
        combat = ComBatHarmonization(cv_method=None, ref_batch=None, 
                                     regression_fit=0, feat_detail=feat_to_harm, 
                                     feat_of_no_interest=None)
        
        X_train_dict={'data': X_train , 'covariates': pd.get_dummies(cov_train, columns=['Diagn'], drop_first=False, dtype=int)}
        
        X_val_dict={'data': X_val , 'covariates': pd.get_dummies(cov_val, columns=['Diagn'], drop_first=False, dtype=int)}
        
        X_train_harm = combat.fit_transform(X_train_dict)
        
        X_val_harm = combat.transform(X_val_dict)
        
        
        # RECONSTRUCT FC MATRIX
        
        ## TRaining set 
        
        num_ROI = 160
        n_train = X_train_harm.shape[0]
        
        X_train = np.zeros((n_train,num_ROI, num_ROI))
        corr_matrix=np.zeros((num_ROI, num_ROI))

        for m in range(n_train):
            
            conn_vector = X_train_harm[m,:]
            
            corr_matrix[0,1:num_ROI]=conn_vector[-num_ROI+1:]
            
            last = -num_ROI+1
            final = -num_ROI+1
            for j in range(1,num_ROI-1):
                
                initial = last
                final = final + (-num_ROI+1+j)
                
                corr_matrix[j, j+1:num_ROI] = conn_vector[final:initial]
                
                last = final
            
            corr_matrix_=corr_matrix + corr_matrix.T - np.diag(np.diag(corr_matrix))
            X_train[m,:,:] = corr_matrix_
        
        ## Validation set
        n_val = X_val_harm.shape[0]
        
        X_val = np.zeros((n_val,num_ROI, num_ROI))
        corr_matrix=np.zeros((num_ROI, num_ROI))

        for m in range(n_val):
            
            conn_vector = X_val_harm[m,:]
            
            corr_matrix[0,1:num_ROI]=conn_vector[-num_ROI+1:]
            
            last = -num_ROI+1
            final = -num_ROI+1
            for j in range(1,num_ROI-1):
                
                initial = last
                final = final + (-num_ROI+1+j)
                
                corr_matrix[j, j+1:num_ROI] = conn_vector[final:initial]
                
                last = final
            
            corr_matrix_=corr_matrix + corr_matrix.T - np.diag(np.diag(corr_matrix))
            X_val[m,:,:] = corr_matrix_
            
        
        # Initialize model
        
        ####______________________Define Model_________________________________________
                
        seed_value = 2020
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
    
        ## FC - ENCODER
        input_matrix = Input(shape=(160,160,1))
        x = input_matrix
        #x = SpatialDropout2D(0.3, data_format="channels_last", input_shape=(18,18,6))(x)
        x = Conv2D(16, (3, 3), data_format="channels_last", activation='selu', kernel_initializer="lecun_normal", padding='same', kernel_regularizer=l2(0.1))(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='selu', kernel_initializer="lecun_normal", padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='selu', kernel_initializer="lecun_normal", padding='same')(x)
        h1_layer = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), strides=1, activation='selu', kernel_initializer="lecun_normal", padding='same')(h1_layer)
        fc_encoded = MaxPooling2D((4,4), padding='same')(x)
        fc_flatten = tf.keras.layers.Flatten()(fc_encoded) 

        fc_latent_dim=list(fc_flatten.shape)
        
        FC_encoder = Model(input_matrix, fc_flatten)
        FC_encoder.summary()
        
        
        ## ENV - ENCODER
        
        input_array = Input(shape=(23,1))
        x = input_array
        x = Conv1D(20, 7, data_format="channels_last", activation='selu', kernel_initializer="lecun_normal", strides=2, padding='same')(x)
        x=BatchNormalization()(x)
        #x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(20, 5, activation='selu', kernel_initializer="lecun_normal", strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        env_encoded = MaxPooling1D(pool_size=2)(x)
        env_flatten = tf.keras.layers.Flatten()(env_encoded)

        latent_dim=list(env_flatten.shape)
        ENV_encoder = Model(input_array, env_flatten)
        ENV_encoder.summary()
        
        
        ## MERGE ENCODER BRANCHES
        env_input = Input(shape=env_flatten.shape[1])
        fc_input = Input(shape=fc_flatten.shape[1])
        
        concat_flatten = Concatenate(axis=1)([fc_input, env_input])
        x = Dense(256, activation='selu', kernel_initializer="lecun_normal")(concat_flatten)
        x = Dropout(0.3)(x)
        z_merged = Dense(128, activation='selu', kernel_initializer="lecun_normal")(x)
        
        inputs = [fc_input,env_input]
        fusion = Model(inputs, z_merged, name = 'fusion_branch')
        fusion.summary()
        
        ## RECONSTRUCT FC MATRIX
        
        latent_inputs=Input(shape=z_merged.shape[1], name='z_merged_latent_inputs')
        x = Dense(3200, activation='selu', kernel_initializer="lecun_normal")(latent_inputs)
        x = Reshape((5, 5, 128))(x)  
        x = UpSampling2D((4, 4))(x)
        x = Conv2D(64, (3, 3), strides=1, activation='selu', kernel_initializer="lecun_normal", padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='selu', kernel_initializer="lecun_normal", padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='selu', kernel_initializer="lecun_normal", padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='selu', kernel_initializer="lecun_normal", padding='same')(x) #activation='sigmoid'
        
        fc_decoder = Model(latent_inputs, decoded, name='decoded')
        fc_decoder.summary()
        
        ## RECONSTRUCT ENVIR ARRAY
        
        latent_inputs=Input(shape=z_merged.shape[1], name='z_merged_latent_inputs')
        x = Dense(120, activation='selu', kernel_initializer="lecun_normal")(latent_inputs)
        x = Reshape((6,20))(x) 
        x = UpSampling1D(size=2)(x)
        x = Conv1D(20, 5, activation='selu', kernel_initializer="lecun_normal", strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        #x = UpSampling1D(size=2)(x)
        x = Conv1DTranspose(1, 7, activation='selu', kernel_initializer="lecun_normal", strides=2, padding='same')(x)
        env_decoded = Cropping1D((0,1))(x)
        
        env_decoder = Model(latent_inputs, env_decoded, name='env_decoded')
        env_decoder.summary()
        
        
        ## DEFINING AUTOENCODER COMPLETE MODEL
        
        #autoencoder = Model(input_matrix, fc_decoder(encoder(input_matrix))
        
        fc_encoded = FC_encoder(input_matrix)
        env_encoded = ENV_encoder(input_array)
        z = fusion([FC_encoder(input_matrix), ENV_encoder(input_array)])      
        
        decoded_fc_output = fc_decoder(z)
        
        decoded_env_output = env_decoder(z)
        
        autoencoder = Model(inputs=[input_matrix, input_array], outputs=[decoded_fc_output, decoded_env_output])
        autoencoder.summary()
    
        batch=128
        epochs=2000
        
        #callback = EarlyStopping(monitor='val_loss', patience = 250, restore_best_weights= True)
    
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), 
                            loss='mean_squared_error') #'binary_cross_entropy
    
        history = autoencoder.fit([X_train, env_train], [X_train, env_train],
                        epochs=epochs,
                        batch_size=batch,
                        shuffle=True, 
                        validation_data=([X_val, env_val], [X_val, env_val]))
        
        loss=history.history['loss']
        loss_val= history.history['val_loss']
        
        fold_results.loc[i, 'train_loss'] = loss[-1]
        fold_results.loc[i, 'val_loss'] = loss_val[-1]
        fold_results.loc[i, 'epochs'] = len(loss)
    
        # plot training
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('MSE loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        
        # plot training
        plt.plot(autoencoder.history.history['decoded_loss'])
        plt.plot(autoencoder.history.history['val_decoded_loss'])
        plt.title('model FC reconstruction loss')
        plt.ylabel('MSE loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        
        # plot training
        plt.plot(autoencoder.history.history['env_decoded_loss'])
        plt.plot(autoencoder.history.history['val_env_decoded_loss'])
        plt.title('model env reconstruction loss')
        plt.ylabel('MSE loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        
        ### 
        y_train_multiclass = np.array(y_train_multiclass)
        y_train_sites =  pd.DataFrame(y_train_multiclass).idxmax(axis=1).values
        
        y_val_multiclass = np.array(y_val_multiclass)
        y_val_sites = pd.DataFrame(y_val_multiclass).idxmax(axis=1).values
        sites = np.unique(y_train_sites)
        
        
        fc_train_encoded = FC_encoder.predict(X_train)
        env_train_encoded = ENV_encoder.predict(env_train)
        z_train = fusion.predict([fc_train_encoded, env_train_encoded])      
        
        fc_train_decoded = fc_decoder.predict(z_train)
        
        fc_val_encoded = FC_encoder.predict(X_val)
        env_val_encoded = ENV_encoder.predict(env_val)
        z_val = fusion.predict([fc_val_encoded, env_val_encoded])      
        
        fc_val_decoded = fc_decoder.predict(z_val)
    
    
        X_val_flatten = X_val.reshape(X_val.shape[0],-1)        
        fold_results.loc[i, 'val_re_rmse'] = mean_squared_error(X_val_flatten, fc_val_decoded.reshape(X_val.shape[0],-1), squared=False)
        
        X_train_flatten = X_train.reshape(X_train.shape[0],-1)        
        tril = np.tril(fc_train_decoded[:,:,:,0])
        triu = np.triu(fc_train_decoded[:,:,:,0])
        fold_results.loc[i, 'train_sym_rmse'] =  mean_squared_error(triu[0,:,:].T, tril[0,:,:], squared=False)
        
        tril = np.tril(fc_val_decoded[:,:,:,0])
        triu = np.triu(fc_val_decoded[:,:,:,0])
        fold_results.loc[i, 'val_sym_rmse'] =  mean_squared_error(triu[0,:,:].T, tril[0,:,:], squared=False)
        
        # plot reconstruction images
        n=6
        for j in range(n):
            ax = plt.subplot(2, n, j+1)
            ax.imshow(X_val[j,:,:].reshape(160,160))
    
            ax.set_axis_off()
            
            ax = plt.subplot(2, n, j+1+n)
            ax.imshow(fc_val_decoded[j,:,:].reshape(160,160))
    
            #plt.axis('off')
            
        #plt.title('Original/ Reconstruction Validation', loc='left')    
        plt.subplots_adjust(wspace=0.8)
    
        plt.show()
    
        # plot reconstruction images
        n=6
        for j in range(n):
            ax = plt.subplot(2, n, j+1)
            ax.imshow(X_train[j,:,:].reshape(160,160))
            ax.set_axis_off()
    
            ax = plt.subplot(2, n, j+1+n)
            ax.imshow(fc_train_decoded[j,:,:].reshape(160,160))
    
        plt.subplots_adjust(wspace=0.8)
        plt.show()
        
        ###
        n_train=X_train.shape[0]
        n_val=X_val.shape[0]
        
        ### CLASSIFICATION CONFOUNDERS APOSTERIORI
        
        print('ARRIVES HERE')
        ## SITE CLASSIFICATION
        
        knn = KNeighborsClassifier(n_neighbors=3)
        svc= SVC(probability=True, random_state=0)
        dt = DecisionTreeClassifier(random_state=0)
        
        vc = VotingClassifier(estimators=[('knn', knn), ('svm', svc ), ('dt', dt)], voting='soft')
        vc.fit(z_train, y_train_sites)
        y_val_pred = vc.predict_proba(z_val)
        
        fold_results.loc[i,'z_site_auc_roc'] = roc_auc_score(y_val_multiclass, y_val_pred, average='macro',multi_class='ovo')
        
        y_val_pred = pd.DataFrame(y_val_pred).idxmax(axis=1).values # vc.predict(z_val)
                
        fold_results.loc[i,'z_site_f1'] = f1_score(y_val_sites, y_val_pred, labels=sites, average='macro')

        print(i, fold_results.loc[i,'z_site_f1'])
        
        #_____based on fc_z_encoded
        
        knn = KNeighborsClassifier(n_neighbors=3)
        svc= SVC(probability=True, random_state=0)
        dt = DecisionTreeClassifier(random_state=0)
        
        vc = VotingClassifier(estimators=[('knn', knn), ('svm', svc ), ('dt', dt)], voting='soft')
        vc.fit(fc_train_encoded, y_train_sites)
        y_val_pred = vc.predict_proba(fc_val_encoded)
        
        
        fold_results.loc[i,'fc_z_site_auc_roc'] = roc_auc_score(y_val_multiclass, y_val_pred, average='macro',multi_class='ovo')
        
        y_val_pred = pd.DataFrame(y_val_pred).idxmax(axis=1).values
        fold_results.loc[i,'fc_z_site_f1'] = f1_score(y_val_sites, y_val_pred, labels=sites, average='macro')
        
        #____based on encoded env
        knn = KNeighborsClassifier(n_neighbors=3)
        svc= SVC(probability=True, random_state=0)
        dt = DecisionTreeClassifier(random_state=0)
        
        vc = VotingClassifier(estimators=[('knn', knn), ('svm', svc ), ('dt', dt)], voting='soft')
        vc.fit(env_train_encoded, y_train_sites)
        y_val_pred = vc.predict_proba(env_val_encoded)

        
        fold_results.loc[i,'env_z_site_auc_roc'] = roc_auc_score(y_val_multiclass, y_val_pred, average='macro',multi_class='ovo')
        
        y_val_pred = pd.DataFrame(y_val_pred).idxmax(axis=1).values
        fold_results.loc[i,'env_z_site_f1'] = f1_score(y_val_sites, y_val_pred, labels=sites, average='macro')
        
        #____based on env
        knn = KNeighborsClassifier(n_neighbors=3)
        svc= SVC(probability=True, random_state=0)
        dt = DecisionTreeClassifier(random_state=0)
        
        vc = VotingClassifier(estimators=[('knn', knn), ('svm', svc ), ('dt', dt)], voting='soft')
        vc.fit(env_train, y_train_sites)
        y_val_pred = vc.predict_proba(env_val)
        
        
        fold_results.loc[i,'env_site_auc_roc'] = roc_auc_score(y_val_multiclass, y_val_pred, average='macro',multi_class='ovo')
        
        y_val_pred = pd.DataFrame(y_val_pred).idxmax(axis=1).values
        fold_results.loc[i,'env_site_f1'] = f1_score(y_val_sites, y_val_pred, labels=sites, average='macro')
        
        
        # SEX CLASSIFICATION
        
        y_train_binary=cov_train.iloc[:,3].values.astype('float32').reshape((-1,1))
        y_val_binary=cov_val.iloc[:,3].values.astype('float32').reshape((-1,1))
        
        knn= KNeighborsClassifier(n_neighbors=3)
        svc= SVC(probability=True,random_state=0)
        dt = DecisionTreeClassifier( random_state=0)
        vc = VotingClassifier(estimators=[('knn', knn), ('svm', svc ), ('dt', dt)], voting='soft')
        vc.fit(z_train, y_train_binary.ravel())
        
        y_val_pred = vc.predict_proba(z_val)
        fold_results.loc[i,'z_Sex_auc_roc'] = roc_auc_score(y_val_binary.ravel(), y_val_pred[:,1])
        
        y_val_pred = pd.DataFrame(y_val_pred).idxmax(axis=1).values
        fold_results.loc[i,'z_Sex_f1'] = f1_score(y_val_binary.ravel(), y_val_pred)
        
        #___based on fc_z_encoded

        
        knn= KNeighborsClassifier(n_neighbors=3)
        svc= SVC(probability=True,random_state=0)
        dt = DecisionTreeClassifier( random_state=0)
        vc = VotingClassifier(estimators=[('knn', knn), ('svm', svc ), ('dt', dt)], voting='soft')
        vc.fit(fc_train_encoded, y_train_binary.ravel())
        y_val_pred = vc.predict_proba(fc_val_encoded)
        
        fold_results.loc[i,'fc_z_Sex_auc_roc'] = roc_auc_score(y_val_binary.ravel(), y_val_pred[:,1])
        
        y_val_pred = pd.DataFrame(y_val_pred).idxmax(axis=1).values
        fold_results.loc[i,'fc_z_Sex_f1'] = f1_score(y_val_binary.ravel(), y_val_pred)
        
        #___based on env_z_encoded
      
        
        knn= KNeighborsClassifier(n_neighbors=3)
        svc= SVC(probability=True,random_state=0)
        dt = DecisionTreeClassifier( random_state=0)
        vc = VotingClassifier(estimators=[('knn', knn), ('svm', svc ), ('dt', dt)], voting='soft')
        vc.fit(env_train_encoded, y_train_binary.ravel())
        y_val_pred = vc.predict_proba(env_val_encoded)
        
        fold_results.loc[i,'env_z_Sex_auc_roc'] = roc_auc_score(y_val_binary.ravel(), y_val_pred[:,1])
        
        y_val_pred = pd.DataFrame(y_val_pred).idxmax(axis=1).values
        fold_results.loc[i,'env_z_Sex_f1'] = f1_score(y_val_binary.ravel(), y_val_pred)
        
        #___based on env

        knn= KNeighborsClassifier(n_neighbors=3)
        svc= SVC(probability=True,random_state=0)
        dt = DecisionTreeClassifier( random_state=0)
        vc = VotingClassifier(estimators=[('knn', knn), ('svm', svc ), ('dt', dt)], voting='soft')
        vc.fit(env_train, y_train_binary.ravel())
        y_val_pred = vc.predict_proba(env_val)
        
        fold_results.loc[i,'env_Sex_auc_roc'] = roc_auc_score(y_val_binary.ravel(), y_val_pred[:,1])
        
        y_val_pred = pd.DataFrame(y_val_pred).idxmax(axis=1).values
        fold_results.loc[i,'env_Sex_f1'] = f1_score(y_val_binary.ravel(), y_val_pred)
        
        
        ### HC vs. P CLASSIFICATION
        
        y_train_binary=np.abs(labels_train.iloc[:,0].values.ravel().astype('float32')-1).reshape((-1,1))
        y_val_binary=np.abs(labels_val.iloc[:,0].values.ravel().astype('float32')-1).reshape((-1,1))
        
        knn= KNeighborsClassifier(n_neighbors=3)
        svc= SVC(probability=True,random_state=0)
        dt = DecisionTreeClassifier( random_state=0)
        vc = VotingClassifier(estimators=[('knn', knn), ('svm', svc ), ('dt', dt)], voting='soft')
        vc.fit(z_train, y_train_binary.ravel())
        y_val_pred = vc.predict_proba(z_val)
        
        fold_results.loc[i,'z_Diag_auc_roc'] = roc_auc_score(y_val_binary.ravel(), y_val_pred[:,1])
        
        y_val_pred = pd.DataFrame(y_val_pred).idxmax(axis=1).values
        fold_results.loc[i,'z_Diag_f1'] = f1_score(y_val_binary.ravel(), y_val_pred)
        
        #___based on fc_z_encoded
        knn= KNeighborsClassifier(n_neighbors=3)
        svc= SVC(probability=True,random_state=0)
        dt = DecisionTreeClassifier( random_state=0)
        vc = VotingClassifier(estimators=[('knn', knn), ('svm', svc ), ('dt', dt)], voting='soft')
        vc.fit(fc_train_encoded, y_train_binary.ravel())
        y_val_pred = vc.predict_proba(fc_val_encoded)
        fold_results.loc[i,'fc_z_Diag_auc_roc'] = roc_auc_score(y_val_binary.ravel(), y_val_pred[:,1])
        
        y_val_pred = pd.DataFrame(y_val_pred).idxmax(axis=1).values
        fold_results.loc[i,'fc_z_Diag_f1'] = f1_score(y_val_binary.ravel(), y_val_pred)    
        
        #___based on env_z_encoded
        
        knn= KNeighborsClassifier(n_neighbors=3)
        svc= SVC(probability=True,random_state=0)
        dt = DecisionTreeClassifier( random_state=0)
        vc = VotingClassifier(estimators=[('knn', knn), ('svm', svc ), ('dt', dt)], voting='soft')
        vc.fit(env_train_encoded, y_train_binary.ravel())
        y_val_pred = vc.predict_proba(env_val_encoded)
        fold_results.loc[i,'env_z_Diag_auc_roc'] = roc_auc_score(y_val_binary.ravel(), y_val_pred[:,1])
        
        y_val_pred = pd.DataFrame(y_val_pred).idxmax(axis=1).values
        fold_results.loc[i,'env_z_Diag_f1'] = f1_score(y_val_binary.ravel(), y_val_pred) 
        
        #___based on env
        
        knn= KNeighborsClassifier(n_neighbors=3)
        svc= SVC(probability=True,random_state=0)
        dt = DecisionTreeClassifier( random_state=0)
        vc = VotingClassifier(estimators=[('knn', knn), ('svm', svc ), ('dt', dt)], voting='soft')
        vc.fit(env_train, y_train_binary.ravel())
        y_val_pred = vc.predict_proba(env_val)
        fold_results.loc[i,'env_Diag_auc_roc'] = roc_auc_score(y_val_binary.ravel(), y_val_pred[:,1])
        
        y_val_pred = pd.DataFrame(y_val_pred).idxmax(axis=1).values
        fold_results.loc[i,'env_Diag_f1'] = f1_score(y_val_binary.ravel(), y_val_pred) 
        
        ### ROP vs. ROD CLASSIFICATION

        train_ind_rop = np.where(labels_train.loc[:,3].values==1)[0]
        train_ind_rod = np.where(labels_train.loc[:,2].values==1)[0]
        
        train_ind_p = list(train_ind_rop) + list(train_ind_rod)
        
        y_train_binary=np.abs(labels_train.iloc[train_ind_p ,3].values.ravel().astype('float32')-1).reshape((-1,1))
        
        
        val_ind_rop = np.where(labels_val.loc[:,3].values==1)[0]
        val_ind_rod = np.where(labels_val.loc[:,2].values==1)[0]
        
        val_ind_p = list(val_ind_rop) + list(val_ind_rod)
        
        y_val_binary=np.abs(labels_val.iloc[val_ind_p,3].values.ravel().astype('float32')-1).reshape((-1,1))
        
        #___based on z
        knn= KNeighborsClassifier(n_neighbors=3)
        svc= SVC(probability=True,random_state=0)
        dt = DecisionTreeClassifier( random_state=0)
        vc = VotingClassifier(estimators=[('knn', knn), ('svm', svc ), ('dt', dt)], voting='soft')
        vc.fit(z_train[train_ind_p,:], y_train_binary.ravel())
        y_val_pred = vc.predict_proba(z_val[val_ind_p,:])
        
        fold_results.loc[i,'z_P_Diag_auc_roc'] = roc_auc_score(y_val_binary.ravel(), y_val_pred[:,1])
        
        y_val_pred = pd.DataFrame(y_val_pred).idxmax(axis=1).values
        fold_results.loc[i,'z_P_Diag_f1'] = f1_score(y_val_binary.ravel(), y_val_pred)
        
        #___based on fc_z_encoded
        
        knn= KNeighborsClassifier(n_neighbors=3)
        svc= SVC(probability=True,random_state=0)
        dt = DecisionTreeClassifier( random_state=0)
        vc = VotingClassifier(estimators=[('knn', knn), ('svm', svc ), ('dt', dt)], voting='soft')
        vc.fit(fc_train_encoded[train_ind_p,:], y_train_binary.ravel())
        y_val_pred = vc.predict_proba(fc_val_encoded[val_ind_p,:])
        
        fold_results.loc[i,'fc_z_P_Diag_auc_roc'] = roc_auc_score(y_val_binary.ravel(), y_val_pred[:,1])
        
        y_val_pred = pd.DataFrame(y_val_pred).idxmax(axis=1).values
        fold_results.loc[i,'fc_z_P_Diag_f1'] = f1_score(y_val_binary.ravel(), y_val_pred)
        
        #___based on env_z_encoded
        
        knn= KNeighborsClassifier(n_neighbors=3)
        svc= SVC(probability=True,random_state=0)
        dt = DecisionTreeClassifier( random_state=0)
        vc = VotingClassifier(estimators=[('knn', knn), ('svm', svc ), ('dt', dt)], voting='soft')
        vc.fit(env_train_encoded[train_ind_p,:], y_train_binary.ravel())
        y_val_pred = vc.predict_proba(env_val_encoded[val_ind_p,:])
        
        fold_results.loc[i,'env_z_P_Diag_auc_roc'] = roc_auc_score(y_val_binary.ravel(), y_val_pred[:,1])
        
        y_val_pred = pd.DataFrame(y_val_pred).idxmax(axis=1).values
        fold_results.loc[i,'env_z_P_Diag_f1'] = f1_score(y_val_binary.ravel(), y_val_pred)
        
        #___based on env
        
        knn= KNeighborsClassifier(n_neighbors=3)
        svc= SVC(probability=True,random_state=0)
        dt = DecisionTreeClassifier( random_state=0)
        vc = VotingClassifier(estimators=[('knn', knn), ('svm', svc ), ('dt', dt)], voting='soft')
        vc.fit(env_train[train_ind_p,:], y_train_binary.ravel())
        y_val_pred = vc.predict_proba(env_val[val_ind_p,:])
        
        fold_results.loc[i,'env_P_Diag_auc_roc'] = roc_auc_score(y_val_binary.ravel(), y_val_pred[:,1])
        
        y_val_pred = pd.DataFrame(y_val_pred).idxmax(axis=1).values
        fold_results.loc[i,'env_P_Diag_f1'] = f1_score(y_val_binary.ravel(), y_val_pred)
        
        ### AGE REGRESSOR
        
        knn = KNeighborsRegressor(n_neighbors=3)
        svr = SVR()
        dt = DecisionTreeRegressor( random_state=0)
        vc = VotingRegressor(estimators=[('knn', knn), ('svm', svr ), ('dt', dt)])
        vc.fit(z_train,  cov_train.iloc[:,2].values)
        y_val_pred = vc.predict(z_val)
        fold_results.loc[i,'z_Age_rmse'] = mean_squared_error(cov_val.iloc[:,2].values.ravel(), y_val_pred, squared=False)
        
        #___based on fc_z_encoded
        knn = KNeighborsRegressor(n_neighbors=3)
        svr = SVR()
        dt = DecisionTreeRegressor( random_state=0)
        vc = VotingRegressor(estimators=[('knn', knn), ('svm', svr ), ('dt', dt)])
        vc.fit(fc_train_encoded,  cov_train.iloc[:,2].values)
        y_val_pred = vc.predict(fc_val_encoded)
        fold_results.loc[i,'fc_z_Age_rmse'] = mean_squared_error(cov_val.iloc[:,2].values.ravel(), y_val_pred, squared=False)
        
        #___based on env_z_encoded
        knn = KNeighborsRegressor(n_neighbors=3)
        svr = SVR()
        dt = DecisionTreeRegressor( random_state=0)
        vc = VotingRegressor(estimators=[('knn', knn), ('svm', svr ), ('dt', dt)])
        vc.fit(env_train_encoded,  cov_train.iloc[:,2].values)
        y_val_pred = vc.predict(env_val_encoded)
        fold_results.loc[i,'env_z_Age_rmse'] = mean_squared_error(cov_val.iloc[:,2].values.ravel(), y_val_pred, squared=False)
        
        
        ### ERS REGRESSOR
        
        knn = KNeighborsRegressor(n_neighbors=3)
        svr = SVR()
        dt = DecisionTreeRegressor( random_state=0)
        vc = VotingRegressor(estimators=[('knn', knn), ('svm', svr ), ('dt', dt)])
        vc.fit(z_train,  train_env_risk)
        y_val_pred = vc.predict(z_val)
        fold_results.loc[i,'z_ERS_rmse'] = mean_squared_error(val_env_risk.ravel(), y_val_pred, squared=False)
        
    
        ### Save 
        
        fold_results.to_csv(os.path.join(save_results, "FC_ENV_fusion_combat_model.csv"))



fold_results.iloc[i+1,:]=fold_results.iloc[0:10,:].mean(axis=0)

fold_results.T.to_csv(os.path.join(save_results, "FC_ENV_fusion_combat_model.csv"))


