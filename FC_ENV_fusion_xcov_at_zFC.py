# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 19:07:35 2024

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

for i in range(len(FC_matrices)):
    stacked_FC_matrices[i,:,:]=FC_matrices[i]


# random_state = 42 and 24 -> 2x repeated 5-fold CV
save_results = "C:\\Users\\admin\\Desktop\\Inês\\MIICAI\\new_submission\\FC_ENV_fusion_model_results\\with_fc_z_site_xcov\\"

fold_results = pd.DataFrame(np.zeros((11,46)), columns=['epochs','train_loss', 'val_loss', 'val_re_rmse',
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
                                                        'env_P_Diag_auc_roc','env_P_Diag_f1',
                                                        'h1_site_auc_roc', 'h1_site_f1',
                                                        'softmax_site_auc_roc', 'softmax_site_f1'])


random_state=[42,24]
start=[0,5]

for r in range(0,2):
    # random_state = 42 and 24 -> 2x repeated 5-fold CV
    #r=0
    skf = StratifiedKFold(n_splits=5, random_state=random_state[r], shuffle=True)
    print(r)
    
    #i=0
    #(train_ind, val_ind)= list(skf.split(stacked_FC_matrices,labels.iloc[:,0].values))[0]
    for i, (train_ind, val_ind) in enumerate(skf.split(stacked_FC_matrices,labels.iloc[:,0].values)):
        
        i= start[r] + i
        
        print(i)
        
        X_train = stacked_FC_matrices[train_ind,:, :]
        X_val = stacked_FC_matrices[val_ind,:, :]
        
        cov_train=covariates[train_ind,:]
        cov_val=covariates[val_ind,:]
    
        env_train=envir[train_ind,:]
        env_val=envir[val_ind,:]
        
        labels_train = labels.iloc[train_ind,:]
        labels_val = labels.iloc[val_ind,:]
        
        BS_withnan_train=BS_withnan[train_ind,:]
        BS_withnan_val = BS_withnan[val_ind,:]
    
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

        y_train_multiclass=cov_train[:,3:].copy()
        y_val_multiclass=cov_val[:,3:].copy()

        num_classes=y_train_multiclass.shape[1]
        
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
        
        #FC_encoder = Model(input_matrix, fc_flatten)
        #FC_encoder.summary()
        
        # site disentanglement block
        if num_classes>1:
            af='softmax'
            loss_sup=tf.keras.losses.CategoricalCrossentropy()
        else:
            af='sigmoid'
            loss_sup=tf.keras.losses.BinaryCrossentropy()
        
        x = tf.keras.layers.Flatten()(h1_layer)
        x=BatchNormalization()(x)
        x=Dropout(0.5)(x)
        #x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
        supervised_output = Dense(num_classes, activation=af, kernel_regularizer=l2(0.01))(x) #, kernel_regularizer=l2(0.9)
        #supervised_layer = Model(latent_inputs, supervised_output, name='supervised_layer')

        outputs=[fc_flatten, h1_layer, supervised_output]
        FC_encoder = Model(input_matrix, outputs)
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
        y_hat_inputs = Input(shape=(num_classes,), name='y_inputs')
        concat_inputs = Concatenate(axis=1)([latent_inputs, y_hat_inputs ])
        
        x = Dense(3200, activation='selu', kernel_initializer="lecun_normal")(concat_inputs)
        x = Reshape((5, 5, 128))(x)  
        x = UpSampling2D((4, 4))(x)
        x = Conv2D(64, (3, 3), strides=1, activation='selu', kernel_initializer="lecun_normal", padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='selu', kernel_initializer="lecun_normal", padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='selu', kernel_initializer="lecun_normal", padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='selu', kernel_initializer="lecun_normal", padding='same')(x) #activation='sigmoid'
        
        inputs = [latent_inputs, y_hat_inputs]
        fc_decoder = Model(inputs, decoded, name='decoded')
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
        
        class AE(tf.keras.Model):
            def __init__(self, FC_encoder,fc_decoder, ENV_encoder, env_decoder, fusion, n_batch_size, **kwargs):
                super().__init__(**kwargs)
                self.FC_encoder = FC_encoder
                self.fc_decoder = fc_decoder
                self.ENV_encoder = ENV_encoder
                self.env_decoder = env_decoder
                self.fusion = fusion
                self.n_batch_size = n_batch_size
                #self.clf_layer = supervised_layer
                self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
                self.fc_reconstruction_loss_tracker = tf.keras.metrics.Mean(
                    name="fc_reconstruction_loss"
                )
                self.env_reconstruction_loss_tracker = tf.keras.metrics.Mean(
                    name="env_reconstruction_loss"
                )
                self.xcov_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
                self.clf_loss_tracker = tf.keras.metrics.Mean(name='clf_loss')
            
            def compile(self, optimizer_ae, optimizer_clf):
                super().compile()
                self.optimizer_ae = optimizer_ae
                self.optimizer_clf = optimizer_clf
            
            @property
            def metrics(self):
                return [
                    self.total_loss_tracker,
                    self.fc_reconstruction_loss_tracker,
                    self.env_reconstruction_loss_tracker,
                    self.xcov_loss_tracker,
                    self.clf_loss_tracker,
                ]
            
            #def model(self):
            #    x= keras.Input(shape=(160,160,1))
            #    return keras.Model(inputs=[x], outputs=self.decoder(self.encoder(x)))
        
            def train_step(self, data):
        
                x, y=data
                fc, env = x
        
                with tf.GradientTape() as tape: # foward propagation
                
                    fc_z, h1, y_pred = self.FC_encoder(fc, training=True)
                    env_z = self.ENV_encoder(env, training=True)
                    z = self.fusion([fc_z,env_z], training=True)
                    
                    FC_reconstruction = self.fc_decoder([z, y_pred], training=True)
                    ENV_reconstruction = self.env_decoder(z, training=True)
                    
                    
                    re_loss=tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')
                    fc_reconstruction_loss=re_loss(fc, FC_reconstruction) 
                    env_reconstruction_loss=re_loss(env, ENV_reconstruction) 
                  
                    clf_loss=loss_sup(y, y_pred)
                          
                    y_pred_mean_over_batches=tf.reduce_mean(y_pred,axis=0, keepdims=True)
                    z_mean_over_batches=tf.reduce_mean(fc_z, axis=0, keepdims=True) # should do the mean over all subjects in batch
        
                    z_flatten = tf.keras.layers.Flatten()(fc_z-z_mean_over_batches)
                    z_conf_flatten = (y_pred-y_pred_mean_over_batches)
        
                    #tf.matmul does (n_batch,n_classes)x(n_batch,n_lantent_feat) -> (n_classes, n_latent_feat): each cell in this
                    # matrix has come from (class_11 * feat_11 + class_12 * feat_12 + ... + class_1N * feat_1N )
                    # so each matrix entry is the summation of multiplication of feature i and class i for n_batches 
                    # each cell must be then divided by N_batches
                    xcov_loss = 0.5*tf.reduce_sum(
                        tf.square(
                        tf.matmul(
                            z_conf_flatten,z_flatten,
                            transpose_a=True
                            ) / self.n_batch_size
                            )
                        )
                    
                    gamma=15
                   
                    alpha=1
                    beta=1
                    total_loss = alpha*fc_reconstruction_loss  + gamma*xcov_loss + alpha*env_reconstruction_loss 
        
                print("Shape of y in train_step:", y.shape)
                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer_ae.apply_gradients(zip(grads, self.trainable_weights))
                self.total_loss_tracker.update_state(total_loss)
                self.fc_reconstruction_loss_tracker.update_state(fc_reconstruction_loss)
                self.env_reconstruction_loss_tracker.update_state(env_reconstruction_loss)
                self.xcov_loss_tracker.update_state(xcov_loss)
                
                with tf.GradientTape() as tape: # foward propagation
                    fc_z, h1, y_pred = self.FC_encoder(fc, training=True)
                    env_z = self.ENV_encoder(env, training=True)
                    z = self.fusion([fc_z,env_z], training=True)
                    
                    FC_reconstruction = self.fc_decoder([z, y_pred], training=True)
                    ENV_reconstruction = self.env_decoder(z, training=True)
        
                    clf_loss=loss_sup(y, y_pred)
                
                grads = tape.gradient(clf_loss, self.trainable_weights)
                self.optimizer_clf.apply_gradients(zip(grads, self.trainable_weights))
                self.clf_loss_tracker.update_state(clf_loss)
                    
                return {
                    "loss": self.total_loss_tracker.result(),
                    "fc_reconstruction_loss": self.fc_reconstruction_loss_tracker.result(),
                    "env_reconstruction_loss": self.env_reconstruction_loss_tracker.result(),
                    "clf_loss": self.clf_loss_tracker.result(),
                    "xcov_loss": self.xcov_loss_tracker.result()
                }
            
            def test_step(self, data):
                
                x, y=data
                fc, env = x
                
                fc_z, h1, y_pred = self.FC_encoder(fc, training=False)
                env_z = self.ENV_encoder(env, training=False)
                z = self.fusion([fc_z,env_z], training=False)
                    
                FC_reconstruction = self.fc_decoder([z, y_pred], training=False)
                ENV_reconstruction = self.env_decoder(z, training=False)
        
                re_loss=tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')#reduction='sum_over_batch_size'
                fc_reconstruction_loss=re_loss(fc, FC_reconstruction)
                env_reconstruction_loss=re_loss(env, ENV_reconstruction)

        
                clf_loss=loss_sup(y, y_pred)
        
                y_pred_mean_over_batches=tf.reduce_mean(y_pred,axis=0, keepdims=True) #z_confounded
                z_mean_over_batches=tf.reduce_mean(fc_z, axis=0, keepdims=True) # should do the mean over all subjects in batch
                #print("Shape of z_mean in train_step:", z_mean_over_batches.shape)
                #print("Shape of z-z_mean in train_step:", (z-z_mean_over_batches).shape)
                z_flatten = tf.keras.layers.Flatten()(fc_z-z_mean_over_batches)
                z_conf_flatten = (y_pred-y_pred_mean_over_batches) #z_confounded
                n_samples=tf.cast(tf.shape(fc)[0], tf.float32)
                #print("Shape of encoded_flatten in train_step:", z_flatten.shape)
        
                xcov_loss = 0.5*tf.reduce_sum(
                    tf.square(
                    tf.matmul(
                        z_conf_flatten,z_flatten,
                        transpose_a=True
                        ) / n_samples
                        )
                    )
        
                gamma=15
            
                alpha=1
                beta=1
                total_loss = alpha*fc_reconstruction_loss + gamma*xcov_loss + alpha*env_reconstruction_loss 
              
                self.total_loss_tracker.update_state(total_loss)
                self.fc_reconstruction_loss_tracker.update_state(fc_reconstruction_loss)
                self.env_reconstruction_loss_tracker.update_state(env_reconstruction_loss)
                self.clf_loss_tracker.update_state(clf_loss)
                self.xcov_loss_tracker.update_state(xcov_loss)
        
                return {
                    "loss": self.total_loss_tracker.result(),
                    "fc_reconstruction_loss": self.fc_reconstruction_loss_tracker.result(),
                    "env_reconstruction_loss": self.env_reconstruction_loss_tracker.result(),
                    "clf_loss": self.clf_loss_tracker.result(),
                    "xcov_loss": self.xcov_loss_tracker.result()
                }
        
        
        batch=128
        epochs=2000
        
        initial_learning_rate = 0.0001
        final_learning_rate = 0.00001 #0.000001
        learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)
        steps_per_epoch = int(X_train.shape[0]/batch)
        
        lrs = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, steps_per_epoch, learning_rate_decay_factor, staircase=False, name=None
        )
        
        autoencoder_2 = AE(FC_encoder, fc_decoder, ENV_encoder, env_decoder, fusion, batch)
        
        
        autoencoder_2.compile(optimizer_ae=tf.keras.optimizers.Adam(learning_rate=0.0002),
                              optimizer_clf=tf.keras.optimizers.Adam(learning_rate=lrs)) #0.00002 learning_rate=0.00009
        #callback = EarlyStopping(monitor='val_clf_loss', patience = 250, restore_best_weights= True)
        

        history = autoencoder_2.fit(x=[X_train, env_train], y=y_train_multiclass,
                        epochs=epochs,
                        batch_size=batch,
                        shuffle=True, 
                        validation_data=([X_val, env_val], y_val_multiclass))
        ##_______________________________________________________________
        
        total_loss = autoencoder_2.history.history['loss']
        fold_results.loc[i, 'train_total_loss'] = total_loss[-1]
        
        val_total_loss = autoencoder_2.history.history['val_loss']
        fold_results.loc[i, 'val_total_loss'] = val_total_loss[-1]
        
        loss = autoencoder_2.history.history['fc_reconstruction_loss']
        fold_results.loc[i, 'train_fc_re_loss'] = loss[-1]
        val_loss = autoencoder_2.history.history['val_fc_reconstruction_loss']
        fold_results.loc[i, 'val_fc_re_loss'] = val_loss[-1]
        
        plt.figure()
        epochs = range(len(loss))
        fold_results.loc[i, 'epochs'] = len(loss)
        plt.plot(epochs, loss, 'b', label='Training FC RE loss')
        plt.plot(epochs, val_loss, 'g', label='Validation FC RE loss')
        plt.title('Training and validation FC reconstruction loss')
        plt.legend()
        plt.show()
        
        loss = autoencoder_2.history.history['env_reconstruction_loss']
        fold_results.loc[i, 'train_env_re_loss'] = loss[-1]
        val_loss = autoencoder_2.history.history['val_env_reconstruction_loss']
        fold_results.loc[i, 'val_env_re_loss'] = val_loss[-1]
        
        plt.figure()
        epochs = range(len(loss))
        fold_results.loc[i, 'epochs'] = len(loss)
        plt.plot(epochs, loss, 'b', label='Training ENV RE loss')
        plt.plot(epochs, val_loss, 'g', label='Validation ENV RE loss')
        plt.title('Training and validation ENV reconstruction loss')
        plt.legend()
        plt.show()
          
        sup_loss = autoencoder_2.history.history['clf_loss']
        sup_val_loss = autoencoder_2.history.history['val_clf_loss']
        fold_results.loc[i, 'train_clf_loss'] = sup_loss[-1]
        fold_results.loc[i, 'val_clf_loss'] = sup_val_loss[-1]
        
        plt.figure()
        plt.plot(epochs, sup_loss, 'b', label='Training clf loss')
        plt.plot(epochs, sup_val_loss, 'g', label='Validation clf loss')
        plt.title('Training and validation clf loss')
        plt.legend()
        plt.show()
        
        xcov_loss = autoencoder_2.history.history['xcov_loss']
        xcov_val_loss = autoencoder_2.history.history['val_xcov_loss']
        fold_results.loc[i, 'train_xcov_loss'] = xcov_loss[-1]
        fold_results.loc[i, 'val_xcov_loss'] = xcov_val_loss[-1]
        plt.figure()
        plt.plot(epochs, xcov_loss, 'b', label='Training xcov loss')
        plt.plot(epochs, xcov_val_loss, 'g', label='Validation xcov loss')
        plt.title('Training and validation classification loss')
        plt.legend()
        plt.show()
        
        ### 
        
        y_train_sites =  pd.DataFrame(y_train_multiclass).idxmax(axis=1).values
        y_val_sites = pd.DataFrame(y_val_multiclass).idxmax(axis=1).values
        sites = np.unique(y_train_sites)
        
        
        fc_train_encoded, h1_train, y_train_sites_pred = FC_encoder.predict(X_train)
        env_train_encoded = ENV_encoder.predict(env_train)
        z_train = fusion.predict([fc_train_encoded, env_train_encoded])      
        
        fc_train_decoded = fc_decoder.predict([z_train,y_train_sites_pred] )
        
        fc_val_encoded,  h1_val , y_val_sites_pred = FC_encoder.predict(X_val)
        env_val_encoded = ENV_encoder.predict(env_val)
        z_val = fusion.predict([fc_val_encoded, env_val_encoded])      
        
        fc_val_decoded = fc_decoder.predict([z_val, y_val_sites_pred])
        
        #____Site classification based on softmax layer
      
        fold_results.loc[i,'softmax_site_auc_roc'] = roc_auc_score(y_val_multiclass, y_val_sites_pred, average='macro',multi_class='ovo')

        y_val_pred = pd.DataFrame(y_val_sites_pred).idxmax(axis=1).values
        fold_results.loc[i,'softmax_site_f1'] = f1_score(y_val_sites, y_val_pred, labels=sites, average='macro')
        
    
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
        
        """
        #____based on h1
        knn = KNeighborsClassifier(n_neighbors=3)
        svc= SVC(probability=True, random_state=0)
        dt = DecisionTreeClassifier(random_state=0)
        
        vc = VotingClassifier(estimators=[('knn', knn), ('svm', svc ), ('dt', dt)], voting='soft')
        h1_train_flat = h1_train.reshape(h1_train.shape[0],-1)  
        vc.fit(h1_train_flat, y_train_sites)
        h1_val_flat = h1_val.reshape(h1_val.shape[0],-1)  
        y_val_pred = vc.predict_proba(h1_val_flat)
        
        fold_results.loc[i,'h1_site_auc_roc'] = roc_auc_score(y_val_multiclass, y_val_pred, average='macro',multi_class='ovo')
        
        y_val_pred = pd.DataFrame(y_val_pred).idxmax(axis=1).values
        fold_results.loc[i,'h1_site_f1'] = f1_score(y_val_sites, y_val_pred, labels=sites, average='macro')
        """
        
        # SEX CLASSIFICATION
        
        y_train_binary=cov_train[:,1].astype('float32').reshape((-1,1))
        y_val_binary=cov_val[:,1].astype('float32').reshape((-1,1))
        
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
        y_train_binary=cov_train[:,1].astype('float32').reshape((-1,1))
        y_val_binary=cov_val[:,1].astype('float32').reshape((-1,1))
        
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
        y_train_binary=cov_train[:,1].astype('float32').reshape((-1,1))
        y_val_binary=cov_val[:,1].astype('float32').reshape((-1,1))
        
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
        y_train_binary=cov_train[:,1].astype('float32').reshape((-1,1))
        y_val_binary=cov_val[:,1].astype('float32').reshape((-1,1))
        
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
        vc.fit(z_train,  cov_train[:,0])
        y_val_pred = vc.predict(z_val)
        fold_results.loc[i,'z_Age_rmse'] = mean_squared_error(cov_val[:,0].ravel(), y_val_pred, squared=False)
        
        #___based on fc_z_encoded
        knn = KNeighborsRegressor(n_neighbors=3)
        svr = SVR()
        dt = DecisionTreeRegressor( random_state=0)
        vc = VotingRegressor(estimators=[('knn', knn), ('svm', svr ), ('dt', dt)])
        vc.fit(fc_train_encoded,  cov_train[:,0])
        y_val_pred = vc.predict(fc_val_encoded)
        fold_results.loc[i,'fc_z_Age_rmse'] = mean_squared_error(cov_val[:,0].ravel(), y_val_pred, squared=False)
        
        #___based on env_z_encoded
        knn = KNeighborsRegressor(n_neighbors=3)
        svr = SVR()
        dt = DecisionTreeRegressor( random_state=0)
        vc = VotingRegressor(estimators=[('knn', knn), ('svm', svr ), ('dt', dt)])
        vc.fit(env_train_encoded,  cov_train[:,0])
        y_val_pred = vc.predict(env_val_encoded)
        fold_results.loc[i,'env_z_Age_rmse'] = mean_squared_error(cov_val[:,0].ravel(), y_val_pred, squared=False)
        
        
        ### ERS REGRESSOR
        
        knn = KNeighborsRegressor(n_neighbors=3)
        svr = SVR()
        dt = DecisionTreeRegressor( random_state=0)
        vc = VotingRegressor(estimators=[('knn', knn), ('svm', svr ), ('dt', dt)])
        vc.fit(z_train,  train_env_risk)
        y_val_pred = vc.predict(z_val)
        fold_results.loc[i,'z_ERS_rmse'] = mean_squared_error(val_env_risk.ravel(), y_val_pred, squared=False)
        
        ### Save 
        
        fold_results.to_csv(os.path.join(save_results, "FC_ENV_fusion_xcov_model.csv"))



fold_results.iloc[i+1,:]=fold_results.iloc[0:10,:].mean(axis=0)

fold_results.T.to_csv(os.path.join(save_results, "FC_ENV_fusion_xcov_model.csv"))


