"""
xcov version of the FC+ENV harmonization script.

Base architecture is the original XCov script, with:
- xcov loss using an RBF kernel
- median heuristic for sigma when no fixed sigma is provided
- MLP predictors for all identifiability tasks
- convergence tracking for every MLP model
- separate CSV exports for main results, MLP results, and convergence summary
"""

import os
import numpy as np
import scipy as sp
import scipy.io
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import traceback

from keras.layers import (
    BatchNormalization,
    Input,
    Dense,
    MaxPooling1D,
    UpSampling1D,
    Cropping1D,
    Conv2D,
    Conv1DTranspose,
    Conv1D,
    Concatenate,
    MaxPooling2D,
    UpSampling2D,
    Reshape,
    Dropout,
)
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model

from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor


data_folder = "/mnt/datafast/ines/pronia_fc/FC_matrices/"
info_data_path = "/mnt/datafast/ines/pronia_fc/pronia_dataset_new.mat"
diag_labels_path = "/mnt/datafast/ines/pronia_fc/diag_dummy.mat"

save_results = "results/fusion_z_site_xcov_15_05_26/"
os.makedirs(save_results, exist_ok=True)
output_base_name = "fusion_z_site_xcov_h1_gamma100_alphabeta1_jointopt_sex"

seed_value = 2020
batch_size = 128
epochs = 2000
initial_learning_rate = 0.0001
final_learning_rate = 0.00001
learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / epochs)
gamma_xcov = 100.0 #15
alpha = 1.0
beta = 1.0
random_state = [42, 24]
start = [0, 5]


def _load_data():
    info_data = sp.io.loadmat(info_data_path)
    _ = sp.io.loadmat(diag_labels_path)

    covariates = info_data["demo_vars"]
    labels = pd.DataFrame(info_data["diag_dummy"])
    envir = info_data["ENV"]
    itv_measure = info_data["ITV_measure"]
    id_cases = info_data["SUBJ_CODES"]

    excluded_subject_idx = 368
    covariates = np.delete(covariates, excluded_subject_idx, axis=0)
    labels = labels.drop(excluded_subject_idx, axis=0).reset_index(drop=True)
    envir = np.delete(envir, excluded_subject_idx, axis=0)
    itv_measure = np.delete(itv_measure, excluded_subject_idx, axis=0)
    id_cases = np.delete(id_cases, excluded_subject_idx, axis=0)
    _ = itv_measure, id_cases

    bs_with_nan = np.where(envir[:, 10:] == 5, np.nan, envir[:, 10:])

    num_subj = len(covariates)
    num_rois = 160
    fc_files = sorted(os.listdir(data_folder))
    fc_matrices = [
        pd.read_csv(os.path.join(data_folder, subj), delimiter=" ", header=None).to_numpy()
        for subj in fc_files
    ]

    stacked_fc_matrices = np.zeros((num_subj, num_rois, num_rois), dtype=np.float32)
    for i, mat in enumerate(fc_matrices[:num_subj]):
        stacked_fc_matrices[i, :, :] = mat

    return stacked_fc_matrices, covariates, labels, envir, bs_with_nan



def _xcov_loss(z, y_pred, n_batch_size):
    
    y_pred_mean_over_batches=tf.reduce_mean(y_pred,axis=0, keepdims=True)
    z_mean_over_batches=tf.reduce_mean(z, axis=0, keepdims=True) # should do the mean over all subjects in batch
        
    z_flatten = tf.keras.layers.Flatten()(z-z_mean_over_batches)
    z_conf_flatten = (y_pred-y_pred_mean_over_batches)
        
    #tf.matmul does (n_batch,n_classes)x(n_batch,n_lantent_feat) -> (n_classes, n_latent_feat): each cell in this
    # matrix has come from (class_11 * feat_11 + class_12 * feat_12 + ... + class_1N * feat_1N )
    # so each matrix entry is the summation of multiplication of feature i and class i for n_batches 
    # each cell must be then divided by N_batches
    xcov_loss = 0.5*tf.reduce_sum(tf.square(tf.matmul(z_conf_flatten,
                                                      z_flatten,
                                                      transpose_a=True
                                                    ) / n_batch_size
                                           )
                             )
                    
    return xcov_loss



def _fit_voting_classifier(X_train, y_train, X_val, y_val, average_mode="binary", labels_for_f1=None):
    knn = KNeighborsClassifier(n_neighbors=3)
    svc = SVC(probability=True, random_state=0)
    dt = DecisionTreeClassifier(random_state=0)
    vc = VotingClassifier(estimators=[("knn", knn), ("svm", svc), ("dt", dt)], voting="soft")
    vc.fit(X_train, y_train)
    y_val_pred_proba = vc.predict_proba(X_val)
    y_val_pred = pd.DataFrame(y_val_pred_proba).idxmax(axis=1).values
    if average_mode == "binary":
        auc = roc_auc_score(y_val, y_val_pred_proba[:, 1])
        f1 = f1_score(y_val, y_val_pred)
    else:
        auc = roc_auc_score(y_val, y_val_pred_proba, average="macro", multi_class="ovo")
        f1 = f1_score(y_val, y_val_pred, labels=labels_for_f1, average="macro")
    return auc, f1



def _fit_mlp_classifier(X_train, y_train, X_val, name, fold_idx, convergence_rows, hidden_layers=(64,), max_iter=300):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=max_iter, random_state=0)
    mlp.fit(X_train, y_train)
    y_val_pred_proba = mlp.predict_proba(X_val)
    converged = mlp.n_iter_ < mlp.max_iter
    if not converged:
        convergence_rows.append(
            {
                "fold": fold_idx,
                "model": name,
                "max_iter": mlp.max_iter,
                "n_iter": mlp.n_iter_,
                "converged": False,
            }
        )
    return mlp, y_val_pred_proba, converged



def _fit_mlp_regressor(X_train, y_train, X_val, name, fold_idx, convergence_rows, hidden_layers=(128, 64), max_iter=500):
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=max_iter, random_state=0)
    mlp.fit(X_train, y_train)
    y_val_pred = mlp.predict(X_val)
    converged = mlp.n_iter_ < mlp.max_iter
    if not converged:
        convergence_rows.append(
            {
                "fold": fold_idx,
                "model": name,
                "max_iter": mlp.max_iter,
                "n_iter": mlp.n_iter_,
                "converged": False,
            }
        )
    return mlp, y_val_pred, converged



def _safe_auc_f1_binary(y_true, y_pred_proba):
    auc = roc_auc_score(y_true.ravel(), y_pred_proba[:, 1])
    y_pred = pd.DataFrame(y_pred_proba).idxmax(axis=1).values
    f1 = f1_score(y_true.ravel(), y_pred)
    return auc, f1



def _safe_auc_f1_multiclass(y_true_onehot, y_pred_proba, labels_for_f1):
    # Convert one-hot encoded labels to class indices for sklearn metrics
    y_true_labels = pd.DataFrame(y_true_onehot).idxmax(axis=1).values
    auc = roc_auc_score(y_true_onehot, y_pred_proba, average="macro", multi_class="ovo")
    y_pred = pd.DataFrame(y_pred_proba).idxmax(axis=1).values
    f1 = f1_score(y_true_labels, y_pred, labels=labels_for_f1, average="macro")
    return auc, f1



def _build_models(num_classes, num_classes_sex):
    input_matrix = Input(shape=(160, 160, 1))
    x = input_matrix
    x = Conv2D(16, (3, 3), data_format="channels_last", activation="selu", kernel_initializer="lecun_normal", padding="same", kernel_regularizer=l2(0.1))(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(32, (3, 3), activation="selu", kernel_initializer="lecun_normal", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(64, (3, 3), activation="selu", kernel_initializer="lecun_normal", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(128, (3, 3), strides=1, activation="selu", kernel_initializer="lecun_normal", padding="same")(x)
    fc_encoded = MaxPooling2D((4, 4), padding="same")(x)
    fc_flatten = tf.keras.layers.Flatten()(fc_encoded)

    FC_encoder = Model(input_matrix, fc_flatten)
    FC_encoder.summary()


    input_array = Input(shape=(23, 1))
    x = input_array
    x = Conv1D(20, 7, data_format="channels_last", activation="selu", kernel_initializer="lecun_normal", strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv1D(20, 5, activation="selu", kernel_initializer="lecun_normal", strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    env_encoded = MaxPooling1D(pool_size=2)(x)
    env_flatten = tf.keras.layers.Flatten()(env_encoded)
    ENV_encoder = Model(input_array, env_flatten, name="env_encoder")

    env_input = Input(shape=(env_flatten.shape[1],))
    fc_input = Input(shape=(fc_flatten.shape[1],))

    concat_flatten = Concatenate(axis=1)([fc_input, env_input])
    h1_layer = Dense(256, activation="selu", kernel_initializer="lecun_normal")(concat_flatten)
    x = Dropout(0.3)(h1_layer)
    z_merged = Dense(128, activation="selu", kernel_initializer="lecun_normal")(x)

    if num_classes > 1:
        af = "softmax"
        loss_sup = tf.keras.losses.CategoricalCrossentropy()
    else:
        af = "sigmoid"
        loss_sup = tf.keras.losses.BinaryCrossentropy()

    x = tf.keras.layers.Flatten()(h1_layer) #fc_encoded)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    supervised_output = Dense(num_classes, activation=af, kernel_regularizer=l2(0.01))(x)
    
    if num_classes_sex > 1:
        af = "softmax"
        loss_sup_sex = tf.keras.losses.CategoricalCrossentropy()
    else:
        af = "sigmoid"
        loss_sup_sex = tf.keras.losses.BinaryCrossentropy()

    # Sex disentanglement block
    x2 = tf.keras.layers.Flatten()(h1_layer)
    x2=BatchNormalization()(x2)
    x2=Dropout(0.5)(x2)
    #x2 = Dense(128, activation='selu', kernel_initializer="lecun_normal",  kernel_regularizer=l2(0.05))(x2)
    #x2=Dropout(0.5)(x2)
    supervised_output_sex = Dense(num_classes_sex, activation='softmax', kernel_regularizer=l2(0.05))(x2)
        

    fusion = Model([fc_input,env_input], [z_merged, h1_layer, supervised_output, supervised_output_sex], name="fusion_branch")
    fusion.summary()


    latent_inputs = Input(shape=(z_merged.shape[1],), name="z_merged_latent_inputs")
    y_hat_inputs = Input(shape=(num_classes,), name="y_inputs")
    y_hat_sex_inputs = Input(shape=(num_classes_sex,), name="y_sex_inputs")

    concat_inputs = Concatenate(axis=1)([latent_inputs, y_hat_inputs, y_hat_sex_inputs])
    x = Dense(3200, activation="selu", kernel_initializer="lecun_normal")(concat_inputs)
    x = Reshape((5, 5, 128))(x)
    x = UpSampling2D((4, 4))(x)
    x = Conv2D(64, (3, 3), strides=1, activation="selu", kernel_initializer="lecun_normal", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation="selu", kernel_initializer="lecun_normal", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation="selu", kernel_initializer="lecun_normal", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation="selu", kernel_initializer="lecun_normal", padding="same")(x)
    fc_decoder = Model([latent_inputs, y_hat_inputs, y_hat_sex_inputs], decoded, name="decoded")

    latent_inputs = Input(shape=(z_merged.shape[1],), name="z_merged_latent_inputs")
    y_hat_inputs = Input(shape=(num_classes,), name='y_inputs')
    concat_inputs = Concatenate(axis=1)([latent_inputs, y_hat_inputs, y_hat_sex_inputs ])
    x = Dense(120, activation="selu", kernel_initializer="lecun_normal")(concat_inputs)
    x = Reshape((6, 20))(x)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(20, 5, activation="selu", kernel_initializer="lecun_normal", strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv1DTranspose(1, 7, activation="selu", kernel_initializer="lecun_normal", strides=2, padding="same")(x)
    env_decoded = Cropping1D((0, 1))(x)
    env_decoder = Model([latent_inputs, y_hat_inputs, y_hat_sex_inputs], env_decoded, name="env_decoded")

    return FC_encoder, ENV_encoder, fusion, fc_decoder, env_decoder, loss_sup, loss_sup_sex


class AE(tf.keras.Model):
    def __init__(self, FC_encoder, fc_decoder, ENV_encoder, env_decoder, fusion, loss_sup,loss_sup_sex, n_batch_size, alpha=1.0, gamma=1.0, **kwargs):
        super().__init__(**kwargs)
        self.FC_encoder = FC_encoder
        self.fc_decoder = fc_decoder
        self.ENV_encoder = ENV_encoder
        self.env_decoder = env_decoder
        self.fusion = fusion
        self.loss_sup = loss_sup
        self.loss_sup_sex = loss_sup_sex
        self.n_batch_size = n_batch_size
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.fc_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="fc_reconstruction_loss")
        self.env_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="env_reconstruction_loss")
        self.xcov_loss_tracker = tf.keras.metrics.Mean(name="xcov_loss")
        self.xcov_sex_loss_tracker = tf.keras.metrics.Mean(name="xcov_sex_loss")
        self.clf_sex_loss_tracker = tf.keras.metrics.Mean(name="clf_sex_loss")
        self.clf_loss_tracker = tf.keras.metrics.Mean(name="clf_loss")

    def compile(self, optimizer_ae, jit_compile=False): #optimizer_clf
        super().compile(jit_compile=jit_compile)
        self.optimizer_ae = optimizer_ae
        #self.optimizer_clf = optimizer_clf

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.fc_reconstruction_loss_tracker,
            self.env_reconstruction_loss_tracker,
            self.xcov_loss_tracker,
            self.xcov_sex_loss_tracker,
            self.clf_sex_loss_tracker,
            self.clf_loss_tracker,
        ]

    def train_step(self, data):
        x, y = data
        fc, env = x
        y, y_sex_true = y

        with tf.GradientTape() as tape:

            fc_z  = self.FC_encoder(fc, training=True)
            env_z = self.ENV_encoder(env, training=True)
            z, _ ,y_pred, y_sex_pred = self.fusion([fc_z, env_z], training=True)

            fc_reconstruction = self.fc_decoder([z, y_pred, y_sex_pred], training=True)
            env_reconstruction = self.env_decoder([z, y_pred, y_sex_pred], training=True)

            re_loss = tf.keras.losses.MeanSquaredError(reduction="sum_over_batch_size")
            fc_reconstruction_loss = re_loss(fc, fc_reconstruction)
            env_reconstruction_loss = re_loss(env, env_reconstruction)

            clf_loss = self.loss_sup(y, y_pred)
            clf_sex_loss = self.loss_sup_sex(y_sex_true, y_sex_pred)

            xcov_loss = _xcov_loss(z, y_pred, self.n_batch_size)
            xcov_sex_loss = _xcov_loss(z, y_sex_pred, self.n_batch_size)

            total_loss = self.alpha * (fc_reconstruction_loss + env_reconstruction_loss) + self.beta* (clf_loss + clf_sex_loss) + self.gamma * (xcov_loss +  xcov_sex_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer_ae.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.fc_reconstruction_loss_tracker.update_state(fc_reconstruction_loss)
        self.env_reconstruction_loss_tracker.update_state(env_reconstruction_loss)
        self.xcov_loss_tracker.update_state(xcov_loss)
        self.xcov_sex_loss_tracker.update_state(xcov_sex_loss)
        self.clf_loss_tracker.update_state(clf_loss)
        self.clf_sex_loss_tracker.update_state(clf_sex_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "fc_reconstruction_loss": self.fc_reconstruction_loss_tracker.result(),
            "env_reconstruction_loss": self.env_reconstruction_loss_tracker.result(),
            "xcov_loss": self.xcov_loss_tracker.result(),
            "xcov_sex_loss": self.xcov_sex_loss_tracker.result(),
            "clf_loss": self.clf_loss_tracker.result(),
            "clf_sex_loss": self.clf_sex_loss_tracker.result(),
        }

    def test_step(self, data):
        x, y = data
        fc, env = x
        y, y_sex_true = y

        fc_z  = self.FC_encoder(fc, training=False)
        env_z = self.ENV_encoder(env, training=False)
        z, _ ,y_pred, y_sex_pred = self.fusion([fc_z, env_z], training=False)
        fc_reconstruction = self.fc_decoder([z, y_pred, y_sex_pred], training=False)
        env_reconstruction = self.env_decoder([z, y_pred, y_sex_pred], training=False)

        re_loss = tf.keras.losses.MeanSquaredError(reduction="sum_over_batch_size")
        fc_reconstruction_loss = re_loss(fc, fc_reconstruction)
        env_reconstruction_loss = re_loss(env, env_reconstruction)

        clf_loss = self.loss_sup(y, y_pred)
        clf_sex_loss = self.loss_sup_sex(y_sex_true, y_sex_pred)

        xcov_loss = _xcov_loss(z, y_pred, self.n_batch_size)
        xcov_sex_loss = _xcov_loss(z, y_sex_pred, self.n_batch_size)
        total_loss = self.alpha * (fc_reconstruction_loss + env_reconstruction_loss) + self.beta* (clf_loss + clf_sex_loss) +  self.gamma * (xcov_loss + xcov_sex_loss)

        self.total_loss_tracker.update_state(total_loss)
        self.fc_reconstruction_loss_tracker.update_state(fc_reconstruction_loss)
        self.env_reconstruction_loss_tracker.update_state(env_reconstruction_loss)
        self.xcov_loss_tracker.update_state(xcov_loss)
        self.xcov_sex_loss_tracker.update_state(xcov_sex_loss)
        self.clf_loss_tracker.update_state(clf_loss)
        self.clf_sex_loss_tracker.update_state(clf_sex_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "fc_reconstruction_loss": self.fc_reconstruction_loss_tracker.result(),
            "env_reconstruction_loss": self.env_reconstruction_loss_tracker.result(),
            "xcov_loss": self.xcov_loss_tracker.result(),
            "xcov_sex_loss": self.xcov_sex_loss_tracker.result(),
            "clf_loss": self.clf_loss_tracker.result(),
            "clf_sex_loss": self.clf_sex_loss_tracker.result(),
        }



def main():
    stacked_fc_matrices, covariates, labels, envir, bs_with_nan = _load_data()

    result_columns = [
        "epochs",
        "loss",
        "fc_reconstruction_loss",
        "env_reconstruction_loss",
        "xcov_loss",
        "xcov_sex_loss",
        "clf_loss",
        "clf_sex_loss",
        "val_loss",
        "val_fc_reconstruction_loss",
        "val_env_reconstruction_loss",
        "val_xcov_loss",
        "val_xcov_sex_loss",
        "val_clf_loss",
        "val_clf_sex_loss",
        "val_re_rmse",
        "train_sym_rmse",
        "val_sym_rmse",
        "z_site_auc_roc",
        "z_site_f1",
        "z_site_mlp_auc_roc",
        "z_site_mlp_f1",
        "fc_z_site_auc_roc",
        "fc_z_site_f1",
        "fc_z_site_mlp_auc_roc",
        "fc_z_site_mlp_f1",
        "env_z_site_auc_roc",
        "env_z_site_f1",
        "env_z_site_mlp_auc_roc",
        "env_z_site_mlp_f1",
        "env_site_auc_roc",
        "env_site_f1",
        "env_site_mlp_auc_roc",
        "env_site_mlp_f1",
        "z_Sex_auc_roc",
        "z_Sex_f1",
        "z_Sex_mlp_auc_roc",
        "z_Sex_mlp_f1",
        "fc_z_Sex_auc_roc",
        "fc_z_Sex_f1",
        "fc_z_Sex_mlp_auc_roc",
        "fc_z_Sex_mlp_f1",
        "env_z_Sex_auc_roc",
        "env_z_Sex_f1",
        "env_z_Sex_mlp_auc_roc",
        "env_z_Sex_mlp_f1",
        "env_Sex_auc_roc",
        "env_Sex_f1",
        "env_Sex_mlp_auc_roc",
        "env_Sex_mlp_f1",
        "z_Diag_auc_roc",
        "z_Diag_f1",
        "z_Diag_mlp_auc_roc",
        "z_Diag_mlp_f1",
        "fc_z_Diag_auc_roc",
        "fc_z_Diag_f1",
        "fc_z_Diag_mlp_auc_roc",
        "fc_z_Diag_mlp_f1",
        "env_z_Diag_auc_roc",
        "env_z_Diag_f1",
        "env_z_Diag_mlp_auc_roc",
        "env_z_Diag_mlp_f1",
        "env_Diag_auc_roc",
        "env_Diag_f1",
        "env_Diag_mlp_auc_roc",
        "env_Diag_mlp_f1",
        "z_P_Diag_auc_roc",
        "z_P_Diag_f1",
        "z_P_Diag_mlp_auc_roc",
        "z_P_Diag_mlp_f1",
        "fc_z_P_Diag_auc_roc",
        "fc_z_P_Diag_f1",
        "fc_z_P_Diag_mlp_auc_roc",
        "fc_z_P_Diag_mlp_f1",
        "env_z_P_Diag_auc_roc",
        "env_z_P_Diag_f1",
        "env_z_P_Diag_mlp_auc_roc",
        "env_z_P_Diag_mlp_f1",
        "env_P_Diag_auc_roc",
        "env_P_Diag_f1",
        "env_P_Diag_mlp_auc_roc",
        "env_P_Diag_mlp_f1",
        "z_Age_rmse",
        "fc_z_Age_rmse",
        "env_z_Age_rmse",
        "z_Age_mlp_rmse",
        "fc_z_Age_mlp_rmse",
        "env_z_Age_mlp_rmse",
        "z_ERS_rmse",
        "softmax_site_auc_roc",
        "softmax_site_f1",
        # Cross-site generalizability metrics (site 1 as test, rest as train)
        "z_Sex_xsite_auc_roc",
        "z_Sex_xsite_f1",
        "z_Sex_xsite_mlp_auc_roc",
        "z_Sex_xsite_mlp_f1",
        "fc_z_Sex_xsite_auc_roc",
        "fc_z_Sex_xsite_f1",
        "fc_z_Sex_xsite_mlp_auc_roc",
        "fc_z_Sex_xsite_mlp_f1",
        "z_Diag_xsite_auc_roc",
        "z_Diag_xsite_f1",
        "z_Diag_xsite_mlp_auc_roc",
        "z_Diag_xsite_mlp_f1",
        "fc_z_Diag_xsite_auc_roc",
        "fc_z_Diag_xsite_f1",
        "fc_z_Diag_xsite_mlp_auc_roc",
        "fc_z_Diag_xsite_mlp_f1",
        "z_P_Diag_xsite_auc_roc",
        "z_P_Diag_xsite_f1",
        "z_P_Diag_xsite_mlp_auc_roc",
        "z_P_Diag_xsite_mlp_f1",
        "fc_z_P_Diag_xsite_auc_roc",
        "fc_z_P_Diag_xsite_f1",
        "fc_z_P_Diag_xsite_mlp_auc_roc",
        "fc_z_P_Diag_xsite_mlp_f1",
        "z_Age_xsite_rmse",
        "fc_z_Age_xsite_rmse",
        "z_Age_xsite_mlp_rmse",
        "fc_z_Age_xsite_mlp_rmse",
    ]

    fold_records = []
    convergence_rows = []

    for r in range(2):
        skf = StratifiedKFold(n_splits=5, random_state=random_state[r], shuffle=True)
        print(r)

        for i, (train_ind, val_ind) in enumerate(skf.split(stacked_fc_matrices, labels.iloc[:, 0].values)):
            fold_idx = start[r] + i
            print(fold_idx)

            X_train = stacked_fc_matrices[train_ind, :, :]
            X_val = stacked_fc_matrices[val_ind, :, :]
            cov_train = covariates[train_ind, :]
            cov_val = covariates[val_ind, :]
            env_train = envir[train_ind, :].copy()
            env_val = envir[val_ind, :].copy()
            labels_train = labels.iloc[train_ind, :]
            labels_val = labels.iloc[val_ind, :]
            bs_train = bs_with_nan[train_ind, :].copy()
            bs_val = bs_with_nan[val_ind, :].copy()

            col_mean = np.nanmean(bs_train, axis=0)
            inds = np.where(np.isnan(bs_train))
            env_train[:, 10:][inds] = np.take(col_mean, inds[1])
            inds = np.where(np.isnan(bs_val))
            env_val[:, 10:][inds] = np.take(col_mean, inds[1])

            max_score_per_subject = 6 * 9 + 4 * 13
            train_env_risk = np.sum(env_train, axis=1) / max_score_per_subject
            val_env_risk = np.sum(env_val, axis=1) / max_score_per_subject

            y_train_multiclass = cov_train[:, 3:].copy()
            y_val_multiclass = cov_val[:, 3:].copy()
            num_classes = y_train_multiclass.shape[1]
            num_classes_sex = 2

            # Sex labels (binary) - one-hot encoded
            y_train_sex_raw = cov_train[:, 1].astype('float32')
            y_val_sex_raw = cov_val[:, 1].astype('float32')
            y_train_sex_multiclass = tf.keras.utils.to_categorical(y_train_sex_raw.astype(int), num_classes=2)
            y_val_sex_multiclass = tf.keras.utils.to_categorical(y_val_sex_raw.astype(int), num_classes=2)
            num_classes_sex = 2

            np.random.seed(seed_value)
            tf.random.set_seed(seed_value)

            FC_encoder, ENV_encoder, fusion, fc_decoder, env_decoder, loss_sup, loss_sup_sex = _build_models(num_classes, num_classes_sex)

            steps_per_epoch = max(1, int(X_train.shape[0] / batch_size))
            lrs = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                steps_per_epoch,
                learning_rate_decay_factor,
                staircase=False,
            )

            autoencoder_2 = AE(
                FC_encoder,
                fc_decoder,
                ENV_encoder,
                env_decoder,
                fusion,
                loss_sup,
                loss_sup_sex,
                batch_size,
                alpha=alpha,
                gamma=gamma_xcov,
            )

            autoencoder_2.compile(
                optimizer_ae=tf.keras.optimizers.Adam(learning_rate=0.0002),
                jit_compile=False,
            )

            autoencoder_2.fit(
                x=[X_train, env_train],
                y=[y_train_multiclass, y_train_sex_multiclass],
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=([X_val, env_val], [y_val_multiclass, y_val_sex_multiclass]),
                verbose=1,
            )

            train_loss = autoencoder_2.history.history["loss"]
            val_loss = autoencoder_2.history.history["val_loss"]
            train_fc_loss = autoencoder_2.history.history["fc_reconstruction_loss"]
            val_fc_loss = autoencoder_2.history.history["val_fc_reconstruction_loss"]
            train_env_loss = autoencoder_2.history.history["env_reconstruction_loss"]
            val_env_loss = autoencoder_2.history.history["val_env_reconstruction_loss"]
            train_xcov_loss = autoencoder_2.history.history["xcov_loss"]
            val_xcov_loss = autoencoder_2.history.history["val_xcov_loss"]
            train_xcov_sex_loss = autoencoder_2.history.history["xcov_sex_loss"]
            val_xcov_sex_loss = autoencoder_2.history.history["val_xcov_sex_loss"]
            train_clf_loss = autoencoder_2.history.history["clf_loss"]
            val_clf_loss = autoencoder_2.history.history["val_clf_loss"]
            train_clf_sex_loss = autoencoder_2.history.history["clf_sex_loss"]
            val_clf_sex_loss = autoencoder_2.history.history["val_clf_sex_loss"]

            # Save loss evolution plot for first fold only
            if fold_idx == start[0]:
                epochs_range = range(1, len(train_loss) + 1)
                fig, axes = plt.subplots(7, 1, figsize=(12, 20))
                
                axes[0].plot(epochs_range, train_loss, "b-", label="Train", linewidth=1.5)
                axes[0].plot(epochs_range, val_loss, "r-", label="Validation", linewidth=1.5)
                axes[0].set_ylabel("Total Loss")
                axes[0].set_title(f"Total Loss - Fold {fold_idx}")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                axes[1].plot(epochs_range, train_fc_loss, "b-", label="Train", linewidth=1.5)
                axes[1].plot(epochs_range, val_fc_loss, "r-", label="Validation", linewidth=1.5)
                axes[1].set_ylabel("FC Reconstruction Loss")
                axes[1].set_title(f"FC Reconstruction Loss - Fold {fold_idx}")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                axes[2].plot(epochs_range, train_env_loss, "b-", label="Train", linewidth=1.5)
                axes[2].plot(epochs_range, val_env_loss, "r-", label="Validation", linewidth=1.5)
                axes[2].set_ylabel("ENV Reconstruction Loss")
                axes[2].set_title(f"ENV Reconstruction Loss - Fold {fold_idx}")
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                axes[3].plot(epochs_range, train_xcov_loss, "b-", label="Train", linewidth=1.5)
                axes[3].plot(epochs_range, val_xcov_loss, "r-", label="Validation", linewidth=1.5)
                axes[3].set_ylabel("XCov Loss (Site)")
                axes[3].set_title(f"XCov Loss (Site) - Fold {fold_idx}")
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)
                
                axes[4].plot(epochs_range, train_xcov_sex_loss, "b-", label="Train", linewidth=1.5)
                axes[4].plot(epochs_range, val_xcov_sex_loss, "r-", label="Validation", linewidth=1.5)
                axes[4].set_ylabel("XCov Loss (Sex)")
                axes[4].set_title(f"XCov Loss (Sex) - Fold {fold_idx}")
                axes[4].legend()
                axes[4].grid(True, alpha=0.3)
                
                axes[5].plot(epochs_range, train_clf_loss, "b-", label="Train", linewidth=1.5)
                axes[5].plot(epochs_range, val_clf_loss, "r-", label="Validation", linewidth=1.5)
                axes[5].set_ylabel("Classification Loss (Site)")
                axes[5].set_title(f"Classification Loss (Site) - Fold {fold_idx}")
                axes[5].legend()
                axes[5].grid(True, alpha=0.3)
                
                axes[6].plot(epochs_range, train_clf_sex_loss, "b-", label="Train", linewidth=1.5)
                axes[6].plot(epochs_range, val_clf_sex_loss, "r-", label="Validation", linewidth=1.5)
                axes[6].set_xlabel("Epoch")
                axes[6].set_ylabel("Classification Loss (Sex)")
                axes[6].set_title(f"Classification Loss (Sex) - Fold {fold_idx}")
                axes[6].legend()
                axes[6].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = os.path.join(save_results, f"{output_base_name}_losses.png")
                plt.savefig(plot_path, dpi=100)
                plt.close()
                print(f"Saved loss plot: {plot_path}")


            fc_train_encoded = FC_encoder.predict(X_train, verbose=0)
            env_train_encoded = ENV_encoder.predict(env_train, verbose=0)
            z_train, h1_train, y_train_sites_pred, y_train_sex_pred = fusion.predict([fc_train_encoded, env_train_encoded], verbose=0)

            fc_train_decoded = fc_decoder.predict([z_train, y_train_sites_pred, y_train_sex_pred], verbose=0)

            fc_val_encoded  = FC_encoder.predict(X_val, verbose=0)
            env_val_encoded = ENV_encoder.predict(env_val, verbose=0)
            z_val, h1_val, y_val_sites_pred, y_val_sex_pred = fusion.predict([fc_val_encoded, env_val_encoded], verbose=0)
            
            fc_val_decoded = fc_decoder.predict([z_val, y_val_sites_pred, y_val_sex_pred], verbose=0)

            y_train_sites = pd.DataFrame(y_train_multiclass).idxmax(axis=1).values
            y_val_sites = pd.DataFrame(y_val_multiclass).idxmax(axis=1).values
            sites = np.unique(y_train_sites)

            fold_result = {column: np.nan for column in result_columns}
            fold_result["epochs"] = len(train_loss)
            fold_result["loss"] = train_loss[-1]
            fold_result["fc_reconstruction_loss"] = train_fc_loss[-1]
            fold_result["env_reconstruction_loss"] = train_env_loss[-1]
            fold_result["xcov_loss"] = train_xcov_loss[-1]
            fold_result["xcov_sex_loss"] = train_xcov_sex_loss[-1]
            fold_result["clf_loss"] = train_clf_loss[-1]
            fold_result["clf_sex_loss"] = train_clf_sex_loss[-1]
            fold_result["val_loss"] = val_loss[-1]
            fold_result["val_fc_reconstruction_loss"] = val_fc_loss[-1]
            fold_result["val_env_reconstruction_loss"] = val_env_loss[-1]
            fold_result["val_xcov_loss"] = val_xcov_loss[-1]
            fold_result["val_xcov_sex_loss"] = val_xcov_sex_loss[-1]
            fold_result["val_clf_loss"] = val_clf_loss[-1]
            fold_result["val_clf_sex_loss"] = val_clf_sex_loss[-1]

            X_val_flatten = X_val.reshape(X_val.shape[0], -1)
            fold_result["val_re_rmse"] = mean_squared_error(X_val_flatten, fc_val_decoded.reshape(X_val.shape[0], -1), squared=False)

            tril = np.tril(fc_train_decoded[:, :, :, 0])
            triu = np.triu(fc_train_decoded[:, :, :, 0])
            fold_result["train_sym_rmse"] = mean_squared_error(triu[0, :, :].T, tril[0, :, :], squared=False)
            tril = np.tril(fc_val_decoded[:, :, :, 0])
            triu = np.triu(fc_val_decoded[:, :, :, 0])
            fold_result["val_sym_rmse"] = mean_squared_error(triu[0, :, :].T, tril[0, :, :], squared=False)

            try:
                auc, f1 = _safe_auc_f1_multiclass(y_val_multiclass, y_val_sites_pred, sites)
                fold_result["softmax_site_auc_roc"] = auc
                fold_result["softmax_site_f1"] = f1
            except Exception as e:
                print(f"ERROR in softmax site classification (fold {fold_idx}): {e}")
                traceback.print_exc()

            # Site classification
            for prefix, Xtr, Xva in [
                ("z", z_train, z_val),
                ("fc_z", fc_train_encoded, fc_val_encoded),
                ("env_z", env_train_encoded, env_val_encoded),
                ("env", env_train, env_val),
            ]:
                try:
                    auc, f1 = _fit_voting_classifier(Xtr, y_train_sites, Xva, y_val_sites, average_mode="multiclass", labels_for_f1=sites)
                    fold_result[f"{prefix}_site_auc_roc"] = auc
                    fold_result[f"{prefix}_site_f1"] = f1
                except Exception as e:
                    print(f"ERROR in {prefix} voting site classification (fold {fold_idx}): {e}")
                    traceback.print_exc()
                try:
                    _, y_prob, _ = _fit_mlp_classifier(
                        Xtr,
                        y_train_sites,
                        Xva,
                        f"{prefix}_site_mlp",
                        fold_idx,
                        convergence_rows,
                        hidden_layers=(128,),
                        max_iter=300,
                    )
                    auc, f1 = _safe_auc_f1_multiclass(y_val_multiclass, y_prob, sites)
                    fold_result[f"{prefix}_site_mlp_auc_roc"] = auc
                    fold_result[f"{prefix}_site_mlp_f1"] = f1
                except Exception as e:
                    print(f"ERROR in {prefix} MLP site classification (fold {fold_idx}): {e}")
                    traceback.print_exc()

            # Sex classification
            y_train_binary = cov_train[:, 1].astype("float32").reshape((-1, 1))
            y_val_binary = cov_val[:, 1].astype("float32").reshape((-1, 1))
            for prefix, Xtr, Xva in [
                ("z", z_train, z_val),
                ("fc_z", fc_train_encoded, fc_val_encoded),
                ("env_z", env_train_encoded, env_val_encoded),
                ("env", env_train, env_val),
            ]:
                try:
                    auc, f1 = _fit_voting_classifier(Xtr, y_train_binary.ravel(), Xva, y_val_binary.ravel())
                    fold_result[f"{prefix}_Sex_auc_roc"] = auc
                    fold_result[f"{prefix}_Sex_f1"] = f1
                except Exception as e:
                    print(f"ERROR in {prefix} voting Sex classification (fold {fold_idx}): {e}")
                    traceback.print_exc()
                try:
                    _, y_prob, _ = _fit_mlp_classifier(
                        Xtr,
                        y_train_binary.ravel(),
                        Xva,
                        f"{prefix}_Sex_mlp",
                        fold_idx,
                        convergence_rows,
                        hidden_layers=(64,),
                        max_iter=300,
                    )
                    auc, f1 = _safe_auc_f1_binary(y_val_binary, y_prob)
                    fold_result[f"{prefix}_Sex_mlp_auc_roc"] = auc
                    fold_result[f"{prefix}_Sex_mlp_f1"] = f1
                except Exception as e:
                    print(f"ERROR in {prefix} MLP Sex classification (fold {fold_idx}): {e}")
                    traceback.print_exc()

            # HC vs P
            y_train_binary = np.abs(labels_train.iloc[:, 0].values.ravel().astype("float32") - 1).reshape((-1, 1))
            y_val_binary = np.abs(labels_val.iloc[:, 0].values.ravel().astype("float32") - 1).reshape((-1, 1))
            for prefix, Xtr, Xva in [
                ("z", z_train, z_val),
                ("fc_z", fc_train_encoded, fc_val_encoded),
                ("env_z", env_train_encoded, env_val_encoded),
                ("env", env_train, env_val),
            ]:
                try:
                    auc, f1 = _fit_voting_classifier(Xtr, y_train_binary.ravel(), Xva, y_val_binary.ravel())
                    fold_result[f"{prefix}_Diag_auc_roc"] = auc
                    fold_result[f"{prefix}_Diag_f1"] = f1
                except Exception:
                    pass
                try:
                    _, y_prob, _ = _fit_mlp_classifier(
                        Xtr,
                        y_train_binary.ravel(),
                        Xva,
                        f"{prefix}_Diag_mlp",
                        fold_idx,
                        convergence_rows,
                        hidden_layers=(64,),
                        max_iter=300,
                    )
                    auc, f1 = _safe_auc_f1_binary(y_val_binary, y_prob)
                    fold_result[f"{prefix}_Diag_mlp_auc_roc"] = auc
                    fold_result[f"{prefix}_Diag_mlp_f1"] = f1
                except Exception as e:
                    print(f"ERROR in {prefix} MLP Diag classification (fold {fold_idx}): {e}")
                    traceback.print_exc()

            # ROP vs ROD
            train_ind_rop = np.where(labels_train.loc[:, 3].values == 1)[0]
            train_ind_rod = np.where(labels_train.loc[:, 2].values == 1)[0]
            train_ind_p = list(train_ind_rop) + list(train_ind_rod)
            y_train_binary = np.abs(labels_train.iloc[train_ind_p, 3].values.ravel().astype("float32") - 1).reshape((-1, 1))
            val_ind_rop = np.where(labels_val.loc[:, 3].values == 1)[0]
            val_ind_rod = np.where(labels_val.loc[:, 2].values == 1)[0]
            val_ind_p = list(val_ind_rop) + list(val_ind_rod)
            y_val_binary = np.abs(labels_val.iloc[val_ind_p, 3].values.ravel().astype("float32") - 1).reshape((-1, 1))

            for prefix, Xtr, Xva in [
                ("z", z_train[train_ind_p, :], z_val[val_ind_p, :]),
                ("fc_z", fc_train_encoded[train_ind_p, :], fc_val_encoded[val_ind_p, :]),
                ("env_z", env_train_encoded[train_ind_p, :], env_val_encoded[val_ind_p, :]),
                ("env", env_train[train_ind_p, :], env_val[val_ind_p, :]),
            ]:
                try:
                    auc, f1 = _fit_voting_classifier(Xtr, y_train_binary.ravel(), Xva, y_val_binary.ravel())
                    fold_result[f"{prefix}_P_Diag_auc_roc"] = auc
                    fold_result[f"{prefix}_P_Diag_f1"] = f1
                except Exception as e:
                    print(f"ERROR in {prefix} voting P_Diag classification (fold {fold_idx}): {e}")
                    traceback.print_exc()
                try:
                    _, y_prob, _ = _fit_mlp_classifier(
                        Xtr,
                        y_train_binary.ravel(),
                        Xva,
                        f"{prefix}_P_Diag_mlp",
                        fold_idx,
                        convergence_rows,
                        hidden_layers=(64,),
                        max_iter=300,
                    )
                    auc, f1 = _safe_auc_f1_binary(y_val_binary, y_prob)
                    fold_result[f"{prefix}_P_Diag_mlp_auc_roc"] = auc
                    fold_result[f"{prefix}_P_Diag_mlp_f1"] = f1
                except Exception as e:
                    print(f"ERROR in {prefix} MLP P_Diag classification (fold {fold_idx}): {e}")
                    traceback.print_exc()

            # Age regression
            for prefix, Xtr, Xva in [
                ("z", z_train, z_val),
                ("fc_z", fc_train_encoded, fc_val_encoded),
                ("env_z", env_train_encoded, env_val_encoded),
            ]:
                try:
                    knn = KNeighborsRegressor(n_neighbors=3)
                    svr = SVR()
                    dt = DecisionTreeRegressor(random_state=0)
                    vc = VotingRegressor(estimators=[("knn", knn), ("svm", svr), ("dt", dt)])
                    vc.fit(Xtr, cov_train[:, 0])
                    y_val_pred = vc.predict(Xva)
                    fold_result[f"{prefix}_Age_rmse"] = mean_squared_error(cov_val[:, 0].ravel(), y_val_pred, squared=False)
                except Exception as e:
                    print(f"ERROR in {prefix} voting Age regression (fold {fold_idx}): {e}")
                    traceback.print_exc()
                try:
                    _, y_pred, _ = _fit_mlp_regressor(
                        Xtr,
                        cov_train[:, 0],
                        Xva,
                        f"{prefix}_Age_mlp",
                        fold_idx,
                        convergence_rows,
                        hidden_layers=(128, 64),
                        max_iter=500,
                    )
                    fold_result[f"{prefix}_Age_mlp_rmse"] = mean_squared_error(cov_val[:, 0].ravel(), y_pred, squared=False)
                except Exception as e:
                    print(f"ERROR in {prefix} MLP Age regression (fold {fold_idx}): {e}")
                    traceback.print_exc()

            # ERS regression
            try:
                knn = KNeighborsRegressor(n_neighbors=3)
                svr = SVR()
                dt = DecisionTreeRegressor(random_state=0)
                vc = VotingRegressor(estimators=[("knn", knn), ("svm", svr), ("dt", dt)])
                vc.fit(z_train, train_env_risk)
                y_val_pred = vc.predict(z_val)
                fold_result["z_ERS_rmse"] = mean_squared_error(val_env_risk.ravel(), y_val_pred, squared=False)
            except Exception as e:
                print(f"ERROR in z ERS regression (fold {fold_idx}): {e}")
                traceback.print_exc()

            # ============================================================================
            # Cross-site generalizability evaluation (site 1 as test, rest as train)
            # ============================================================================
            try:
                # Identify site 1 (biggest site) in the combined train+val set
                y_sites_combined = np.concatenate([y_train_sites, y_val_sites])
                z_combined = np.concatenate([z_train, z_val], axis=0)
                fc_z_combined = np.concatenate([fc_train_encoded, fc_val_encoded], axis=0)
                cov_combined = np.concatenate([cov_train, cov_val], axis=0)
                labels_combined = pd.concat([labels_train, labels_val], ignore_index=True)

                # Unique sites and their counts
                unique_sites, site_counts = np.unique(y_sites_combined, return_counts=True)
                biggest_site = unique_sites[np.argmax(site_counts)]

                # Split: biggest site as test, rest as train
                xsite_test_mask = y_sites_combined == biggest_site
                xsite_train_mask = ~xsite_test_mask

                z_xsite_train = z_combined[xsite_train_mask, :]
                z_xsite_test = z_combined[xsite_test_mask, :]
                fc_z_xsite_train = fc_z_combined[xsite_train_mask, :]
                fc_z_xsite_test = fc_z_combined[xsite_test_mask, :]
                cov_xsite_train = cov_combined[xsite_train_mask, :]
                cov_xsite_test = cov_combined[xsite_test_mask, :]
                labels_xsite_train = labels_combined[xsite_train_mask]
                labels_xsite_test = labels_combined[xsite_test_mask]

                # Sex classification (cross-site)
                y_sex_xsite_train = cov_xsite_train[:, 1].astype("float32").ravel()
                y_sex_xsite_test = cov_xsite_test[:, 1].astype("float32").ravel()

                for prefix, Xtr, Xte in [
                    ("z", z_xsite_train, z_xsite_test),
                    ("fc_z", fc_z_xsite_train, fc_z_xsite_test),
                ]:
                    try:
                        auc, f1 = _fit_voting_classifier(Xtr, y_sex_xsite_train, Xte, y_sex_xsite_test)
                        fold_result[f"{prefix}_Sex_xsite_auc_roc"] = auc
                        fold_result[f"{prefix}_Sex_xsite_f1"] = f1
                    except Exception as e:
                        print(f"ERROR in {prefix} voting Sex xsite classification (fold {fold_idx}): {e}")
                        traceback.print_exc()
                    try:
                        _, y_prob, _ = _fit_mlp_classifier(
                            Xtr, y_sex_xsite_train, Xte, f"{prefix}_Sex_xsite_mlp", fold_idx,
                            convergence_rows, hidden_layers=(64,), max_iter=300,
                        )
                        y_sex_xsite_test_binary = y_sex_xsite_test.reshape((-1, 1))
                        auc, f1 = _safe_auc_f1_binary(y_sex_xsite_test_binary, y_prob)
                        fold_result[f"{prefix}_Sex_xsite_mlp_auc_roc"] = auc
                        fold_result[f"{prefix}_Sex_xsite_mlp_f1"] = f1
                    except Exception as e:
                        print(f"ERROR in {prefix} MLP Sex xsite classification (fold {fold_idx}): {e}")
                        traceback.print_exc()

                # HC vs P (Diag) classification (cross-site)
                y_diag_xsite_train = np.abs(labels_xsite_train.iloc[:, 0].values.ravel().astype("float32") - 1)
                y_diag_xsite_test = np.abs(labels_xsite_test.iloc[:, 0].values.ravel().astype("float32") - 1)

                for prefix, Xtr, Xte in [
                    ("z", z_xsite_train, z_xsite_test),
                    ("fc_z", fc_z_xsite_train, fc_z_xsite_test),
                ]:
                    try:
                        auc, f1 = _fit_voting_classifier(Xtr, y_diag_xsite_train, Xte, y_diag_xsite_test)
                        fold_result[f"{prefix}_Diag_xsite_auc_roc"] = auc
                        fold_result[f"{prefix}_Diag_xsite_f1"] = f1
                    except Exception as e:
                        print(f"ERROR in {prefix} voting Diag xsite classification (fold {fold_idx}): {e}")
                        traceback.print_exc()
                    try:
                        _, y_prob, _ = _fit_mlp_classifier(
                            Xtr, y_diag_xsite_train, Xte, f"{prefix}_Diag_xsite_mlp", fold_idx,
                            convergence_rows, hidden_layers=(64,), max_iter=300,
                        )
                        y_diag_xsite_test_binary = y_diag_xsite_test.reshape((-1, 1))
                        auc, f1 = _safe_auc_f1_binary(y_diag_xsite_test_binary, y_prob)
                        fold_result[f"{prefix}_Diag_xsite_mlp_auc_roc"] = auc
                        fold_result[f"{prefix}_Diag_xsite_mlp_f1"] = f1
                    except Exception as e:
                        print(f"ERROR in {prefix} MLP Diag xsite classification (fold {fold_idx}): {e}")
                        traceback.print_exc()

                # ROP vs ROD (P_Diag) classification (cross-site)
                train_ind_rop_xsite = np.where(labels_xsite_train.loc[:, 3].values == 1)[0]
                train_ind_rod_xsite = np.where(labels_xsite_train.loc[:, 2].values == 1)[0]
                train_ind_p_xsite = list(train_ind_rop_xsite) + list(train_ind_rod_xsite)
                
                test_ind_rop_xsite = np.where(labels_xsite_test.loc[:, 3].values == 1)[0]
                test_ind_rod_xsite = np.where(labels_xsite_test.loc[:, 2].values == 1)[0]
                test_ind_p_xsite = list(test_ind_rop_xsite) + list(test_ind_rod_xsite)

                if len(train_ind_p_xsite) > 0 and len(test_ind_p_xsite) > 0:
                    y_pdiag_xsite_train = np.abs(labels_xsite_train.iloc[train_ind_p_xsite, 3].values.ravel().astype("float32") - 1)
                    y_pdiag_xsite_test = np.abs(labels_xsite_test.iloc[test_ind_p_xsite, 3].values.ravel().astype("float32") - 1)

                    for prefix, Xtr, Xte in [
                        ("z", z_xsite_train[train_ind_p_xsite, :], z_xsite_test[test_ind_p_xsite, :]),
                        ("fc_z", fc_z_xsite_train[train_ind_p_xsite, :], fc_z_xsite_test[test_ind_p_xsite, :]),
                    ]:
                        try:
                            auc, f1 = _fit_voting_classifier(Xtr, y_pdiag_xsite_train, Xte, y_pdiag_xsite_test)
                            fold_result[f"{prefix}_P_Diag_xsite_auc_roc"] = auc
                            fold_result[f"{prefix}_P_Diag_xsite_f1"] = f1
                        except Exception as e:
                            print(f"ERROR in {prefix} voting P_Diag xsite classification (fold {fold_idx}): {e}")
                            traceback.print_exc()
                        try:
                            _, y_prob, _ = _fit_mlp_classifier(
                                Xtr, y_pdiag_xsite_train, Xte, f"{prefix}_P_Diag_xsite_mlp", fold_idx,
                                convergence_rows, hidden_layers=(64,), max_iter=300,
                            )
                            y_pdiag_xsite_test_binary = y_pdiag_xsite_test.reshape((-1, 1))
                            auc, f1 = _safe_auc_f1_binary(y_pdiag_xsite_test_binary, y_prob)
                            fold_result[f"{prefix}_P_Diag_xsite_mlp_auc_roc"] = auc
                            fold_result[f"{prefix}_P_Diag_xsite_mlp_f1"] = f1
                        except Exception as e:
                            print(f"ERROR in {prefix} MLP P_Diag xsite classification (fold {fold_idx}): {e}")
                            traceback.print_exc()

                # Age regression (cross-site)
                for prefix, Xtr, Xte in [
                    ("z", z_xsite_train, z_xsite_test),
                    ("fc_z", fc_z_xsite_train, fc_z_xsite_test),
                ]:
                    try:
                        knn = KNeighborsRegressor(n_neighbors=3)
                        svr = SVR()
                        dt = DecisionTreeRegressor(random_state=0)
                        vc = VotingRegressor(estimators=[("knn", knn), ("svm", svr), ("dt", dt)])
                        vc.fit(Xtr, cov_xsite_train[:, 0])
                        y_pred_xsite = vc.predict(Xte)
                        fold_result[f"{prefix}_Age_xsite_rmse"] = mean_squared_error(cov_xsite_test[:, 0].ravel(), y_pred_xsite, squared=False)
                    except Exception as e:
                        print(f"ERROR in {prefix} voting Age xsite regression (fold {fold_idx}): {e}")
                        traceback.print_exc()
                    try:
                        _, y_pred_xsite, _ = _fit_mlp_regressor(
                            Xtr, cov_xsite_train[:, 0], Xte, f"{prefix}_Age_xsite_mlp", fold_idx,
                            convergence_rows, hidden_layers=(128, 64), max_iter=500,
                        )
                        fold_result[f"{prefix}_Age_xsite_mlp_rmse"] = mean_squared_error(cov_xsite_test[:, 0].ravel(), y_pred_xsite, squared=False)
                    except Exception as e:
                        print(f"ERROR in {prefix} MLP Age xsite regression (fold {fold_idx}): {e}")
                        traceback.print_exc()

            except Exception as e:
                print(f"ERROR in cross-site generalizability evaluation (fold {fold_idx}): {e}")
                traceback.print_exc()

            fold_records.append(fold_result)

    fold_results = pd.DataFrame(fold_records, columns=result_columns)

    if len(fold_results) >= 10:
        mean_row = fold_results.iloc[:10, :].mean(axis=0)
        fold_results.loc[len(fold_results)] = mean_row

    fold_results.index = np.arange(1, len(fold_results) + 1)
    fold_results.T.to_csv(os.path.join(save_results, f"{output_base_name}.csv"))

    mlp_cols = [col for col in fold_results.columns if "_mlp_" in col]
    if mlp_cols:
        fold_results.loc[:, mlp_cols].T.to_csv(os.path.join(save_results, f"{output_base_name}_mlp.csv"))

    convergence_df = pd.DataFrame(convergence_rows)
    convergence_df.to_csv(os.path.join(save_results, f"{output_base_name}_convergence.csv"), index=False)

    print("Saved results to:")
    print(os.path.join(save_results, f"{output_base_name}.csv"))
    print(os.path.join(save_results, f"{output_base_name}_mlp.csv"))
    print(os.path.join(save_results, f"{output_base_name}_convergence.csv"))


if __name__ == "__main__":
    main()
