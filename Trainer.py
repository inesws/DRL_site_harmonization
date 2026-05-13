import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv1D,
    Conv1DTranspose,
    Conv2D,
    Cropping1D,
    Dense,
    Dropout,
    Input,
    MaxPooling1D,
    MaxPooling2D,
    Reshape,
    UpSampling1D,
    UpSampling2D,
)
from keras.models import Model
from keras.regularizers import l2
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class VariableSpec:
    name: str
    kind: str


@dataclass
class TrainerConfig:
    x_all: np.ndarray
    covariates: np.ndarray
    labels: pd.DataFrame
    envir: np.ndarray
    bs_with_nan: np.ndarray
    save_results_dir: str
    results_filename: str = "cv_results.csv"
    target_variable: str = "site"
    eval_specs: Tuple[VariableSpec, ...] = ()

    n_splits: int = 5
    random_states: Tuple[int, ...] = (42, 24)

    penalty: str = "xcov"
    gamma: float = 1.0
    alpha_fc: float = 1.0
    alpha_env: float = 1.0
    hsic_sigma: float = 1.0

    batch_size: int = 128
    epochs: int = 2000
    ae_learning_rate: float = 2e-4
    clf_initial_lr: float = 1e-4
    clf_final_lr: float = 1e-5

    seed_value: int = 2020
    use_early_stopping: bool = False
    early_stopping_patience: int = 250


class DRLTrainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self._set_seed()

    def _set_seed(self) -> None:
        np.random.seed(self.cfg.seed_value)
        tf.random.set_seed(self.cfg.seed_value)

    @staticmethod
    def _flatten_labels(values: np.ndarray) -> np.ndarray:
        if values.ndim == 2 and values.shape[1] > 1:
            return np.argmax(values, axis=1)
        return values.reshape(-1)

    @staticmethod
    def _format_target(values: np.ndarray, variable_name: str) -> np.ndarray:
        if variable_name in {"site"}:
            return values.astype(np.float32)

        if values.ndim == 1:
            return values.reshape(-1, 1).astype(np.float32)

        return values.astype(np.float32)

    def _extract_variable(
        self,
        variable_name: str,
        covariates: np.ndarray,
        labels: pd.DataFrame,
        envir: np.ndarray,
    ) -> np.ndarray:
        name = variable_name.lower()

        if name == "site":
            return covariates[:, 3:].astype(np.float32)
        if name == "sex":
            return covariates[:, 1].astype(np.float32).reshape(-1, 1)
        if name == "age":
            return covariates[:, 0].astype(np.float32).reshape(-1, 1)
        if name == "diagnosis":
            return labels.iloc[:, 0].values.astype(np.float32).reshape(-1, 1)
        if name == "ers":
            max_score_per_subject = 6 * 9 + 4 * 13
            return (np.sum(envir, axis=1) / max_score_per_subject).astype(np.float32).reshape(-1, 1)

        raise ValueError(f"Unsupported variable name: {variable_name}")

    @staticmethod
    def _rbf_kernel(x: tf.Tensor, sigma: float) -> tf.Tensor:
        x2 = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        sq_dist = x2 - 2.0 * tf.matmul(x, x, transpose_b=True) + tf.transpose(x2)
        return tf.exp(-sq_dist / (2.0 * sigma * sigma + 1e-8))

    def _hsic_loss(self, z: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
        n = tf.shape(z)[0]
        n_f = tf.cast(n, tf.float32)
        h = tf.eye(n) - tf.ones((n, n), dtype=tf.float32) / n_f
        kz = self._rbf_kernel(z, self.cfg.hsic_sigma)
        kc = self._rbf_kernel(c, self.cfg.hsic_sigma)
        return tf.linalg.trace(tf.matmul(tf.matmul(kz, h), tf.matmul(kc, h))) / ((n_f - 1.0) ** 2 + 1e-8)

    def _mmd_loss(self, z: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_class = tf.argmax(y_pred, axis=1)
        classes = tf.unique(y_class)[0]

        def pair_loss(i, j):
            mask_i = tf.equal(y_class, i)
            mask_j = tf.equal(y_class, j)
            zi = tf.boolean_mask(z, mask_i)
            zj = tf.boolean_mask(z, mask_j)

            ni = tf.shape(zi)[0]
            nj = tf.shape(zj)[0]

            def compute_mmd():
                k_ii = self._rbf_kernel(zi, self.cfg.mmd_sigma)
                k_jj = self._rbf_kernel(zj, self.cfg.mmd_sigma)
                zcat = tf.concat([zi, zj], axis=0)
                k_cat = self._rbf_kernel(zcat, self.cfg.mmd_sigma)
                k_ij = k_cat[:ni, ni : ni + nj]
                return tf.reduce_mean(k_ii) + tf.reduce_mean(k_jj) - 2.0 * tf.reduce_mean(k_ij)

            return tf.cond(tf.logical_or(ni < 2, nj < 2), lambda: tf.constant(0.0, dtype=tf.float32), compute_mmd)

        total = tf.constant(0.0, dtype=tf.float32)
        pairs = tf.constant(0.0, dtype=tf.float32)

        c_len = tf.shape(classes)[0]
        for i in tf.range(c_len):
            for j in tf.range(i + 1, c_len):
                total += pair_loss(classes[i], classes[j])
                pairs += 1.0

        return tf.cond(pairs > 0.0, lambda: total / pairs, lambda: tf.constant(0.0, dtype=tf.float32))

    def _compute_drl_penalty(self, fc_z: tf.Tensor, y_pred: tf.Tensor, batch_n: tf.Tensor) -> tf.Tensor:
        z_mean = tf.reduce_mean(fc_z, axis=0, keepdims=True)
        y_mean = tf.reduce_mean(y_pred, axis=0, keepdims=True)

        z_flat = tf.keras.layers.Flatten()(fc_z - z_mean)
        y_centered = y_pred - y_mean

        penalty = self.cfg.penalty.lower()
        if penalty == "none":
            return tf.constant(0.0, dtype=tf.float32)

        if penalty == "xcov":
            return 0.5 * tf.reduce_sum(
                tf.square(tf.matmul(y_centered, z_flat, transpose_a=True) / (batch_n + 1e-8))
            )

        if penalty == "hsic":
            return self._hsic_loss(z_flat, y_centered)

        #if penalty == "mmd":
        #   return self._mmd_loss(z_flat, y_pred)

        raise ValueError(f"Unsupported penalty: {self.cfg.penalty}")

    def _build_models(self, num_classes: int):
        input_matrix = Input(shape=(160, 160, 1))
        x = Conv2D(
            16,
            (3, 3),
            data_format="channels_last",
            activation="selu",
            kernel_initializer="lecun_normal",
            padding="same",
            kernel_regularizer=l2(0.1),
        )(input_matrix)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(32, (3, 3), activation="selu", kernel_initializer="lecun_normal", padding="same")(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(64, (3, 3), activation="selu", kernel_initializer="lecun_normal", padding="same")(x)
        h1_layer = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(128, (3, 3), activation="selu", kernel_initializer="lecun_normal", padding="same")(h1_layer)
        fc_encoded = MaxPooling2D((4, 4), padding="same")(x)
        fc_flatten = tf.keras.layers.Flatten()(fc_encoded)

        if num_classes > 1:
            af = "softmax"
            loss_sup = tf.keras.losses.CategoricalCrossentropy()
        else:
            af = "sigmoid"
            loss_sup = tf.keras.losses.BinaryCrossentropy()

        x = tf.keras.layers.Flatten()(h1_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        supervised_output = Dense(num_classes, activation=af, kernel_regularizer=l2(0.01))(x)

        fc_encoder = Model(input_matrix, [fc_flatten, h1_layer, supervised_output], name="fc_encoder")

        input_array = Input(shape=(23, 1))
        x = Conv1D(
            20,
            7,
            data_format="channels_last",
            activation="selu",
            kernel_initializer="lecun_normal",
            strides=2,
            padding="same",
        )(input_array)
        x = BatchNormalization()(x)
        x = Conv1D(20, 5, activation="selu", kernel_initializer="lecun_normal", strides=1, padding="same")(x)
        x = BatchNormalization()(x)
        env_encoded = MaxPooling1D(pool_size=2)(x)
        env_flatten = tf.keras.layers.Flatten()(env_encoded)
        env_encoder = Model(input_array, env_flatten, name="env_encoder")

        env_input = Input(shape=(env_flatten.shape[1],))
        fc_input = Input(shape=(fc_flatten.shape[1],))
        x = Concatenate(axis=1)([fc_input, env_input])
        x = Dense(256, activation="selu", kernel_initializer="lecun_normal")(x)
        x = Dropout(0.3)(x)
        z_merged = Dense(128, activation="selu", kernel_initializer="lecun_normal")(x)
        fusion = Model([fc_input, env_input], z_merged, name="fusion_branch")

        latent_inputs = Input(shape=(z_merged.shape[1],), name="z_merged_latent_inputs")
        y_hat_inputs = Input(shape=(num_classes,), name="y_inputs")
        x = Concatenate(axis=1)([latent_inputs, y_hat_inputs])
        x = Dense(3200, activation="selu", kernel_initializer="lecun_normal")(x)
        x = Reshape((5, 5, 128))(x)
        x = UpSampling2D((4, 4))(x)
        x = Conv2D(64, (3, 3), activation="selu", kernel_initializer="lecun_normal", padding="same")(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation="selu", kernel_initializer="lecun_normal", padding="same")(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation="selu", kernel_initializer="lecun_normal", padding="same")(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation="selu", kernel_initializer="lecun_normal", padding="same")(x)
        fc_decoder = Model([latent_inputs, y_hat_inputs], decoded, name="decoded")

        latent_inputs_env = Input(shape=(z_merged.shape[1],), name="z_merged_latent_inputs_env")
        x = Dense(120, activation="selu", kernel_initializer="lecun_normal")(latent_inputs_env)
        x = Reshape((6, 20))(x)
        x = UpSampling1D(size=2)(x)
        x = Conv1D(20, 5, activation="selu", kernel_initializer="lecun_normal", strides=1, padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv1DTranspose(1, 7, activation="selu", kernel_initializer="lecun_normal", strides=2, padding="same")(x)
        env_decoded = Cropping1D((0, 1))(x)
        env_decoder = Model(latent_inputs_env, env_decoded, name="env_decoded")

        return fc_encoder, fc_decoder, env_encoder, env_decoder, fusion, loss_sup

    def _build_autoencoder(
        self,
        fc_encoder: Model,
        fc_decoder: Model,
        env_encoder: Model,
        env_decoder: Model,
        fusion: Model,
        loss_sup,
    ):
        cfg = self.cfg
        trainer = self

        class AE(tf.keras.Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.fc_encoder = fc_encoder
                self.fc_decoder = fc_decoder
                self.env_encoder = env_encoder
                self.env_decoder = env_decoder
                self.fusion = fusion

                self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
                self.fc_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="fc_reconstruction_loss")
                self.env_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="env_reconstruction_loss")
                self.penalty_loss_tracker = tf.keras.metrics.Mean(name="penalty_loss")
                self.clf_loss_tracker = tf.keras.metrics.Mean(name="clf_loss")

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
                    self.penalty_loss_tracker,
                    self.clf_loss_tracker,
                ]

            def train_step(self, data):
                (fc, env), y = data

                with tf.GradientTape() as tape_ae:
                    fc_z, _h1, y_pred = self.fc_encoder(fc, training=True)
                    env_z = self.env_encoder(env, training=True)
                    z = self.fusion([fc_z, env_z], training=True)

                    fc_rec = self.fc_decoder([z, y_pred], training=True)
                    env_rec = self.env_decoder(z, training=True)

                    re_loss = tf.keras.losses.MeanSquaredError(reduction="sum_over_batch_size")
                    fc_re = re_loss(fc, fc_rec)
                    env_re = re_loss(env, env_rec)

                    n_samples = tf.cast(tf.shape(fc)[0], tf.float32)
                    penalty_loss = trainer._compute_drl_penalty(fc_z, y_pred, n_samples)
                    total_loss = cfg.alpha_fc * fc_re + cfg.alpha_env * env_re + cfg.gamma * penalty_loss

                grads = tape_ae.gradient(total_loss, self.trainable_weights)
                self.optimizer_ae.apply_gradients(zip(grads, self.trainable_weights))

                with tf.GradientTape() as tape_clf:
                    _fc_z, _h1, y_pred = self.fc_encoder(fc, training=True)
                    clf_loss = loss_sup(y, y_pred)
                grads = tape_clf.gradient(clf_loss, self.trainable_weights)
                self.optimizer_clf.apply_gradients(zip(grads, self.trainable_weights))

                self.total_loss_tracker.update_state(total_loss)
                self.fc_reconstruction_loss_tracker.update_state(fc_re)
                self.env_reconstruction_loss_tracker.update_state(env_re)
                self.penalty_loss_tracker.update_state(penalty_loss)
                self.clf_loss_tracker.update_state(clf_loss)

                return {m.name: m.result() for m in self.metrics}

            def test_step(self, data):
                (fc, env), y = data

                fc_z, _h1, y_pred = self.fc_encoder(fc, training=False)
                env_z = self.env_encoder(env, training=False)
                z = self.fusion([fc_z, env_z], training=False)

                fc_rec = self.fc_decoder([z, y_pred], training=False)
                env_rec = self.env_decoder(z, training=False)

                re_loss = tf.keras.losses.MeanSquaredError(reduction="sum_over_batch_size")
                fc_re = re_loss(fc, fc_rec)
                env_re = re_loss(env, env_rec)

                n_samples = tf.cast(tf.shape(fc)[0], tf.float32)
                penalty_loss = trainer._compute_drl_penalty(fc_z, y_pred, n_samples)
                clf_loss = loss_sup(y, y_pred)

                total_loss = cfg.alpha_fc * fc_re + cfg.alpha_env * env_re + cfg.gamma * penalty_loss

                self.total_loss_tracker.update_state(total_loss)
                self.fc_reconstruction_loss_tracker.update_state(fc_re)
                self.env_reconstruction_loss_tracker.update_state(env_re)
                self.penalty_loss_tracker.update_state(penalty_loss)
                self.clf_loss_tracker.update_state(clf_loss)

                return {m.name: m.result() for m in self.metrics}

        return AE()

    @staticmethod
    def _safe_macro_roc_auc(y_true_onehot: np.ndarray, y_score: np.ndarray) -> float:
        try:
            return float(roc_auc_score(y_true_onehot, y_score, average="macro", multi_class="ovo"))
        except ValueError:
            return np.nan

    @staticmethod
    def _safe_binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        try:
            return float(roc_auc_score(y_true, y_score))
        except ValueError:
            return np.nan

    def _evaluate_identifiability(
        self,
        specs: Tuple[VariableSpec, ...],
        representations_train: Dict[str, np.ndarray],
        representations_val: Dict[str, np.ndarray],
        cov_train: np.ndarray,
        cov_val: np.ndarray,
        labels_train: pd.DataFrame,
        labels_val: pd.DataFrame,
        envir_train: np.ndarray,
        envir_val: np.ndarray,
    ) -> Dict[str, float]:
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.svm import SVC, SVR
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        metrics: Dict[str, float] = {}

        for spec in specs:
            y_train = self._extract_variable(spec.name, cov_train, labels_train, envir_train)
            y_val = self._extract_variable(spec.name, cov_val, labels_val, envir_val)

            if spec.kind.lower() == "regression":
                y_train_flat = y_train.reshape(-1)
                y_val_flat = y_val.reshape(-1)
                for rep_name, x_train_rep in representations_train.items():
                    x_val_rep = representations_val[rep_name]
                    knn = KNeighborsRegressor(n_neighbors=3)
                    svr = SVR()
                    dt = DecisionTreeRegressor(random_state=0)
                    model = VotingRegressor(estimators=[("knn", knn), ("svm", svr), ("dt", dt)])
                    model.fit(x_train_rep, y_train_flat)
                    y_pred = model.predict(x_val_rep)
                    metrics[f"{rep_name}_{spec.name}_rmse"] = float(mean_squared_error(y_val_flat, y_pred, squared=False))
                continue

            y_train_flat = self._flatten_labels(y_train)
            y_val_flat = self._flatten_labels(y_val)
            classes = np.unique(y_train_flat)
            for rep_name, x_train_rep in representations_train.items():
                x_val_rep = representations_val[rep_name]
                knn = KNeighborsClassifier(n_neighbors=3)
                svc = SVC(probability=True, random_state=0)
                dt = DecisionTreeClassifier(random_state=0)
                model = VotingClassifier(estimators=[("knn", knn), ("svm", svc), ("dt", dt)], voting="soft")
                model.fit(x_train_rep, y_train_flat)
                y_score = model.predict_proba(x_val_rep)
                if y_score.shape[1] == 2:
                    metrics[f"{rep_name}_{spec.name}_auc_roc"] = self._safe_binary_auc(y_val_flat, y_score[:, 1])
                else:
                    metrics[f"{rep_name}_{spec.name}_auc_roc"] = self._safe_macro_roc_auc(y_val, y_score)
                y_pred = np.argmax(y_score, axis=1)
                metrics[f"{rep_name}_{spec.name}_f1"] = float(f1_score(y_val_flat, y_pred, labels=classes, average="macro"))

        return metrics

    def _evaluate_fold(
        self,
        fold_idx: int,
        x_train: np.ndarray,
        x_val: np.ndarray,
        env_train: np.ndarray,
        env_val: np.ndarray,
        cov_train: np.ndarray,
        cov_val: np.ndarray,
        labels_train: pd.DataFrame,
        labels_val: pd.DataFrame,
        y_train_target: np.ndarray,
        y_val_target: np.ndarray,
        fc_encoder: Model,
        env_encoder: Model,
        fusion: Model,
        fc_decoder: Model,
        history,
    ) -> Dict[str, float]:
        fold_row: Dict[str, float] = {"fold": fold_idx}

        fold_row["epochs"] = len(history.history["loss"])
        fold_row["train_loss"] = float(history.history["loss"][-1])
        fold_row["val_loss"] = float(history.history["val_loss"][-1])
        fold_row["train_fc_re_loss"] = float(history.history["fc_reconstruction_loss"][-1])
        fold_row["val_fc_re_loss"] = float(history.history["val_fc_reconstruction_loss"][-1])
        fold_row["train_env_re_loss"] = float(history.history["env_reconstruction_loss"][-1])
        fold_row["val_env_re_loss"] = float(history.history["val_env_reconstruction_loss"][-1])
        fold_row["train_penalty_loss"] = float(history.history["penalty_loss"][-1])
        fold_row["val_penalty_loss"] = float(history.history["val_penalty_loss"][-1])
        fold_row["train_clf_loss"] = float(history.history["clf_loss"][-1])
        fold_row["val_clf_loss"] = float(history.history["val_clf_loss"][-1])

        fc_train_encoded, _, y_train_pred = fc_encoder.predict(x_train, verbose=0)
        env_train_encoded = env_encoder.predict(env_train, verbose=0)
        z_train = fusion.predict([fc_train_encoded, env_train_encoded], verbose=0)

        fc_val_encoded, _, y_val_pred = fc_encoder.predict(x_val, verbose=0)
        env_val_encoded = env_encoder.predict(env_val, verbose=0)
        z_val = fusion.predict([fc_val_encoded, env_val_encoded], verbose=0)

        fc_val_decoded = fc_decoder.predict([z_val, y_val_pred], verbose=0)
        fold_row["val_re_rmse"] = float(
            mean_squared_error(x_val.reshape(x_val.shape[0], -1), fc_val_decoded.reshape(x_val.shape[0], -1), squared=False)
        )

        target_name = self.cfg.target_variable.lower()
        y_val_target_flat = self._flatten_labels(y_val_target)
        y_val_target_pred = (
            pd.DataFrame(y_val_pred).idxmax(axis=1).values
            if y_val_pred.shape[1] > 1
            else (y_val_pred[:, 0] >= 0.5).astype(int)
        )

        if y_val_pred.shape[1] == 1:
            fold_row[f"{target_name}_auc_roc"] = self._safe_binary_auc(y_val_target_flat, y_val_pred[:, 0])
        elif y_val_pred.shape[1] == 2:
            fold_row[f"{target_name}_auc_roc"] = self._safe_binary_auc(y_val_target_flat, y_val_pred[:, 1])
        else:
            fold_row[f"{target_name}_auc_roc"] = self._safe_macro_roc_auc(y_val_target, y_val_pred)
        fold_row[f"{target_name}_f1"] = float(f1_score(y_val_target_flat, y_val_target_pred, average="macro"))

        fold_row["z_dim"] = int(z_train.shape[1])
        fold_row["fc_z_dim"] = int(fc_train_encoded.shape[1])
        fold_row["env_z_dim"] = int(env_train_encoded.shape[1])

        eval_specs = self.cfg.eval_specs
        if not any(spec.name.lower() == self.cfg.target_variable.lower() for spec in eval_specs):
            eval_specs = (VariableSpec(name=self.cfg.target_variable, kind="categorical"),) + tuple(eval_specs)

        fold_row.update(
            self._evaluate_identifiability(
                eval_specs,
                representations_train={
                    "z": z_train,
                    "fc_z": fc_train_encoded,
                    "env_z": env_train_encoded,
                    "env": env_train.reshape(env_train.shape[0], -1),
                },
                representations_val={
                    "z": z_val,
                    "fc_z": fc_val_encoded,
                    "env_z": env_val_encoded,
                    "env": env_val.reshape(env_val.shape[0], -1),
                },
                cov_train=cov_train,
                cov_val=cov_val,
                labels_train=labels_train,
                labels_val=labels_val,
                envir_train=env_train.reshape(env_train.shape[0], -1),
                envir_val=env_val.reshape(env_val.shape[0], -1),
            )
        )

        return fold_row

    def run(self) -> pd.DataFrame:
        x_all = self.cfg.x_all
        covariates = self.cfg.covariates
        labels = self.cfg.labels
        envir = self.cfg.envir
        bs_with_nan = self.cfg.bs_with_nan

        target_all = self._extract_variable(self.cfg.target_variable, covariates, labels, envir)
        stratify_labels = self._flatten_labels(target_all)
        
        os.makedirs(self.cfg.save_results_dir, exist_ok=True)

        all_fold_rows: List[Dict[str, float]] = []
        global_fold_idx = 0

        for rs in self.cfg.random_states:
            skf = StratifiedKFold(n_splits=self.cfg.n_splits, random_state=rs, shuffle=True)

            for train_idx, val_idx in skf.split(x_all, stratify_labels):
                x_train = x_all[train_idx, :, :].astype(np.float32)
                x_val = x_all[val_idx, :, :].astype(np.float32)

                cov_train = covariates[train_idx, :]
                cov_val = covariates[val_idx, :]

                env_train = envir[train_idx, :].astype(np.float32)
                env_val = envir[val_idx, :].astype(np.float32)

                bs_train = bs_with_nan[train_idx, :]
                bs_val = bs_with_nan[val_idx, :]

                col_mean = np.nanmean(bs_train, axis=0)
                inds = np.where(np.isnan(bs_train))
                env_train[:, 10:][inds] = np.take(col_mean, inds[1])
                inds = np.where(np.isnan(bs_val))
                env_val[:, 10:][inds] = np.take(col_mean, inds[1])

                x_train = np.expand_dims(x_train, axis=-1)
                x_val = np.expand_dims(x_val, axis=-1)
                env_train = np.expand_dims(env_train, axis=-1)
                env_val = np.expand_dims(env_val, axis=-1)

                y_train_target_raw = self._extract_variable(self.cfg.target_variable, cov_train, labels.iloc[train_idx, :], env_train.reshape(env_train.shape[0], -1))
                y_val_target_raw = self._extract_variable(self.cfg.target_variable, cov_val, labels.iloc[val_idx, :], env_val.reshape(env_val.shape[0], -1))
                y_train_target = self._format_target(y_train_target_raw, self.cfg.target_variable)
                y_val_target = self._format_target(y_val_target_raw, self.cfg.target_variable)
                num_classes = y_train_target.shape[1] if y_train_target.ndim == 2 else 1

                fc_encoder, fc_decoder, env_encoder, env_decoder, fusion, loss_sup = self._build_models(num_classes)
                autoencoder = self._build_autoencoder(fc_encoder, fc_decoder, env_encoder, env_decoder, fusion, loss_sup)

                steps_per_epoch = max(int(x_train.shape[0] / max(self.cfg.batch_size, 1)), 1)
                decay_factor = (self.cfg.clf_final_lr / self.cfg.clf_initial_lr) ** (1.0 / max(self.cfg.epochs, 1))
                lrs = tf.keras.optimizers.schedules.ExponentialDecay(
                    self.cfg.clf_initial_lr,
                    steps_per_epoch,
                    decay_factor,
                    staircase=False,
                )

                autoencoder.compile(
                    optimizer_ae=tf.keras.optimizers.Adam(learning_rate=self.cfg.ae_learning_rate),
                    optimizer_clf=tf.keras.optimizers.Adam(learning_rate=lrs),
                )

                callbacks = []
                if self.cfg.use_early_stopping:
                    callbacks.append(
                        EarlyStopping(
                            monitor="val_clf_loss",
                            patience=self.cfg.early_stopping_patience,
                            restore_best_weights=True,
                        )
                    )

                history = autoencoder.fit(
                    x=[x_train, env_train],
                    y=y_train_target,
                    validation_data=([x_val, env_val], y_val_target),
                    epochs=self.cfg.epochs,
                    batch_size=self.cfg.batch_size,
                    shuffle=True,
                    callbacks=callbacks,
                    verbose=1,
                )

                row = self._evaluate_fold(
                    global_fold_idx,
                    x_train,
                    x_val,
                    env_train,
                    env_val,
                    cov_train,
                    cov_val,
                    labels.iloc[train_idx, :],
                    labels.iloc[val_idx, :],
                    y_train_target,
                    y_val_target,
                    fc_encoder,
                    env_encoder,
                    fusion,
                    fc_decoder,
                    history,
                )
                row["random_state"] = rs
                row["penalty"] = self.cfg.penalty
                row["gamma"] = self.cfg.gamma
                all_fold_rows.append(row)
                global_fold_idx += 1

        fold_results = pd.DataFrame(all_fold_rows)
        fold_results.loc[len(fold_results)] = fold_results.mean(numeric_only=True)
        fold_results.loc[len(fold_results) - 1, "fold"] = -1
        fold_results.loc[len(fold_results) - 1, "penalty"] = "mean"

        output_csv = os.path.join(self.cfg.save_results_dir, self.cfg.results_filename)
        fold_results.to_csv(output_csv, index=False)
        return fold_results
