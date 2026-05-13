import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import scipy as sp
import os

from Trainer import DRLTrainer, TrainerConfig, VariableSpec


def _parse_random_states(raw: str) -> Tuple[int, ...]:
	values = [v.strip() for v in raw.split(",") if v.strip()]
	return tuple(int(v) for v in values)


def _parse_eval_specs(raw: str) -> Tuple[VariableSpec, ...]:
	specs = []
	for item in raw.split(","):
		item = item.strip()
		if not item:
			continue
		parts = [part.strip() for part in item.split(":")]
		if len(parts) != 2:
			raise ValueError(
				f"Invalid eval spec '{item}'. Use 'name:kind' with kind in {{categorical, regression}}."
			)
		name, kind = parts
		specs.append(VariableSpec(name=name, kind=kind))
	return tuple(specs)


def load_and_preprocess_data(
	data_folder: str,
	info_data_path: str,
	diag_labels_path: str,
	excluded_subject_idx: int = 368,
):
	"""Load and preprocess FC matrices and metadata.

	Returns:
		tuple: (stacked_fc_matrices, covariates, labels, envir, bs_with_nan)
	"""
	info_data = sp.io.loadmat(info_data_path)
	_ = sp.io.loadmat(diag_labels_path)

	covariates = info_data["demo_vars"]
	labels = pd.DataFrame(info_data["diag_dummy"])
	envir = info_data["ENV"]
	itv_measure = info_data["ITV_measure"]
	id_cases = info_data["SUBJ_CODES"]

	idx = excluded_subject_idx
	covariates = np.delete(covariates, idx, axis=0)
	labels = labels.drop(idx, axis=0).reset_index(drop=True)
	envir = np.delete(envir, idx, axis=0)
	itv_measure = np.delete(itv_measure, idx, axis=0)
	id_cases = np.delete(id_cases, idx, axis=0)

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


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Run DRL FC+ENV harmonization with configurable CV and disentanglement penalty."
	)

	parser.add_argument(
		"--data-folder",
		type=str,
		required=True,
		default="/mnt/datafast/ines/pronia_fc/FC_matrices/",
		help="Folder containing FC matrices CSV files.",
	)
	parser.add_argument(
		"--info-data-path",
		type=str,
		required=True,
		default="/mnt/datafast/ines/pronia_fc/pronia_dataset_new.mat",
		help="Path to MAT file containing demo_vars, ENV, and labels metadata.",
	)
	parser.add_argument(
		"--diag-labels-path",
		type=str,
		required=True,
		default="/mnt/datafast/ines/pronia_fc/diag_dummy.mat",
		help="Path to diagnosis labels MAT file.",
	)
	parser.add_argument(
		"--save-results-dir",
		type=str,
		required=True,
		default="results/fc_z_site_hsic/",
		help="Directory where CV results CSV will be saved.",
	)
	parser.add_argument(
		"--results-filename",
		type=str,
		default="cv_results.csv",
		help="Output CSV filename.",
	)

	parser.add_argument("--n-splits", type=int, default=5, help="Number of folds for StratifiedKFold.")
	parser.add_argument(
		"--random-states",
		type=str,
		default="42,24",
		help="Comma-separated random states for repeated CV (example: 42,24).",
	)

	parser.add_argument(
		"--penalty",
		type=str,
		default="xcov",
		choices=["none", "xcov", "hsic"],
		help="Disentanglement penalty.",
	)
	parser.add_argument(
		"--target-variable",
		type=str,
		default="site",
		choices=["site", "sex", "diagnosis"],
		help="Variable to deconfound in the supervised head.",
	)
	parser.add_argument(
		"--eval-specs",
		type=str,
		default="site:categorical,sex:categorical,diagnosis:categorical,age:regression,ers:regression",
		help="Comma-separated evaluation specs in the form 'name:kind'.",
	)
	parser.add_argument("--gamma", type=float, default=15.0, help="Penalty weight.")
	parser.add_argument("--alpha-fc", type=float, default=1.0, help="FC reconstruction loss weight.")
	parser.add_argument("--alpha-env", type=float, default=1.0, help="ENV reconstruction loss weight.")
	parser.add_argument("--hsic-sigma", type=float, default=1.0, help="RBF sigma for HSIC.")
	#parser.add_argument("--mmd-sigma", type=float, default=1.0, help="RBF sigma for MMD.")

	parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
	parser.add_argument("--epochs", type=int, default=2000, help="Training epochs.")
	parser.add_argument("--ae-learning-rate", type=float, default=2e-4, help="Autoencoder optimizer LR.")
	parser.add_argument("--clf-initial-lr", type=float, default=1e-4, help="Classifier initial LR.")
	parser.add_argument("--clf-final-lr", type=float, default=1e-5, help="Classifier final LR after decay.")

	parser.add_argument("--seed-value", type=int, default=2020, help="Global random seed.")
	parser.add_argument(
		"--excluded-subject-idx",
		type=int,
		default=368,
		help="Subject index removed before training.",
	)

	parser.add_argument(
		"--use-early-stopping",
		action="store_true",
		help="Enable early stopping on validation clf loss.",
	)
	parser.add_argument(
		"--early-stopping-patience",
		type=int,
		default=250,
		help="Patience for early stopping.",
	)

	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	# Load and preprocess data
	print("Loading and preprocessing data...")
	x_all, covariates, labels, envir, bs_with_nan = load_and_preprocess_data(
		data_folder=args.data_folder,
		info_data_path=args.info_data_path,
		diag_labels_path=args.diag_labels_path,
		excluded_subject_idx=args.excluded_subject_idx,
	)
	print(f"Data loaded: {x_all.shape[0]} subjects, {x_all.shape[1]}x{x_all.shape[2]} FC matrices")
	eval_specs = _parse_eval_specs(args.eval_specs)
	if not any(spec.name.lower() == args.target_variable.lower() for spec in eval_specs):
		eval_specs = (VariableSpec(name=args.target_variable, kind="categorical"),) + eval_specs

	cfg = TrainerConfig(
		x_all=x_all,
		covariates=covariates,
		labels=labels,
		envir=envir,
		bs_with_nan=bs_with_nan,
		save_results_dir=args.save_results_dir,
		results_filename=args.results_filename,
		target_variable=args.target_variable,
		eval_specs=eval_specs,
		n_splits=args.n_splits,
		random_states=_parse_random_states(args.random_states),
		penalty=args.penalty,
		gamma=args.gamma,
		alpha_fc=args.alpha_fc,
		alpha_env=args.alpha_env,
		hsic_sigma=args.hsic_sigma,
		batch_size=args.batch_size,
		epochs=args.epochs,
		ae_learning_rate=args.ae_learning_rate,
		clf_initial_lr=args.clf_initial_lr,
		clf_final_lr=args.clf_final_lr,
		seed_value=args.seed_value,
		use_early_stopping=args.use_early_stopping,
		early_stopping_patience=args.early_stopping_patience,
	)

	trainer = DRLTrainer(cfg)
	results = trainer.run()
	print(results.tail(1))


if __name__ == "__main__":
	main()
