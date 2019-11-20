#!/bin/bash -l

frequency=10.0
smprate=0.1
scheme=random

experiment_name=ReciprocityGAN_freq${frequency}_A_train_${smprate}SamplingRate_${scheme}
repo_name=specialization-exam-and-artifact-defense
path_data=$HOME/data
path_model=$HOME/model/$experiment_name

savePath=$path_model/test

python run_plot_data.py --input_data $path_data --save_path $savePath --result_path $path_model/sample \
	--freq 10.0  --sampling_scheme random