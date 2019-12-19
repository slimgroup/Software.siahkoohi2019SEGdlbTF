# #!/bin/bash -l

experiment_name=ReciprocityGAN_freq10_A_train_0.1SamplingRate_random
repo_name=wavefield-reconstruction

path_script=$HOME/$repo_name/src/
path_data=$HOME/data
path_model=$HOME/model/$experiment_name

CUDA_VISIBLE_DEVICES=0 python $path_script/main.py --experiment_dir=wave_recon --phase test --batch_size 1 \
	--checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample --log_dir $path_model/log \
	--data_path $path_data
