# #!/bin/bash -l

experiment_name=ReciprocityGAN_freq10_A_train_0.1SamplingRate_random
repo_name=specialization-exam-and-artifact-defense

path_script=$HOME/$repo_name/src/wavefield-reconstruction/src
path_data=$HOME/data
path_model=$HOME/model/$experiment_name

mkdir $HOME/data
mkdir $HOME/model/
mkdir $path_model

yes | cp -r $path_script/. $path_model

if [ ! -f $path_data/mapping_result.hdf5 ]; then
	wget https://www.dropbox.com/s/n2ahp8dr4jog6em/mapping_result.hdf5 \
		-O $path_data/mapping_result.hdf5
fi

if [ ! -f $path_data/InterpolatedCoil_freq10.0_A_test_0.1SamplingRate_random.hdf5 ]; then
	wget https://www.dropbox.com/s/gcjetgb3scj31lm/InterpolatedCoil_freq10.0_A_test_0.1SamplingRate_random.hdf5 \
		-O $path_data/InterpolatedCoil_freq10.0_A_test_0.1SamplingRate_random.hdf5
fi

if [ ! -f $path_data/InterpolatedCoil_freq10.0_B_test_0.1SamplingRate_random.hdf5 ]; then
	wget https://www.dropbox.com/s/72w6vyxv1nmkfjm/InterpolatedCoil_freq10.0_B_test_0.1SamplingRate_random.hdf5 \
		-O $path_data/InterpolatedCoil_freq10.0_B_test_0.1SamplingRate_random.hdf5
fi

if [ ! -f $path_data/InterpolatedCoil_freq10.0_Mask_0.1SamplingRate_random.hdf5 ]; then
	wget https://www.dropbox.com/s/e10c9gjbrisjsff/InterpolatedCoil_freq10.0_Mask_0.1SamplingRate_random.hdf5 \
		-O $path_data/InterpolatedCoil_freq10.0_Mask_0.1SamplingRate_random.hdf5
fi

CUDA_VISIBLE_DEVICES=0 python $path_script/main.py --experiment_dir=wave_recon --phase train --batch_size 1 \
	--checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample --log_dir $path_model/log
