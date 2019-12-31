# Deep-learning based ocean bottom seismic wavefield recovery

Codes for generating results in Siahkoohi, A., Kumar, R. and Herrmann, F.J., 2019. Deep-learning based ocean bottom seismic wavefield recovery. In SEG Technical Program Expanded Abstracts 2019 (pp. 2232-2237).  doi: [10.1190/segam2019-3216632.1](https://doi.org/10.1190/segam2019-3216632.1).

## Prerequisites

This code has been tested using Deep Learning AMI (Amazon Linux) Version 24.2 (predefined `tensorflow_p36` conda environment) on Amazon Web Services (AWS). We used `g3s.xlarge` instance. Follow the steps below to install the necessary libraries:

```bash
cd $HOME
git clone git@github.com:alisiahkoohi/wavefield-reconstruction.git
cd $HOME/wavefield-reconstruction
conda create -n tensorflow pip python=3.6
source activate tensorflow
pip install --user -r  requirements.txt

```

If you do not have GPU, replace `tensorflow-gpu==1.10.0` in the `requirements.txt` file with `tensorflow==1.10.0` and run the commands above.

## Dataset

Links have been provided in `RunTraining.sh` script to automatically download the 10 Hz monochromatic seismic data into the necessary directory. Total size of the dataset for each fequency is 6.52GB + 6.52GB + 6.52GB + 118KB.

## Script descriptions

`RunTraining.sh`\: script for running training. It will make `model/` and `data/` directory in `/home/ec2-user/` for storing training/testing data and saved neural net checkpoints and final results, respectively. Next, it will train a neural net for the experiment for 10 Hz monochromatic seismic data.

`RunTesting.sh`\: script for testing the trained neural net. It will reconstruct the entire subsampled 10 Hz monochromatic seismic data and place the result in `sample/` directory to be used for plotting purposes.

`src/main.py`\: constructs `wavefield_reconstrcution` class using given arguments in `RunTraining.sh`\, defined in `model.py` and calls `train` function in the defined  `wavefield_reconstrcution` class.

`src/model.py`: includes `wavefield_reconstrcution` class definition, which involves `train` and `test` functions.


### Running the code

To perform training, run:

```bash
# Running in GPU

bash RunTraining.sh

```

To evaluated the trained network on test data set run the following. It will automatically load the latest checkpoint saved.

```bash
# Running in GPU

bash RunTesting.sh

```

To generate and save figures shown in paper for 10 Hz monochromatic seismic data run the following:

```bash

bash utilities/genFigures.sh

```

The saving directory can be changed by modifying `savePath` variable in `utilities/genFigures.sh`\.


## Questions

Please contact alisk@gatech.edu for further questions.


## Author

Ali Siahkoohi
