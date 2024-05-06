# Reproducibility
Extensive measures have been taken to ensure the reproducibility of results in google colab for this project. If you attempt to reproduce these results on a local computer or EC2 instance, you may experience additional difficulties.

An extra notebook Orca_Detection_Easy_Reproducibility.ipynb has been included to make reproducibility of the project easier to verify. While this notebook can be run locally, it will encounter some errors. If it is run in [google colab](https://colab.research.google.com/drive/1CXfPAcJs8CZ6E-M_ryOL4IIlaz2HLbOl?usp=sharing), you can simply input your kaggle credentials and then run all of the cells.

## Data retrieval and segmenting
For retrieving, segmenting, and splitting the data, the notebooks in this repository will work perfectly fine in colab without any additional tweaking other than adding your own AWS credentials. In an environment other than colab, you will need to ignore the cells that mount google drive to the notebook and replace all directories in the notebook with directories that are appropriate for your local computer or server instance.

## Training, prediction, and evaluation
Package dependancies should be handled differently if you are not running the project in colab. Instead of re-installing packages with the notebook every time it starts, it would be better to create a conda environment with the packages listed in the [Animal-Spot repository](https://github.com/ChristianBergler/ANIMAL-SPOT) as dependancies. The most important packages are:
- torch 1.11.0+cu113
- torchvision 0.12.0+cu113
- torchaudio 0.11.0+cu113
- tensorboardx
The most up-to-date versions of the following packages will work fine, and are also required:
- resampy
- pillow
- soundfile
- scikit-image
- six
- opencv-python

The parameters used for training, prediction, and evaluation are stored in the config files in this repository. Once you have downloaded the animal-spot repository, you should replace the config files in the training, prediction, and evaluation folders with the configs in this repository and rename them to 'config' to ensure the parameters are correct to reproduce my result.