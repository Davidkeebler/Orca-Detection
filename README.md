# Orca-Detection
## Introduction
This repository contains the code I used in google colab to train a machine learning model that can detect the vocalization of Killer Whales in live hydrophone data. It includes the code I used to retrieve the dataset from an s3 web bucket, the code that was used to split the raw WAV files into just the target and background noise, and the code that sets up package dependancies/fixes compatibility errors before actually running the training and prediction scripts.
I used the open source (ANIMAL-SPOT)[https://github.com/ChristianBergler/ANIMAL-SPOT] repository to create my model. It is a generalized framework for bioacoustic signal detection through machine learning that has proven effective for many different species. This repository was created by Christian Bergler and his associates at the Friedrich Alexander University. Thank you to Christian and his team for their contribution that made my project possible!
The Data used from training was sourced from the (Orcasound)[https://www.orcasound.net/] project, which is an open source project which seeks to make avaialable to the public live data from hydrophones located in the salish sea. Information on how to access this live data, as well as curated datasets for machine learning (such as the one used for this project) can be found (in their github repository.)[https://github.com/orcasound/orcadata]
I made frequent use of GPT-4 and Claude 3 to help me write my code and fix bugs throughout the project. Credit to OpenAI and Anthropic for the creation of these models that were extremely useful in the creation of this project.
## Data Understanding
The data used in this project was sourced from the Orcasound project, as stated above. Specifically, a curated dataset specifically for training detection ML models was used. The dataset is composed of a large number of hydrophone recordings that contain Killer Whale vocalizations with their start time and duration labeled in a separate TSV file. Due to the size and nature of the data it is difficult to provide overview statistics that describe it, but it is safe to say that the dataset contains a wide variety of vocalizations in a wide variety of contexts.
The data was retrieved using AWS CLI. Several python scripts were used to separate the original sound files into labeled clips that contain only target signals or background noise according to the original labels. The final model was trained, tested, and validated on a randomly selected subset of 25% of the total data composed of 66% noise and 33% target signals to optimize the model's training time and size.
## Notebooks, Google Colab, and Google Drive
The code for this project was largely created with the assistance of ChatGPT. The code is split across three separate google colab notebooks, which have been downloaded and saved in this respository. They can either be run locally, or directly accessed and copied online with the links below. One notebook was used to retrieve the data from S3, a second was used to split the original WAV files into their labeled segments and then select the subset used for training, and the third was used to install the correct version of dependancies and run the training, prediction, and evaluation scripts.
The files were stored in Google Drive for convenience and integration with Colab. This way of doing the project was wasteful and more expensive than it needed to be - a premium subscription is required for both Google Drive and Google Colab to store the data and actually run the training with a TPU. I recommend anyone interested in replicating this project looks into AWS EC2 or Google Compute Engine as an alternative for running this project's code and storing its data, as they are likely to be cheaper.
-https://colab.research.google.com/drive/1AM6J2YwNfKuJQ_Yn_yKGP0wGy8au2Kc7?usp=sharing (Animal_Spot)
-https://colab.research.google.com/drive/1WNbXQsQeiuD7OeQvzeuSgfWUFQrUOlAU?usp=sharing (Signal_Split)
-https://colab.research.google.com/drive/1G5lpYvgDKOqNJ2fNwEL7p-3xB3jvo0--?usp=sharing (Data_Loading)
## Methodology and Training, Prediction, and Evaluation parameters.
As stated above, the final model was trained on a subset of the total dataset in order to reduce the model's training time. A ratio of 2:1 of noise:target was selected due to the high ratio of background noise to target signals in live hydrophone data. 
The parameters for the Training, Prediciton, and Evaluation scripts of the ANIMAL-SPOT respository used in this project are contained in this respoistory's labeled config files. To use them, place each config file in the Training, Prediction, or Evaluation folders of the ANIMAL-SPOT respository according to their label, and then rename them to 'config'. 
(insert extensive coverage of justification for each choice of parameter)
## Results
The final version of the model had a test accuracy of 84% at the conclusion of training. An example graph showing the model's accuracy as it progressed through epochs and the final confusion matrix can be found below.

![](https://github.com/Davidkeebler/Orca-Detection/blob/main/Images/train_vs_val.png)

![](https://github.com/Davidkeebler/Orca-Detection/blob/main/Images/confusion_matrix.png)

## Example Spectrogram
Below is an example of the model's labeling capacity. It is demonstrated on one of the original files from the training dataset that has not been segmented into target and noise files. The numbered boxes indicate where the model predicts there is a call. The model does not always detect the calls (especially when they are intermixed with static, or are exceptionally low or high volume) but false positives are quite rare, and are actually predictable. The model tends to predict completely blank spectrograms as calls, and we could create an extra script to fix this issue and increase the model's effectiveness. With that additional change and by lowering the model's detection threshold, it should be more than sufficient to expanding the amount of orca call data available to researchers for machine learning.

![](https://github.com/Davidkeebler/Orca-Detection/blob/main/Images/example_spectrogram.png)

# Analysis and Conclusions
As is clear from the example spectrogram, my final model is not perfect - false negatives are common, and false positives are rare but do occur. However, for the model's intended use case, it is good enough - it is quite rare for the model to detect a signal where one is not present, and it is unlikely that occasionally missing signals in live data will present a problem. 
# Next Steps
As stated above, this project is not perfect and there are several ways I plan to improve it in the future. Here are a few improvements I plan to make and expansions to the scope of the project which will eventually be completed:
- Retrain the model with more data that has been better curated. A wider variety of examples of background noise will help prevent the false positives that are occuring. It may also be beneficial to ensure as many different types of Orca vocalization/acoustic communication are present in the training data as possible once I have catalogued these sounds.
- Create a webapp that will retrieve the live hydrophone data and label it on demand. It is unlikely that this webapp will constantly run due to server costs, but I would like a publicly available demonstration of the project to be available.
- Integrate this model into a larger pipeline for the analysis of Orca vocalizations. The original scope of the project included creating a clustering model that identifies different types of vocalizations within the target signals, and then to use that clustered data to train a neural network that directly detects and classifies acoustic communications. The scope of the project had to be reduced partway through due to difficulties encountered in running the ANIMAL-SPOT code which were time-consuming to overcome. The ultimate goal of the project is to create a framework that can be run online and used in real time to detect and classify the vocalizations of the animals so that researchers can label them with appropriate context and determine their meaning.
# Navigation
All of the code used in this project is stored in the ipynb files in this repository. They are:
- Data_Loading.ipynb
- Signal_split.ipynb
- Animal_Spot.ipynb
- Animal_Spot_Eval.ipynb

The configuration files used for Training, Prediction, and Evaluation are labeled as the following in this repository:
- config-train
- config-prediction
- config-evaluation

The images in this markdown file are stored in the Images subfolder. A presentation of the project is saved in the repository as Presentation.pdf