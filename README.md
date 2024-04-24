# Orca-Detection
## Introduction
This repository contains the code I used in google colab to train a machine learning model that can detect the vocalization of Killer Whales in live hydrophone data. It includes:
- The code I used to retrieve the dataset from an s3 web bucket
- The code that was used to split the raw WAV files into just the target and background noise
- The code that sets up package dependancies/fixes compatibility errors before actually running the training and prediction scripts

I used the open source (ANIMAL-SPOT)[https://github.com/ChristianBergler/ANIMAL-SPOT] repository to create my model. It is a generalized framework for bioacoustic signal detection through machine learning that has proven effective for many different species. This repository was created by Christian Bergler and his associates at the Friedrich Alexander University. Thank you to Christian and his team for their contribution that made my project possible!

The Data used from training was sourced from the (Orcasound)[https://www.orcasound.net/] project, which is an open source project which seeks to make avaialable to the public live data from hydrophones located in the salish sea. Information on how to access this live data, as well as curated datasets for machine learning (such as the one used for this project) can be found (in their github repository.)[https://github.com/orcasound/orcadata]

I made frequent use of GPT-4 and Claude 3 to help me write my code and fix bugs throughout the project. Credit to OpenAI and Anthropic for the creation of these models that were extremely useful in the creation of this project! These LLMs were instrumental in my understanding of the technical material and fixing persistent technical issues I encountered.

## Data Understanding
The data used in this project was sourced from the Orcasound project, as stated above. Specifically, a curated dataset specifically for training detection ML models was used. The dataset is composed of a large number of hydrophone recordings that contain Killer Whale vocalizations with their start time and duration labeled in a separate TSV file. Due to the size and nature of the data it is difficult to provide overview statistics that describe it. 
However, the data that composes the corpus was chosen because contains a wide variety of vocalizations in a wide variety of contexts.

The data was retrieved using AWS CLI using the code in the "Data_Loading" Notebook in this repository.

The code in the "Signal_Split" notebook was used to separate the original unsegmented sound files into labeled clips that contain only target signals or background noise according to the original labels. This code renders the sound files into a format that animal-spot will accept.

The final model was trained, tested, and validated on a randomly selected subset of 25% of the total data composed of 66% noise and 33% target signals to optimize the model's training time and size. The code that selects this subset in a reproducible manner can also be foud in the "Signal_Split" notebook.

## Notebooks, Google Colab, and Google Drive
As mentioned above, much of the code for this project created with the assistance of ChatGPT and Claude 3. The code is split across three separate google colab notebooks, and an additional fourth notebook which generated relevant visualizations related to the results. They have been downloaded and saved in this respository. 

The notebooks can either be run locally, or directly accessed in google drive and copied with the links below. I would recommend just copying the notebooks in google drive, as I did all of my experimentation and training in google drive as well.

- The "Data_Loading" notebook was used to retrieve the data from S3. 
- The "Signal_Split split the original WAV files into labeled segments named in the proper format for Animal-Spot. Then, it selects the subset of the data that was used for training.
- The "Animal_Spot" notebook contains code that will install the correct version of dependancies. It contains code to run the training, prediction, and evaluation scripts.
- The "Animal_Spot_Eval" notebook contains the code that was used to generate the visualizations used in this readme file and the final presentation.

The files were stored in Google Drive for convenience and integration with Colab. This way of doing the project was wasteful and more expensive than it needed to be - a premium subscription is required for both Google Drive and Google Colab to store the data and actually run the training with a TPU. A better way of doing this project would be to learn how to configure an Amazon EC2 instance and run the training and prediction there instead. EC2 has a greater availability of compute power than Google Colab, and if I had started doing the project in EC2 from the beginning it would have made it simple to build and deploy a webapp that uses the final model.

Links to the colabs the notebooks were saved from:
- https://colab.research.google.com/drive/1AM6J2YwNfKuJQ_Yn_yKGP0wGy8au2Kc7?usp=sharing (Animal_Spot)
- https://colab.research.google.com/drive/1WNbXQsQeiuD7OeQvzeuSgfWUFQrUOlAU?usp=sharing (Signal_Split)
- https://colab.research.google.com/drive/1G5lpYvgDKOqNJ2fNwEL7p-3xB3jvo0--?usp=sharing (Data_Loading)
- https://colab.research.google.com/drive/16Nd05dEy0YiS1Rng6d6d_z4dkgswGPP_?usp=sharing (Animal_Spot_Eval)

## Methodology and Training, Prediction, and Evaluation parameters.
As stated above, the final model was trained on a subset of the total dataset in order to reduce the model's training time. A ratio of 2:1 of noise:target was selected due to the high ratio of background noise to target signals in live hydrophone data. A configuration option which removed empty  noise files was enabled, which resulted in a final ratio of about 1.8:1 noise:target.

The parameters for the Training, Prediciton, and Evaluation scripts of the ANIMAL-SPOT respository used in this project are contained in this respoistory's labeled config files. To use them, place each config file in the Training, Prediction, or Evaluation folders of the ANIMAL-SPOT respository according to their label, and then rename them to 'config'. 
(insert extensive coverage of justification for each choice of parameter)

## Results
The final version of the model had a test accuracy of 84% at the conclusion of training. An example graph showing the model's accuracy as it progressed through epochs and the final confusion matrix can be found below.

![](https://github.com/Davidkeebler/Orca-Detection/blob/main/Images/train_vs_val.png)

![](https://github.com/Davidkeebler/Orca-Detection/blob/main/Images/confusion_matrix.png)

## Example Spectrogram
Below is an example of the model's labeling capacity. It is demonstrated on one of the original files from the training dataset that has not been segmented into target and noise files. None of the segments from this original file were used in the training of the model.

The numbered boxes indicate where the model predicts there is a call. The model does not always detect the calls (especially when they are intermixed with static, or are exceptionally low or high volume) but false positives are quite rare, and are actually predictable. Because of the configuration option to remove blank spectrograms, the model is only exposed to blank spectrograms in the call data (if anywhere at all). This results in the model predicting blank/empty segments of the spectrogram as calls. We could create an extra script to fix this issue and increase the model's effectiveness. 

False negatives are more common, but are less concerning. Researchers may miss potential calls in the data, but if we run this script to analyze live hydrophone data 24/7, we will still get a lot of high-quality data. 

With the addition of a script that detects empty segments of the spectrograms and ensures they are marked as noise, we can safely increase the sensitivty of the model by lowering its detection threshold and further increase the accuracy of the model. This is on the to-do list for expanding the project.

![](https://github.com/Davidkeebler/Orca-Detection/blob/main/Images/example_spectrogram.png)

# Analysis and Conclusions
As is clear from the example spectrogram, my final model is not perfect - false negatives are common, and false positives are rare but do occur. However, for the model's intended use case, it is good enough. The occurence of these false positives and false negatives only slightly detract from the model's performance, and can even be compensated for in some cases. With an additional step of quick human verification, this model is still capable of saving bioacoustic researchers a lot of time in the process of labeling orca vocalizations in raw hydrophone data. 

# Next Steps
As stated above, this project is not perfect and there are several ways I plan to improve it in the future. Here are a few improvements I plan to make and expansions to the scope of the project which will eventually be completed:

- Retrain the model with more data that has been better curated. A wider variety of examples of background noise will help prevent the false positives that are occuring. It may also be beneficial to ensure as many different types of Orca vocalization/acoustic communication are present in the training data as possible once I have catalogued these sounds.
- Create a webapp that will retrieve the live hydrophone data and label it on demand. It is unlikely that this webapp will constantly run due to server costs, but I would like a publicly available demonstration of the project to be available.
- Integrate this model into a larger pipeline for the analysis of Orca vocalizations. The original scope of the project included creating a clustering model that identifies different types of vocalizations within the target signals, and then to use that clustered data to train a neural network that directly detects and classifies acoustic communications. 
- The scope of the project had to be reduced partway through due to difficulties encountered in running the ANIMAL-SPOT code which were time-consuming to overcome. The ultimate goal of the project is to create a framework that can be run online and used in real time to detect and classify the vocalizations of the animals so that researchers can label them with appropriate context and determine their meaning.

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