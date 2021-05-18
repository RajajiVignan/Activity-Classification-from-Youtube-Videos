# Activity-Classification-from-Youtube-Videos


## Objective
  Finding out the activity done by person from the youtube video link provided. The activity ranges from Adventure sports to Reading and many more.  
  
## Introduction
  This project is made possible with the intense resources available from deepmin.org on the Activity identification. There are several Articles and publications on Kinetics. This is a direct implementation of the same. 
  
## Data Processing
  The UCF datasets are used in this project. One of the tensorflow model 'id3' is used for transfer learning our model. 
  

## Process and outcome
  1. Input the Youtube-Link and the start and end segments of the video to be processed.
  2. Then, the Model downloads the video on the server and then slices out the specific segments mentioned.
  3. Now, we use the pretrained model to predict the activity. In the meanwhile, the video segment is pre-processed to meet the format of the model. 
  4. Display of the prediction results.
