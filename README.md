# README
## _Tumor recognition via Tensorflow and Keras models_

This Python project classifies brain MRIs into 4 labels:
* Glioma
* Meningioma
* Pituary Tumor
* No Tumor

The datasest can be found [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). A smaller and less specific dataset (divided into Tumor and No Tumor) can be found [here](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection). This is useful if you are using a less powerful computer and it's taking a long time to build the model with the original dataset. To change it, simply delete the contents of _"Training"_ and _"Testing"_ folders. Then, divide the _"Yes"_ and _"No"_ folders to fill the former folders. Inside the Testing/Training folders you should have the Yes/No division in a folder, like so:
* Training
    * Yes
    * No

Keras builds all the necessary datasets within the first 30 lines of code. The code also stores both the model and the learning progression.
