# Carmna: Classification and regression models for nitrogenase activity
Usage guide:

For classification task, you can use **xgb.pkl** model for prediction.

For regression task, first put the features extracted by ProtT5 (https://github.com/agemagician/ProtTrans) into **svm_1.pkl** model to get the prediction result, put the prediction result and other features into **svm_2.pkl** model to get the final prediction result.

Please first **normalize** the features you extracted with ProtT5 with the data from columns 2 to 1025 of dataset.xlsx!

./Data contains all the literature collection data and training model data.

./code contains all code involved in the model training process.
