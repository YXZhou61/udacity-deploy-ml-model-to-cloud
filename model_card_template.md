# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Yuxuan created the model. It is Random Forest Classification using the default hyperparameters in scikit- 1.3.0.

## Intended Use
This model should be used to predict whether income exceeds $50K/yr based on census data.

## Training Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/20/census+income). Basic cleaning was applied to the data to remove space and rows with strange values.

The cleaned data set has 30163 rows, and a 80-20 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Evaluation Data
Similar to training data described above.

## Metrics
The model was evaluated using precision, recall and fbeta. The value is 0.0644, 1.0, and 0.1210.

## Ethical Considerations

## Caveats and Recommendations