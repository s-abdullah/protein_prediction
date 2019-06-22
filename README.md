# Tertiary Protein Structure Prediction

## Summary
In this project, we used deep learning to predict tertiary protein structures from primary and secondary protein structures. We built two networks, one for predicting torsion angles and one for predicting distance matrices.

The optimizer used is Adamax and the number of epochs is 5 after which the model starts to overfit. Further regularization could be used in future experimental setups.

[Training Data](https://www.kaggle.com/c/cu-deep-learning-spring19-hw2/data)

## Network Architectures
Both Networks use a series of Convolutions after the data is passed through a Bidirectional LSTM. The activation used is Hyperbolic Tangent. 
### Angle Prediction Network
![alt text](https://github.com/atalero/protein_prediction/blob/master/angle_network.png)
### Distance Prediction Network
![alt text](https://github.com/atalero/protein_prediction/blob/master/distance_network.png)

The Lambda Functions in the Keras summaries perform tasks such as masking (which was needed for dealing with variable length input/output sequences).

## Instructions for Training
* Python 3.6 and Tensorflow 1.13.1
* You must add all the pickle files from the Kaggle link above to the ./data folder.

```
python torsion.py
python distance.py
python output.py
```

**Authors: [Abdullah Siddique](https://github.com/s-abdullah), [Andres Talero](https://github.com/atalero)**
