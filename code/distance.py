#!/usr/bin/env python
# coding: utf-8
from tensorflow import keras
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from helper import *
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Conv2D, LeakyReLU, LSTM, Bidirectional, RNN, InputLayer, Lambda, Reshape, Input
from keras.models import Model
from keras.layers.merge import Multiply
from keras.backend import greater
from keras.backend import zeros

def g(matrix):
    #returns zero everywhere the tensor <= 0 and 1 otherwise
    return tf.math.sign(tf.math.abs(matrix))

def m(inputs):
    #elementwise multiplication of inputs
    return Multiply()(inputs)
#read in all the data from pickle files
indices, pdbs, length_aas, pdb_aas, q8s, dcalphas, psis, phis, msas  = load_pickle("../data/train_fold_1.pkl")
for i in range(2,11):
  indices2, pdbs2, length_aas2, pdb_aas2, q8s2, dcalphas2, psis2, phis2, msas2  = load_pickle("../data/train_fold_"+str(i)+".pkl")
  pdbs += pdbs2
  length_aas += length_aas2
  pdb_aas += pdb_aas2
  q8s += q8s2
  dcalphas += dcalphas2
  psis += psis2
  phis += phis2
  msas += msas2

# chanigng tfrom list to np array
pdbs = np.array(pdbs)
length_aas = np.array(length_aas)
pdb_aas = np.array(pdb_aas)
q8s = np.array(q8s)
dcalphas = np.array(dcalphas)
psis = np.array(psis)
phis = np.array(phis)
msas = np.array(msas)


print(len(pdbs))



indices = np.arange(len(pdbs))
np.random.shuffle(indices)
# val = indices[:344]
train = indices


pdbs = pdbs[train].tolist()
length_aas = length_aas[train].tolist()
pdb_aas = pdb_aas[train].tolist()
q8s = q8s[train].tolist()
dcalphas = dcalphas[train].tolist()
psis = psis[train].tolist()
phis = phis[train].tolist()
msas = msas[train].tolist()



sequence_length = 384 #max sequence length
outputs = np.array(produce_outputs(dcalphas,sequence_length))
inputs = np.array(produce_inputs(pdb_aas, q8s, msas, sequence_length))

max_input_length = 384
output_shape = 384
out_inputs = Input(shape=(output_shape, output_shape), name = "outputs")
main_input = Input(shape=(52, output_shape), name='main_input')

#MODEL


l = Bidirectional(LSTM(400, return_sequences=False), name = "Bidirectional_LSTM1")(main_input)

l = Dense(500, activation='tanh', name = "dense1")(l)
l = Dense(500, activation='tanh', name = "dense2")(l)
l = Dense(500, activation='tanh', name = "dense3")(l)
l = Dense(((output_shape)**2), activation='relu', name = "dense4")(l)

l = Reshape((output_shape, output_shape,1), name = "reshape")(l)


#keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', ...
l = Conv2D(10, 5, strides=(1, 1), padding='same')(l)
l = Conv2D(20, 5, strides=(1, 1),padding='same')(l)
l = Conv2D(30, 5, strides=(1, 1),padding='same')(l)
l = Conv2D(20, 5, strides=(1, 1),padding='same')(l)
l = Conv2D(1, 5, strides=(1, 1),padding='same')(l)


l = Reshape((output_shape**2,), name = "reshape2")(l)


l = Reshape((output_shape,output_shape), name = "reshape3")(l)

# multiplying the out_inputs with the g cunction
mask = Lambda(g, name = "conversion")(out_inputs)

# multiplying the out_inputs with the m cunction
l = Lambda(m, name = "multiply")([l,mask])




# the Input layer and three Dense layers
model = Model(inputs=[main_input,out_inputs], outputs=[l])
model.compile(optimizer='adamax', loss='mean_squared_error', metrics=['mse','mae'])
model.summary()
model.fit(x = [inputs, outputs], y = outputs, batch_size= 10, epochs=5, shuffle=True)


indices_t, pdbs_t, length_aas_t, pdb_aas_t, q8s_t, msas_t = load_pickle("../data/test.pkl")
test_inputs = np.array(produce_inputs(pdb_aas_t , q8s_t, msas_t, output_shape))
test_output = [np.ones((length_aas_t[i], length_aas_t[i])) for i in range(6)]
test_output = np.array(produce_outputs(test_output, max_input_length))


result_t = model.predict(x = [test_inputs, test_output], batch_size=1)

#prepare output data for submission to Kaggle
nums = [0,1,2,3,4,5]

results = {"Id":[], "Predicted": []}


for pname,seq_length,index in zip(pdbs_t, length_aas_t, nums):
  visited = set()

  for i in range(seq_length):
    for j in range(seq_length):

      if (i,j) in visited:
        continue

      results["Id"].append("{}_d_{}_{}".format(pname, i + 1, j + 1))
      results["Predicted"].append(result_t[index][i][j])
      visited.add((i,j))

with open('distance.pickle', 'wb') as handle:
    pickle.dump(results, handle)
