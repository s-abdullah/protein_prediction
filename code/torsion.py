#!/usr/bin/env python
# coding: utf-8
from helper import *
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing import sequence


import tensorflow as tf
import pickle
import pandas as pd
import numpy as np

from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import LeakyReLU

from keras.layers import RNN
from keras.layers import InputLayer
from keras.layers import Input, Dense, Lambda, Reshape
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




def p(matric):
    # multiply by 180
    return tf.math.scalar_mul(180, matric)




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
  


# convert the 360.0 to 0.0
for y in range(len(psis)):
    for x in range(len(psis[y])):
        if psis[y][x] == 360.0:
            psis[y][x] = 0.000001
            
for y in range(len(phis)):
    for x in range(len(phis[y])):
        if phis[y][x] == 360.0:
            phis[y][x] = 0.000001
    
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



# changing back to list representation
# pdbs_v = pdbs[val].tolist()
# length_aas_v = length_aas[val].tolist()
# pdb_aas_v = pdb_aas[val].tolist()
# q8s_v = q8s[val].tolist()
# dcalphas_v = dcalphas[val].tolist()
# psis_v = psis[val].tolist()
# phis_v = phis[val].tolist()
# msas_v = msas[val].tolist()

pdbs = pdbs[train].tolist()
length_aas = length_aas[train].tolist()
pdb_aas = pdb_aas[train].tolist()
q8s = q8s[train].tolist()
dcalphas = dcalphas[train].tolist()
psis = psis[train].tolist()
phis = phis[train].tolist()
msas = msas[train].tolist()



sequence_length = 384 #max sequence length


outputs = np.array(tor_outputs(phis,psis ,sequence_length))
inputs = np.array(produce_inputs(pdb_aas, q8s, msas, sequence_length))





max_input_length = 384
output_shape = 384




out_inputs = Input(shape=(2, output_shape), name = "outputs") 
main_input = Input(shape=(52, output_shape), name='main_input')

l = Bidirectional(LSTM(400, return_sequences=False), name = "Bidirectional_LSTM3")(main_input)

l = Dense(400, activation='tanh', name = "dense1")(l)
l = Dense(400, activation='tanh', name = "dense2")(l)
l = Dense(400, activation='tanh', name = "dense3")(l)



l = Dense(output_shape*2, activation='tanh', name = "dense4")(l)
l = Reshape((2, output_shape,1), name = "reshape")(l)


l = Conv2D(10, 5,  padding='same')(l)

l = Conv2D(20, 5,padding='same')(l)
l = Dropout(0.25)(l)

l = Conv2D(30, 5,padding='same')(l)


l = Conv2D(20, 5,padding='same')(l)

l = Conv2D(1, 5, padding='same')(l)


l = Reshape((2, output_shape), name = "reshape2")(l)



# multiplying the out_inputs with the g cunction
mask = Lambda(g, name = "conversion")(out_inputs)

# multiplying the out_inputs with the m cunction
l = Lambda(m, name = "multiply")([l,mask])

# multiply by 180 degres
l = Lambda(p, name="degrees")(l)




# the Input layer and three Dense layers
model = Model(inputs=[main_input,out_inputs], outputs=[l])




model.compile(optimizer='adamax', loss='mean_squared_error', metrics=['mse','mae'])




model.summary()




model.fit(x = [inputs, outputs], y = outputs, batch_size= 30, epochs=20, shuffle=True)

# RUNNING VALIDATION THROUGH MODEL

# In[100]:


############### VALIDATION ################
# angle_scale = 180 / np.pi

# val_inputs = np.array(produce_inputs(pdb_aas_v , q8s_v, msas_v, output_shape))
# val_output1 = [(( [1]*length_aas_v[i])) for i in range(len(pdbs_v))]
# val_output2 = [(( [1]*length_aas_v[i])) for i in range(len(pdbs_v))]
# val_output = np.array(tor_outputs(val_output1, val_output2, max_input_length))
# # print(val_output.shape)

# result_v = model.predict(x = [val_inputs, val_output], batch_size=30)

# padded_angles_v = np.array(tor_outputs(phis_v, psis_v, 384))

# filtered_length_v = []
# for element in length_aas_v:
#     if element > max_input_length:
#       continue
#     filtered_length_v.append(element)

# padded_angles_v = padded_angles_v.astype(np.float32)

# phi, psi = result_v[:,0, :], result_v[:, 1, :]

# loss_phi_batch = rmsd_torsion_angle(phi, padded_angles_v[:, 0, :], np.array(filtered_length_v))
# loss_psi_batch = rmsd_torsion_angle(psi, padded_angles_v[:, 1, :], np.array(filtered_length_v))
# loss_phi = tf.reduce_mean(loss_phi_batch)  
# loss_psi = tf.reduce_mean(loss_psi_batch)



# with tf.Session() as sess:
#     # This tells Tensorflow we woud like to evaluate the Tensor `result`. 
#     outputpsi = sess.run(loss_psi)
#     outputphi = sess.run(loss_phi)
    
#     print(outputpsi, outputphi)



indices_t, pdbs_t, length_aas_t, pdb_aas_t, q8s_t, msas_t = load_pickle("../data/test.pkl")
test_inputs = np.array(produce_inputs(pdb_aas_t , q8s_t, msas_t, output_shape))

test_output1 = [(( [1]*length_aas_t[i])) for i in range(len(pdbs_t))]
test_output2 = [(( [1]*length_aas_t[i])) for i in range(len(pdbs_t))]
test_output = np.array(tor_outputs(test_output1, test_output2, max_input_length))


result_t = model.predict(x = [test_inputs, test_output], batch_size=1)



nums = [0,1,2,3,4,5]


results = {"Id":[], "Predicted": []}


for pname,seq_length,index in zip(pdbs_t, length_aas_t, nums):
  visited = set()

  for i in range(seq_length):
      if (i) in visited:
        continue

      results["Id"].append("{}_psi_{}".format(pname, i + 1))
      results["Predicted"].append(result_t[index][1][i])

      results["Id"].append("{}_phi_{}".format(pname, i + 1))
      results["Predicted"].append(result_t[index][0][i])
      visited.add((i))


with open('angles.pickle', 'wb') as handle:
    pickle.dump(results, handle)


