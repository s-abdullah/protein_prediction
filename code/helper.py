import pickle

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from geom_ops import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import to_categorical


#compute error for angles
def rmsd_torsion_angle(angles1, angles2, batch_seqlen, name=None):
    """
    angles1, angles2: [BATCH_SIZE, MAX_LEN]
    batch_seqlen: [BATCH_SIZE,]
    """

    weights = np.zeros(shape=angles1.shape, dtype=np.float32)
    for i, length in enumerate(batch_seqlen):
        weights[i, :length] = 1.0

    with tf.name_scope(name, 'RMSD_torsion', [angles1, angles2, weights]) as scope:
        angles1 = tf.convert_to_tensor(angles1, name='angles1')
        angles2 = tf.convert_to_tensor(angles2, name='angles2')
        weights = tf.convert_to_tensor(weights, name='weights')
        diffs = angles1 - angles2                      # [BATCH_SIZE, MAX_LEN]

        norms = reduce_l2_norm(diffs, reduction_indices=[1], weights=weights, name=scope) # [BATCH_SIZE]
        drmsd = norms / tf.sqrt(batch_seqlen.astype(np.float32))
        return drmsd  # [BATCH_SIZE,]


#prepare angles outputs for submission to kaggle
def tor_outputs(angles1, angles2, max_input_length):
    output1 = []
    for x in range(len(angles1)):
        if len(angles1[x]) > max_input_length:
            continue
        if len(angles2[x]) > max_input_length:
            continue

        extra_rows1 = max_input_length - len(angles1[x])
        extra_rows2 = max_input_length - len(angles2[x])

        rows_zeros1 = [0]*extra_rows1
        rows_zeros2 = [0]*extra_rows2

        angles1[x] += rows_zeros1
        angles2[x] += rows_zeros2

        output1.append([angles1[x],angles2[x]])
    return output1

def load_pickle(datafile):
    # load train data for a specific fold
    with open(datafile, 'rb') as f:
        data =  pickle.load(f)
    return data

#compute error for distance matrices
def drmsd_dist_matrix(mat1, mat2, batch_seqlen, name=None):
    """
    mat1, mat2: [BATCH_SIZE, MAX_LEN, MAX_LEN]
    batch_seqlen: [BATCH_SIZE,]
    """

    weights = np.zeros(shape=mat1.shape, dtype=np.float32)
    for i, length in enumerate(batch_seqlen):
        weights[i, :length, :length] = 1

    with tf.name_scope(name, 'dRMSD', [mat1, mat2, weights]) as scope:
        mat1 = tf.convert_to_tensor(mat1, name='mat1')
        mat2 = tf.convert_to_tensor(mat2, name='mat2')
        weights = tf.convert_to_tensor(weights, name='weights')
        diffs = mat1 - mat2                      # [BATCH_SIZE, MAX_LEN, MAX_LEN]
        #diffs = tf.transpose(diffs, [1,2,0])      # [MAX_LEN, MAX_LEN, BATCH_SIZE]
        #weights = tf.transpose(weights, [1,2,0])

        norms = reduce_l2_norm(diffs, reduction_indices=[1, 2], weights=weights, name=scope) # [BATCH_SIZE]
        drmsd = norms / batch_seqlen
        return drmsd  # [BATCH_SIZE,]


def reduce_l2_norm(input_tensor, reduction_indices=None, keep_dims=None, weights=None, epsilon=1e-12, name=None):
    """ Computes the (possibly weighted) L2 norm of a tensor along the dimensions given in reduction_indices.

    Args:
        input_tensor: [..., NUM_DIMENSIONS, ...]
        weights:      [..., NUM_DIMENSIONS, ...]

    Returns:
                      [..., ...]
    """
    # sqrt(sum(x_ij**2))
    with tf.name_scope(name, 'reduce_l2_norm', [input_tensor]) as scope:
        input_tensor = tf.convert_to_tensor(input_tensor, name='input_tensor')

        input_tensor_sq = tf.square(input_tensor)
        if weights is not None: input_tensor_sq = input_tensor_sq * weights

        return tf.sqrt(tf.maximum(tf.reduce_sum(input_tensor_sq, axis=reduction_indices, keepdims=keep_dims), epsilon), name=scope)


#code to produce Kaggle output (provided in class)
def produce_test_output(testfile, save_path):
    test_input = pd.read_csv(testfile, header=None)
    protein_names = np.array(test_input.iloc[:,1])
    protein_len = np.array(test_input.iloc[:,2])

    # concatenate all output to one-dimensional
    all_names = []
    for i, pname in enumerate(protein_names):


        length = protein_len[i]
        dist_names = ["{}_d_{}_{}".format(pname, i + 1, j + 1) for i in range(length) for
                j in range(length)]

        psi_names = ["{}_psi_{}".format(pname, i + 1) for i in range(length)]
        phi_names = ["{}_phi_{}".format(pname, i + 1) for i in range(length)]
        row_names = np.array(dist_names + psi_names + phi_names)
        all_names.append(row_names)

    all_names = np.concatenate(all_names)
    output = {"Id": all_names}
    return output
    #output = pd.DataFrame(output)
    #output.to_csv(save_path, index=False)



#perform masking operation for output (used when training)
def produce_outputs(dcalphas, max_input_length):
    outputs = []
    masks = []

    for element in dcalphas:
        if len(element) > max_input_length:
          continue

        extra_cols = max_input_length - element.shape[1]
        extra_rows = max_input_length - element.shape[0]

        cols_zeros = np.zeros((element.shape[0], extra_cols))
        rows_zeros = np.zeros((extra_rows, max_input_length))

        f_array = np.hstack((element, cols_zeros))
        f_array = np.vstack((f_array, rows_zeros))

        outputs.append(f_array)

    return outputs



#input preprocessing for running neural network
def produce_inputs(pdb_aas, q8s, msas, sequence_length):
    indexer_aa = Tokenizer(lower=False, filters='')
    indexer_q8 = Tokenizer(lower=False, filters='')

    aa_list = 'ACDEFGHIKLMNPQRSTVWXY'
    q8_list = 'GHITEBS-'

    n_values_aa = len(aa_list)+1
    n_values_q8 = len(q8_list)+1

    indexer_aa.fit_on_texts(list(aa_list))
    indexer_q8.fit_on_texts(list(q8_list))

    inputs = []

    for aa, q8, msa in zip(pdb_aas, q8s, msas):
        if len(aa)> sequence_length:
          continue


        v_aa = np.array(indexer_aa.texts_to_sequences(aa)).T
        result_aa = np.eye(n_values_aa)[v_aa] #seq_length * 21

        v_q8 = np.array(indexer_q8.texts_to_sequences(q8)).T
        result_q8 = np.eye(n_values_q8)[v_q8] #seq_length * 8

        result_q8 = result_q8[0]
        result_aa = result_aa[0]

        msa = np.array(msa).astype(np.float)
        #print(msa)

        input_train = np.vstack((result_aa.T, result_q8.T))
        input_train = np.vstack((input_train, msa))

        #pad the sequence
        input_train = sequence.pad_sequences(input_train, maxlen=sequence_length, padding = "post", dtype='float32')

        inputs.append(input_train) #current shape is 52,3

    return inputs
