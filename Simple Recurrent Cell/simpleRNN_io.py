
"""
>> In this code, it is assumed that we have sentiment-text classification problem,
   and depending on it, RNN layer is configured to process the sentences.

>> This simple code aims to describe how parameters in simple RNN layer are used
   so that given input sequences can be processed by its simple cells.

>> In this description, the parameter matrices that simple RNN layers generally
  deploy are mentioned. To get deeper understanding about how these matrices are used
  you can check the figures in "README.pdf" and examine how I code an RNN layer by hand
  in "manual_rnn.py".

"""

import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN


"""
Description of Input:
---------------------
* Input: (1 sentence x 3 words x 2 features)

* There is only 1 sample in the batch. 
* Each sentence refers to 3x2 matrix in 3D tensor. 
* A sentence is composed of 3 words, each of which is represented by a vector of 2 features. 
* Definition of feature vectors for the words can be performed in two different ways:
   + Learning it from word embedding layer
   + One-hot encoding vector

* We assume that we have already learnt feature vectors of words from embedding layers. """

input = tf.random.normal((1, 3, 2))


"""

Description of "input_shape" parameter in RNN Layer:
----------------------------------------------------
* input_shape: (time-steps, features)
  
* "time-steps" determines how many number of simple cells the layer will be accommodating, while features denote the
  size of vectors in input sequence. By setting "input_shape" to (3,2), recurrent layer is configured to be able to
  process input sequences of just 3 words, each of which is represented by a vector of 2 components. 

* "features" also refer to the number of columns in input parameter matrix Wx. In fact, how many number of columns will
  be placed in this matrix is decided by this argument. 


Description of "units" parameter in RNN Layer:
----------------------------------------------
* units: integer

* Each recurrent cell in an RNN layer has state information. 
* The parameter "units" determines the size of that state vector for all cells in the layer.

* The number of rows and columns in state parameter matrix Wa is set to "units"
* The number of rows in input parameter matrix Wx is set to "units"

"""

layer = SimpleRNN(units=4, input_shape=(3, 2))
output = layer(input)


"""
* What it is produced as an output is one feature vector of size 4 to represent entire sentence. This feature vector is
  produced by last cell in RNN layer. Other cells are configured not to generate any output. This is controlled by the
  parameter "return_sequences" in layer.
  
* This feature vector can be passed to a dense layer to perform sentiment text classification.
"""

print("Output: ", output)