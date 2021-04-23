
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # need to disable eager in TF2.x


input_size = 3  # the length of input feature vectors to simple RNN cells
output_size = 5   # the length of output feature vectors produced by simple RNN cells

"""
* When thinking in low level, the operations performed by neural networks are actually nothing but simple mathematical 
  computations. Hence, deep learning libraries such as tensorflow and pytorch utilize special data structure called 
  "computation graph" for the implementation of these networks.
  
* Computation graph defined for basic RNN layer reads input sequence via placeholders in this python script.

* In this point, we assume that input sequences would be composed of exactly 3 pieces like a sentence of just three 
  words, so RNN layer is implemented to have 3 cells. Each cell is supposed to read feature vector representing a piece 
  of input sequence; hence, for each of these cells, one placeholder in 2D (input-size, batch-size) is defined. 
  Since fixed  batch-size is not necessary except for some special conditions like "stateful concept", we left it with 
  None. 
"""

x0 = tf.compat.v1.placeholder(tf.float32, [input_size, None])
x1 = tf.compat.v1.placeholder(tf.float32, [input_size, None])
x2 = tf.compat.v1.placeholder(tf.float32, [input_size, None])

"""
* Network parameters are regularly modified and updated under the supervision of calculated loss value. TF recommends
  these parameters to be maintained by Variables; thus, bias term and parameter matrices are defined by TF Variable 
  class. 
  
* To figure out the shape of parameters clearly, you can check the figures in "README.pdf"
"""

Wx = tf.compat.v1.Variable(tf.random.normal(shape=[output_size, input_size], dtype=tf.float32))
Wa = tf.compat.v1.Variable(tf.random.normal(shape=[output_size, output_size], dtype=tf.float32))
b = tf.compat.v1.Variable(tf.random.normal(shape=[output_size, 1], dtype=tf.float32))

"""
* When we instantiate an RNN layer by using high-level libraries such as Keras, its corresponding computation graph is 
  automatically created. In this point, because we implement entire architecture manually, we have to specify the 
  operations of computation graph explicitly, as being done in the code below. 
"""

a0 = tf.tanh(tf.matmul(Wx, x0) + b)  # first RNN cell
a1 = tf.tanh(tf.matmul(Wx, x1) + tf.matmul(Wa, a0) + b)  # second RNN cell
a2 = tf.tanh(tf.matmul(Wx, x2) + tf.matmul(Wa, a1) + b)  # third RNN cell

"""
* All variables defined to instantiate model parameters are stored in a list. 
* To set these variables to pre-determined initial values in definition section, global variable initializer is used. 
* We do this initialization process inside the session. 
"""

init = tf.compat.v1.global_variables_initializer()


"""
* Each batch is dedicated to fill one placeholder of computation graph.
* While columns refer to the vectors of input sequence, the number of columns specifies how many vectors 
  for one placeholder exist in the batch. In other words, There are 4 vectors for each placeholder through
  the batches, and each input vector has 3 components. 

* The diagram below provides more understandable intuition. 


                 ---------         ---------         ---------
                 |       |         |       |         |       |
Three            |   1   |         |   2   |         |   3   |
Cells            |       |         |       |         |       |
                 ---------         ---------         ---------
                     ^                 ^                 ^
                     |                 |                 |
Batch 1 -->      [0, 1, 2]         [9, 8, 7]          [5, 4, 3]
Batch 2 -->      [3, 4, 5]         [0, 0, 0]          [9, 8, 7]
Batch 3 -->      [6, 7, 8]         [6, 5, 4]          [1, 2, 0]
Batch 4 -->      [9, 0, 1]         [3, 2, 1]          [6, 6, 6]
"""

x0_batch = np.array([[0, 3, 6, 9],
                     [1, 4, 7, 0],
                     [2, 5, 8, 1]])

x1_batch = np.array([[9, 0, 6, 3],
                     [8, 0, 5, 2],
                     [7, 0, 4, 1]])

x2_batch = np.array([[5, 9, 1, 6],
                     [4, 8, 2, 6],
                     [3, 7, 0, 6]])


"""
* First argument of session.run() is named as "fetches". It can be a single operation associated with a tensor or a list 
  of them. It is generally interpreted as one branch of a computation graph.
  
* A simple recurrent network consists of consecutively ordered recurrent cells in theoretical level. Each cell takes one
  input vector of features. Each input vector is processed by its corresponding cell; hence, the argument "fetches" 
  refers to the list of input vectors. On the other hand, the argument "feed_dict" is a dictionary, mapping graph 
  elements to the values.
  
* fetches: [timesteps, batch_size, features]
"""
with tf.compat.v1.Session() as session:
    init.run()  # initializing the variables
    a0_result, a1_result, a2_result = session.run([a0, a1, a2], feed_dict={x0: x0_batch, x1: x1_batch, x2: x2_batch})
    print(a0_result)


