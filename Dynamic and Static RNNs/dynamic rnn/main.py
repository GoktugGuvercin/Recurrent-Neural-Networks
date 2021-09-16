
"""
>> IMDB Dataset
>> Splitting reviews into positive and negative categories
>> Text Classification Problem
>> Single Label - Binary Classification
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense

from tensorflow.keras.datasets import imdb
from data_generator import ImdbBatchGenerator

"""
* Each sample in train and test data refers to particular movie review.

* Movie reviews are typical sentences composed of words, but each word is denoted with unique index number, 
  determined by its frequency of existence in the dataset. For practical and convenient use, reviews are
  organized as a list of indices.

* The labels are of binary values (0: negative review, 1: positive review).

* The parameter "num_words" is used to choose the reviews with the most frequent n words.
  num_words = 10000 --> all reviews composed of the most frequent 10 000 words are chosen for dataset.

"""

num_features = 10000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_features)

"""
* Dynamic RNN has iterative approach rather than constructing entire network, so only one recurrent cell per layer is
  instantiated, and then an iteration over that cell is performed by each batch returned by data generator. The
  length of iteration is set to the length of movie reviews in that batch. Since we opt for stochastic model training,
  one review per batch is applied.
  
* If we place more than one sample in one batch (setting batch size of data generator to any value larger than 1), we
  become encountered by two different problems:
  
  1) Sequences with different length cannot be situated in rigid structure (tensor)
  2) Even if we overcome first problem by using new-released flexible architectures of tensorflow, how many times the
     loop is iterated over the cell cannot be determined. 
"""

batch_generator = ImdbBatchGenerator(x_train, y_train, 1, True)

"""
* "Sequential" enables us to stack layers in linear order for the construction of target neural network which 
   processes input sequences, and classify them.

* Embedding layer defines a feature vector composed of 10,000 components for each word, and then compresses it
  into the one with merely 32 features. If "Flatten", "Dense", or "static RNN" is deployed just after Embedding 
  layer, sequence length should be specified for it. 

* Whether recurrent layer is constructed statically or dynamically is controlled by "unroll" parameter.
  unroll = True --> Static
  unroll = False --> Dynamic
"""

model = Sequential()
model.add(Embedding(num_features, 32))  # adding Embedding layer
model.add(LSTM(32, unroll=False))  # adding LSTM layer
model.add(Dense(1, activation="sigmoid"))  # adding Dense layer

"""
* Model is configured with
    >> Optimizer: Adaptive Momentum
    >> Loss Function: Binary Cross Entropy
    >> Evaluation Metric: Accuracy
    
* Validation split used in training of static RNN is not supported when batch generator is deployed. Hence,
  it is not set to specific value in this code block. Instead, second batch generator can be instantiated with 
  validation data, and passed to "validation_data" parameter of fit() function. 

* Total training takes 10 epochs over the batches of 1 sample. 
"""

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(batch_generator, epochs=10)
