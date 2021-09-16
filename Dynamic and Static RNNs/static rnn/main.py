
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
from tensorflow.keras.preprocessing import sequence


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
seq_length = 400

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_features)

"""
* Movie reviews are of different length; they need to be either truncated or padded to particular time-steps, since
  static RNNs have fixed architecture, the number of recurrent cells in them is pre-determined and it does not change
  after the network is constructed. 

* The function "pad_sequences" pads/truncates input sequences shorter/longer than desired length 
* Which value we will use for padding and where we will position it (pre or post padding) are controlled by function 
  arguments. 
   
"""

x_train = sequence.pad_sequences(x_train, seq_length)
x_test = sequence.pad_sequences(x_test, seq_length)


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
model.add(Embedding(num_features, 32, input_length=seq_length))  # adding Embedding layer
model.add(LSTM(32, unroll=True))  # adding LSTM layer
model.add(Dense(1, activation="sigmoid"))  # adding Dense layer

"""
* Model is configured with
    >> Optimizer: Adaptive Momentum
    >> Loss Function: Binary Cross Entropy
    >> Evaluation Metric: Accuracy

* 20 percentage of train set is dedicated to validation process, and total training takes 10 epochs over the 
  batches of 128 samples. 
"""

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
