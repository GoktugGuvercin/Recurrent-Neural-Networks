
# Dynamic and Static Architectures for Recurrent Networks

Recurrent Neural Networks are the most powerful tool that enables us to proceed and interpret any type of sequence data. Music generation, semantic analysis of verbal sentences, information extraction from human DNA and RNA, speech recognition are only a few of tasks on which we can efficiently use and deploy recurrent networks.

All recurrent cells, which are consequtively associated to construct one recurrent layer, utilize same weight matrix, and perform same mathematical computation, but each of them is fed by different part of input sequence. In this case, defining individual computative unit for each recurrent cell is regarded as meaningless; instead solely one unit is instantiated, and it is iterated through time-steps of input sequence, which is called dynamic RNN. This can be regarded as if passing elements of an array to a function, and accumulating returned values in another list to construct output sequence. This iterative behavior is distinctly performed for each batch. In other words, the length of input sequences in one batch is set to termination condition of loop for dynamic RNN, and different loop is initiated for each batch. This automatically obliges all input sequences in one batch to be of same length, but not for the ones from different batches. Stochastic model 
training (batch size = 1) enables us to operate on varied length of input sequences without padding or truncating them.

<p align="center">
  <img src="https://github.com/GoktugGuvercin/Recurrent-Neural-Networks/blob/main/Dynamic%20and%20Static%20RNNs/Dynamic%20and%20Static%20RNNs.png" />
</p>


## Comparison between Dynamic and Static RNNs


* Dynamic RNNs are deemed as more preferable and advantegous than Static ones in many aspects. First of all, static approach requires the construction of entire network, which results in high memory consumption, and even out of memory errors. On the other hand, dynamic RNNs can handle their job with one cell rather than deploying particular cell unit for each time-step of input sequences. This highly minimizes total memory consumption. 

* Secondly, static RNNs are built before being fed by input, and its architecture cannot be changed, is completely fixed. In fact, how many number of operation nodes will be accommodated in static RNNs is determined by taking into account the length of input sequences. Hence, variable-length sequences cannot be handled by static networks; they at first need to be padded to same length. However, deployment of iterative loop depending on the length of input sequences in a batch allows us to easily work on variable-sized sequences when batch size is set to 1. 

* The only difficulty with dynamic RNNs encountered by researchers and ML engineers in this point is time complexity. When input sequences are padded to same length, they can be vectorized, and fed to the network as a whole rather than one by one. Hence, static RNNs are quite faster than their dynamic version. 

## Source Files

- dynamic rnn/data_generator.py
- dynamic rnn/main.py
- static rnn/main.py

To illustrate the behavior and some implementation details of both RNN architectures, a small project on sentiment classification problem is done with the help of IMDB dataset. Main files are in charge of loading the dataset, constructing the network, and launching training process. The only difference between them is what value "unroll" parameter of recurrent layer is set to, which actually controls static or dynamic execution. Dynamic rnn folder contains extra one source file, called "data_generator.py". This source file helps to prepare custome batches and then serve for the network. The main reason why tranditional batch structure cannot be used is the fact that sequences are of different length. 

## Used Libraries

* Tensorflow
* Keras
* Numpy
