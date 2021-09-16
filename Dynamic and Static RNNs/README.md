
# Dynamic and Static Architectures for Recurrent Networks

Recurrent Neural Networks are the most powerful tool that enables us to proceed and interpret any type of sequence data. Music generation, semantic analysis of verbal sentences, information extraction from human DNA and RNA, speech recognition are only a few of tasks on which we can efficiently use and deploy recurrent networks.

All recurrent cells, which are consequtively associated to construct one recurrent layer, utilize same weight matrix, and perform same mathematical computation, but each of them is fed by different part of input sequence. In this case, defining individual computative unit for each recurrent cell is regarded as meaningless; instead solely one unit is instantiated, and it is iterated through time-steps of input sequence, which is called dynamic RNN. This can be regarded as if passing elements of an array to a function, and accumulating returned values in another list to construct output sequence. This iterative behavior is distinctly performed for each batch. In other words, the length of input sequences in one batch is set to termination condition of loop for dynamic RNN, and different loop is initiated for each batch. This automatically obliges all input sequences in one batch to be of same length, but not for the ones from different batches. Stochastic model 
training (batch size = 1) enables us to operate on varied length of input sequences without padding or truncating them.

<p align="center">
  <img src="https://github.com/GoktugGuvercin/Recurrent-Neural-Networks/blob/main/Dynamic%20and%20Static%20RNNs/Dynamic%20and%20Static%20RNNs.png" />
</p>

