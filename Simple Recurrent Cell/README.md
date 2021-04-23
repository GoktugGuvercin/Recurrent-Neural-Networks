
# Simple Recurrent Neural Network


This small project aims to provide important insights about Simple RNN Cell used in Recurrent Neural Networks. How this cell operates 
and what inputs it takes as argument are essential assets that we need to know in order to get clear understanding about working structure 
of a recurrent layer and to easily deploy it in further projects. Totally two source codes are accommodated in this project:

- simpleRNN_io.py
- manual_rnn.py

## Source Files


- First python script describes how a simple recurrent network is configured by explaining each argument that it has to take as input over Tensorflow's front-end API Keras. The network that it instantiates process randomly-generated input sequence, whose time-step length is fixed to 3.

- Input sequence, cell state, the principle of shared parameters, and how these parameters are used are quite important concepts that we need to understand. Hence, the recurrent network instantiated in previous file is manually coded in this second python script. While doing this, placeholders from TF-V1 were used, since they enable us to analyze the inside of recurrent network in deep and obvious way.

## Used Libraries

* Tensorflow
* Keras
* Numpy
