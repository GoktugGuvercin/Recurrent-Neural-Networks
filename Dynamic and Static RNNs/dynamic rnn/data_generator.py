
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence


class ImdbBatchGenerator(Sequence):

    def __init__(self, reviews, labels, batch_size, shuffle=True):

        """
        * The samples and their corresponding labels are passed to constructor in order to take small batches of
          samples from entire data set during batch construction.
        * Batch size should be set to 1. Otherwise, more than one movie review with distinct length are attempted to
          be put in a tensor, which results in error.

        Parameters:
        -----------
        reviews: movie reviews in a numpy array of lists
        labels: target score for each movie review in numpy array
        batch_size: batch size
        shuffle: Option to shuffle order of samples in each epoch
        """

        self.x_set = reviews
        self.y_set = labels
        self.shuffle = shuffle

        self.size = reviews.shape[0]
        self.batch_size = batch_size
        self.sample_indices = np.arange(self.size)
        self.on_epoch_end()

    def __getitem__(self, index):

        """
        * During training, __getitem__() method is invoked many times. Each call provides next batch of the dataset
          for the model.
        * Batches are encoded with index values ranging from 0 to __len()__ - 1.

        Parameters:
        -----------
        index: the number of next batch to be provided for training the model
        :return: one batch of samples in tensor or numpy format;
                 samples and labels are returned in tuple format
                 if samples are images, one batch is of 4D (batch, 3D image)
                 if samples are text sequences, one batch is of 2D/3D (batch, sequence of numbers/vectors)
        """

        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        indices = self.sample_indices[start:end]

        return self.__construct_batch(indices)

    def __len__(self):

        """
        It computes the number of batches per epoch, return it.
        """

        return math.ceil(self.size / self.batch_size)

    def __construct_batch(self, indices):

        """
        * Each movie review is actually a list of unique numbers. Some of these reviews, actually only one, for each
          batch is randomly chosen, and its indices are passed to this function.

        * Due to varied length of reviews, more than one sample cannot be put in a tensor; tensors are rigid and
          concrete structures with particular shape. And, since we do not prefer to pad or truncate these reviews,
          each batch will be organized to contain only one review. In other words, stochastic approach will be followed.

        * Since batch size is assumed to be equal to 1, the parameter "indices" is nothing but a list of merely one
          number.

        * x_set and y_set are numpy arrays of movie review lists, but we need to construct a batch as ND numpy array or
          a tensor. In other words; the following code block unfortunately does not work. Instead, each element (review)
          is individually accessed and stored in outer list, and then converted to array or tensor. Hence, iteration
          is preferred.

          x_batch = np.array(self.x_set[indices])
          y_batch = np.array(self.y_set[indices])

        Parameters:
        -----------
        indices: A list of indices of randomly chosen samples for next batch
        :return: one batch of samples (actually one sample), 2D numpy array
        """

        x_batch, y_batch = [], []

        for i in indices:
            sample, label = self.x_set[i], self.y_set[i]
            x_batch.append(sample)
            y_batch.append(label)

        return np.array(x_batch), np.array(y_batch)

    def on_epoch_end(self):

        """
        * This function is invoked after each epoch to shuffle sample indices. In that way, different order of samples
          is handled on each epoch.
        """
        
        if self.shuffle:
            np.random.shuffle(self.sample_indices)

