import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor.signal.pool import pool_2d
from utils import dropout_layer

class SoftmaxLayer(object):
    def __init__(self, n_in, n_out, dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.dropout = dropout

        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX), name='w', borrow=True
        )
        self.b = theano.shared(
            np.zeros((n_out, ), dtype=theano.config.floatX), name='b', borrow=True
        )
        self.params = [self.w, self.b]

    def set_input(self, input, input_dropout, mini_batch_size):
        self.input = input.reshape((mini_batch_size, self.n_in))
        self.output = T.nnet.softmax(
            (1-self.dropout) * T.dot(self.input, self.w) + self.b
        )
        self.y_out = T.argmax(self.output, axis=1)
        self.input_dropout = dropout_layer(
            input_dropout.reshape((mini_batch_size, self.n_in)), self.dropout
        )
        self.output_dropout = T.nnet.softmax(
            T.dot(self.input_dropout, self.w) + self.b
        )


    def cost(self, net):
        """Log likelihood cost"""
        return -T.mean(T.log(self.output_dropout)\
                    [T.arange(net.y.shape[0]), net.y])

    
    def accuracy(self, y):
        """Accuracy for a mini batch"""
        return T.mean(T.eq(y, self.y_out))


class ConvPoolLayer(object):

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2), activation_fn=sigmoid):
        """
        Args:
            filter_shape: tuple of 4 length -> number of fiters * number of input feature maps * filter height * filter width
            image_shape: tuple of 4 length -> mini batch size * number of input feature maps * image height * image width
            poolsize: tuple of length 2 -> y and x pooling size
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn

        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1/n_out), size=filter_shape),
                dtype=theano.config.floatX
            ), borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1., size=(filter_shape[0],)), dtype=theano.config.floatX
            ), borrow=True)

        self.params = [self.w, self.b]


    def set_input(self, input, input_dropout, mini_batch_size):
        self.input = input.reshape(self.image_shape)
        conv_out = T.nnet.conv2d(
            input=self.input, filters=self.w, image_shape=self.image_shape, 
            filter_shape=self.filter_shape
        )
        pooled_out = pool_2d(
            conv_out, ws=self.poolsize, ignore_border=True
        )
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        )
        # no dropout in conv
        self.output_dropout = self.output


class FullyConnectedLayer:
    """An implementation followed from Neilson's Book"""
    def __init__(self, n_in, n_out, activation_fn=sigmoid, dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.dropout = dropout

        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0.0, scale=np.sqrt(1./n_out), 
                size=(n_in, n_out)), dtype=theano.config.floatX
            ), name='w', borrow=True
        )
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0.0, scale=1., size=(n_out, )), 
                dtype=theano.config.floatX
            ), name='b', borrow=True
        )
        self.params = [self.w, self.b]


    def set_input(self, input, input_dropout, mini_batch_size):
        # for evaluation
        self.input = input.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1 - self.dropout) * T.dot(self.input, self.w) + self.b
        )
        self.y_out = T.argmax(self.output, axis=1)
        # for training the model
        print(mini_batch_size, self.n_in, self.dropout)
        self.input_dropout = dropout_layer(
            input_dropout.reshape((mini_batch_size, self.n_in)), self.dropout
        )
        self.output_dropout = self.activation_fn(
            T.dot(self.input_dropout, self.w) + self.b
        )


    def accuracy(self, y):
        """Accuracy for a mini batch"""
        return T.mean(T.eq(y, self.y_out))
