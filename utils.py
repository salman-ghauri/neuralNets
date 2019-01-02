import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor.signal.pool import pool_2d
import cPickle
import gzip

def load_data_shared(
    filename="/home/salman-macpak/work/extra/nnbook/data/mnist.pkl.gz"):
    with gzip.open(filename, 'rb') as f:
        training_data, validation_data, test_data = cPickle.load(f)
    def shared(data):
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True
        )
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True
        )
        return shared_x, T.cast(shared_y, "int32")
    return [
        shared(training_data),
        shared(validation_data),
        shared(test_data)
    ]


def size(data):
    """Return size of the dataset `data`"""
    return data[0].get_value(borrow=True).shape[0]


def dropout_layer(layer, dropout):
    """Apply dropout on a layer"""
    random_state = T.shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999)
    )
    mask = random_state.binomial(n=1, p=1-dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)


def linear(z): return z


def ReLU(z): return T.maximum(0.0, z)