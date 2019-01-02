import numpy as np
import theano
import theano.tensor as T
from utils import *
from layers import *

class Network:
    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")

        init_layer = self.layers[0]
        init_layer.set_input(self.x, self.x, self.mini_batch_size)

        for j in xrange(1, len(self.layers)):
            prev_layer, cur_layer = self.layers[j-1], self.layers[j]
            cur_layer.set_input(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size
            )
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta, validation_data, test_data, lmbda=0.0):
        """Network training using mini batch stochastic gradient descent"""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batched = size(test_data)/mini_batch_size

        # resgularized cost function
        l2_norm_squared = sum(
            [(layer.w**2).sum() for layer in self.layers]
        )
        cost = self.layers[-1].cost(self)+\
            0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad) \
                    for param, grad in zip(self.params, grads)]

        i = T.lscalar()
        # mini batch train
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x: training_x[i*self.mini_batch_size:(i+1)*self.mini_batch_size],
                self.y: training_y[i*self.mini_batch_size:(i+1)*self.mini_batch_size]
            }
        )
        # mini batch validation accuracy
        validation_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size:(i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size:(i+1)*self.mini_batch_size]
            }
        )
        # mini batch test accuracy
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size:(i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size:(i+1)*self.mini_batch_size]
            }
        )
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x: test_x[i*self.mini_batch_size:(i+1)*self.mini_batch_size]
            }
        )

        best_validation_accuracy = 0.0
        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches*epoch*minibatch_index
                if iteration:
                    print("Training mini batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration+1):
                    validation_accuracy = np.mean(
                        [validation_mb_accuracy(j) for j in xrange(num_validation_batches)]
                    )
                    print("Epoch {0}: validation accuracy {1:.2%}".format(epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in xrange(num_test_batched)]
                            )
                            print("The corresponding test accuracy in {0:.2%}".format(test_accuracy))

        print("Fininshed training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

if __name__ == "__main__":
    training_data, validation_data, test_data = \
                            load_data_shared()
    batch_size = 10
    net = Network([
            ConvPoolLayer(image_shape=(batch_size, 1, 28, 28), 
                        filter_shape=(20, 1, 5, 5), 
                        poolsize=(2, 2)),
            FullyConnectedLayer(n_in=20*12*12, n_out=100),
            SoftmaxLayer(n_in=100, n_out=10)
        ], batch_size)

    net.SGD(training_data, 60, batch_size, 0.1, validation_data, test_data) 