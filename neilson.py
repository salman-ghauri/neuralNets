from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import  numpy as np
import tensorflow.contrib.slim as slim
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='5'

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

MEAN = np.mean(mnist.train.images)
STD = np.std(mnist.train.images)
model_save_name='checkpoint'

def reshape_data(images):
    norm = (images-MEAN)/STD
    reshaped = np.reshape(norm, [-1, 28, 28, 1])
    return reshaped

def nelnet(inputs, is_training, scope="NielsenNet"):
    with tf.variable_scope(scope):
        net = slim.conv2d(inputs, 20, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, 2, 2, scope='max1')

        net = slim.conv2d(net, 40, [5, 5], padding="VALID", scope='conv2')
        net = slim.max_pool2d(net, 2, 2, scope='max2')

        net = tf.reshape(net, [-1, 5*5*40], name='rehape_conv2')

        net = slim.fully_connected(net, 1000, scope='fc1')
        net = slim.dropout(net, is_training=is_training, scope='do1')

        net = slim.fully_connected(net, 1000, scope='fc2')
        net = slim.dropout(net, is_training=is_training, scope='do2')

        net = slim.fully_connected(net, 10, scope='fc3')
        net = slim.dropout(net, is_training=is_training, scope='do3')

        return net


with tf.Session() as sess:
    # x = slim.model_variable()
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="input")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="label")
    is_training = tf.placeholder(tf.bool, name="is_training")

    logits = nelnet(x, is_training)

    cross_entropy = slim.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=y, name="softmax_prediction")

    cost = tf.reduce_mean(cross_entropy, name="loss")
    optimizer = tf.train.MomentumOptimizer(0.01, 0.5).minimize(cost)

    correct_pred = tf.equal(tf.argmax(logits, 1, name="logit_max"), 
                            tf.argmax(y, 1, name="actual_max"),
                            name="prediction_actual")
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), 
                            name="accuracy_check")

    # To monitor our progress using tensorboard, create two summary operations
    # to track the loss and the accuracy
    cost_summary = tf.summary.scalar("cost", cost)
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    sess.run(tf.global_variables_initializer())
    # pointer to saving model
    train_writer = tf.summary.FileWriter(model_save_name, sess.graph)
    saver = tf.train.Saver(max_to_keep=4)

    # validation data for evaluation
    eval = {
        x: reshape_data(mnist.validation.images),
        y: mnist.validation.labels,
        is_training: False
    }

    # start model training
    for i in range(100000):
        images, labels = mnist.train.next_batch(100)
        train = {
            x: reshape_data(images),
            y: labels,
            is_training: True
        }
        summary, _ = sess.run([cost_summary, optimizer], feed_dict=train)
        train_writer.add_summary(summary, i)

        if i % 1000 == 0:
            # print summary for model after 1000 steps
            summary, acc = sess.run([accuracy_summary, accuracy], feed_dict=eval)
            print("Step: %5d, Validation Accuracy = %5.2f%%" % (i, acc * 100))

    # Print out the accuracy on test dataset:
    test_data = {
        x: reshape_data(mnist.test.images),
        y: mnist.test.labels,
        is_training: False
    }
    acc = sess.run(accuracy, feed_dict=test_data)
    print("Test Accuracy = %5.2f%%" % (100 * acc))
    saver.save(sess, model_save_name+"/model.ckpt")
