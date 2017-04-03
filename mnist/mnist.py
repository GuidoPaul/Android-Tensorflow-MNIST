#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: mnist.py

import argparse
import os
import sys
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.tools.freeze_graph import freeze_graph

FLAGS = None

# Parameters
learning_rate = 0.001
epochs = 10
batch_size = 128
dropout = 0.75
display_every_n = 20
save_every_n = 200

# Number of samples to calculate validation and accuracy
# Decrease this if you're running out of memory to calculate accuracy
# test_valid_size = 256


def build_inference(x, keep_prob=None):
    def weight(shape):
        return tf.Variable(
            tf.truncated_normal(shape, stddev=0.1), name='weight')

    def bias(shape):
        return tf.Variable(tf.random_normal(shape), name='bias')

    def conv2d(x, W, b, strides=1):
        x = tf.nn.conv2d(
            x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b, name='convolution')
        return tf.nn.relu(x)

    def maxpool2d(x, k=2):
        return tf.nn.max_pool(
            x,
            ksize=[1, k, k, 1],
            strides=[1, k, k, 1],
            padding='SAME',
            name='pooling')

    with tf.name_scope('convolution_1'):
        # Layer 1 - 28*28*1 to 14*14*32
        conv1 = conv2d(x, weight([5, 5, 1, 32]), bias([32]))
        conv1 = maxpool2d(conv1, k=2)
    with tf.name_scope('convolution_2'):
        # Layer 2 - 14*14*32 to 7*7*64
        conv2 = conv2d(conv1, weight([5, 5, 32, 64]), bias([64]))
        conv2 = maxpool2d(conv2, k=2)
    with tf.name_scope('fully_connection_1'):
        # Fully connected layer - 7*7*64 to 1024
        fc1 = tf.reshape(conv2, [-1, 7 * 7 * 64])
        fc1 = tf.add(tf.matmul(fc1, weight([7 * 7 * 64, 1024])), bias([1024]))
        fc1 = tf.nn.relu(fc1)
        if (keep_prob is not None):
            fc1 = tf.nn.dropout(fc1, keep_prob)
    with tf.name_scope('readout'):
        # Output Layer - class prediction - 1024 to 10
        out = tf.add(tf.matmul(fc1, weight([1024, 10])), bias([10]))
        return tf.identity(out, name='output')


def build_cost(labels, logits):
    with tf.name_scope('cost'):
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='loss')
        cost = tf.reduce_mean(loss, name='cost')
        tf.summary.scalar('cost', cost)
    return cost


def build_accuracy(labels, logits):
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def build_train(loss, learning_rate):
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return optimizer


def main(_):
    # Setup the directory
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    if tf.gfile.Exists(FLAGS.ckpt_dir):
        tf.gfile.DeleteRecursively(FLAGS.ckpt_dir)
    tf.gfile.MakeDirs(FLAGS.ckpt_dir)
    if tf.gfile.Exists(FLAGS.model_dir):
        tf.gfile.DeleteRecursively(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.model_dir)

    # Read mnist data
    mnist_data = input_data.read_data_sets(
        FLAGS.data_dir, one_hot=True, reshape=False)

    with tf.Graph().as_default():
        # Declare placeholders we'll feed into the graph
        with tf.name_scope('inputs'):
            inputs = tf.placeholder(
                tf.float32, [None, 28, 28, 1], name='inputs')
        with tf.name_scope('targets'):
            targets = tf.placeholder(tf.float32, [None, 10], name='targets')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Build graph
        inference = build_inference(inputs, keep_prob)
        cost = build_cost(targets, inference)
        accuracy = build_accuracy(targets, inference)
        train = build_train(cost, learning_rate)

        merged = tf.summary.merge_all()

        # For writing training checkpoints.
        saver = tf.train.Saver(max_to_keep=1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_summaries_dir = os.path.join(FLAGS.summaries_dir, 'train')
            test_summaries_dir = os.path.join(FLAGS.summaries_dir, 'test')
            train_writer = tf.summary.FileWriter(train_summaries_dir,
                                                 sess.graph)
            test_writer = tf.summary.FileWriter(test_summaries_dir)

            print('Start training...')
            start = time.time()

            n_batches = mnist_data.train.num_examples // batch_size
            iterations = n_batches * epochs

            for epoch in range(epochs):
                for batch in range(n_batches):
                    iteration = epoch * n_batches + batch + 1
                    batch_x, batch_y = mnist_data.train.next_batch(batch_size)
                    _, summary = sess.run(
                        [train, merged],
                        feed_dict={
                            inputs: batch_x,
                            targets: batch_y,
                            keep_prob: dropout
                        })
                    train_writer.add_summary(summary, iteration)

                    if (iteration % display_every_n == 0):
                        # Calculate batch train loss and accuracy
                        loss, acc = sess.run(
                            [cost, accuracy],
                            feed_dict={
                                inputs: batch_x,
                                targets: batch_y,
                                keep_prob: 1.
                            })
                        print(
                            'Iteration {}/{} - Loss: {:>.4f} Training Accuracy: {:.4f}'.
                            format(iteration, iterations, loss, acc))

                    if (iteration % save_every_n == 0) or (
                            iteration == iterations):
                        # Calculate batch validation loss and accuracy
                        loss, valid_acc, summary = sess.run(
                            [cost, accuracy, merged],
                            feed_dict={
                                inputs: mnist_data.validation.images,
                                targets: mnist_data.validation.labels,
                                keep_prob: 1.
                            })
                        test_writer.add_summary(summary, iteration)
                        print(
                            'Epoch {}/{}, Iteration {}/{} - Loss: {:>.4f} Validation Accuracy: {:.4f}'.
                            format(epoch + 1, epochs, iteration, iterations,
                                   loss, valid_acc))
                        ckpt = saver.save(
                            sess,
                            os.path.join(FLAGS.ckpt_dir,
                                         "ckpt_i{}_{:.3f}.ckpt".format(
                                             iteration, loss)))
                        # global_step=iteration)
                        print('Save checkpoint: {}'.format(ckpt))

            elapsed_time = time.time() - start
            print('done!')
            print('Total time: {:.2f} [sec]'.format(elapsed_time))

            # Calculate Test Accuracy
            test_acc = sess.run(
                accuracy,
                feed_dict={
                    inputs: mnist_data.test.images,
                    targets: mnist_data.test.labels,
                    keep_prob: 1.
                })
            print('Testing Accuracy: {}'.format(test_acc))

            # Write graph.
            tf.train.write_graph(
                sess.graph.as_graph_def(),
                FLAGS.model_dir,
                FLAGS.model_name + '.pb',
                as_text=False)
            tf.train.write_graph(
                sess.graph.as_graph_def(),
                FLAGS.model_dir,
                FLAGS.model_name + '.pb.txt',
                as_text=True)

    # Freeze graph.
    freeze_graph(
        input_graph=os.path.join(FLAGS.model_dir,
                                 FLAGS.model_name + '.pb.txt'),
        input_saver='',
        input_binary=False,
        input_checkpoint=ckpt,
        output_node_names='readout/output',
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph=os.path.join(FLAGS.model_dir,
                                  '%s_frozen.pb' % FLAGS.model_name),
        clear_devices=False,
        initializer_nodes='')


if __name__ == '__main__':
    pwd = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.path.join(pwd, 'MNIST_data'),
        help='MNIST data directory.')
    parser.add_argument(
        '--model_dir',
        type=str,
        default=os.path.join(pwd, 'model'),
        help='Model directory.')
    parser.add_argument(
        '--model_name', type=str, default='mnist_graph', help='Model name.')
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default=os.path.join(pwd, 'ckpt'),
        help='Checkpoints directory.')
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default=os.path.join(pwd, 'log'),
        help='Where to save summary logs for TensorBoard.')
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
