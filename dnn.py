import pickle

import numpy as np
import tensorflow as tf

import utils

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './data', 'Directory for storing data')
flags.DEFINE_string('model_path', 'models/etl7.ckpt', 'Model Storing Directory')
flags.DEFINE_string('summaries_dir', '/tmp/ETL7', 'Summaries directory')
flags.DEFINE_integer('mini_batch_size', 100, 'Mini batch training data size')
flags.DEFINE_integer('step_num', 100, 'Training step number')
flags.DEFINE_integer('learning_rate', 2e-4, 'Learning Rate')


class Classifier(object):
    def __init__(self):
        self.sess = tf.Session()
        self._x = None
        self._y = None
        self._y_conv = None
        self._keep_prob = None
        self._train_step = None
        self._accuracy = None

    def __delete__(self, instance):
        self.sess.close()

    def create_network(self):
        with tf.name_scope('Input'):
            self._x = tf.placeholder(tf.float32, shape=[None, 1024], name='X')
            self._y = tf.placeholder(tf.float32, shape=[None, 46], name='Y')

        first_out_channel = 64
        second_out_channel = 64
        with tf.name_scope('LeNetConvPool_1'):
            input_image = tf.reshape(self._x, [-1, 32, 32, 1])
            out_image_layer1 = utils.le_net_conv_pool(input_image, 5, 1, first_out_channel)

        with tf.name_scope('LeNetConvPool_2'):
            out_image_layer2 = utils.le_net_conv_pool(out_image_layer1, 5, first_out_channel, second_out_channel)

        with tf.name_scope('FullConnect'):
            W_fc1 = utils.weight_variable([8 * 8 * second_out_channel, 1024], 'W_fc1')
            b_fc1 = utils.bias_variable([1024], 'b_fc1')
            h_pool2_flat = tf.reshape(out_image_layer2, [-1, 8 * 8 * second_out_channel])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        with tf.name_scope('Dropout'):
            self._keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, self._keep_prob)

        with tf.name_scope('ReadoutLayer'):
            W_fc2 = utils.weight_variable([1024, 46], 'W_fc2')
            b_fc2 = utils.bias_variable([46], 'b_fc2')
            self._y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        with tf.name_scope('Train'):
            cross_entropy = tf.reduce_mean(
                -tf.reduce_sum(self._y * tf.log(tf.clip_by_value(self._y_conv, 1e-10, 1.0)), reduction_indices=[1]))
            self._train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
            tf.scalar_summary('Cross Entropy', cross_entropy)

        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(self._y_conv, 1), tf.argmax(self._y, 1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('Accuracy', self._accuracy)

        self.sess.run(tf.initialize_all_variables())

    def training(self):
        with open('data/train.pickle', 'rb') as file:
            train_dataset = pickle.load(file)
        with open('data/test.pickle', 'rb') as file:
            test_dataset = pickle.load(file)
        train_x = train_dataset['data']
        train_y = train_dataset['label']
        test_x = test_dataset['data']
        test_y = test_dataset['label']
        train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', self.sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
        merged_summaries = tf.merge_all_summaries()

        step = 0
        for _ in range(FLAGS.step_num):
            perm = np.random.permutation(len(train_x))
            # train
            for i in range(0, len(train_x), FLAGS.mini_batch_size):
                step += 1
                indexes = perm[i:i + FLAGS.mini_batch_size]
                batch_x = train_x[indexes]
                batch_y = train_y[indexes]
                summary, _ = self.sess.run([merged_summaries, self._train_step],
                                           feed_dict={self._x: batch_x, self._y: batch_y, self._keep_prob: 0.5})
                train_writer.add_summary(summary, step)
            # test
            summary, accuracy = self.sess.run([merged_summaries, self._accuracy],
                                              feed_dict={self._x: test_x, self._y: test_y, self._keep_prob: 1})
            test_writer.add_summary(summary, step)
            print('step {}: {}'.format(step, accuracy))
        with tf.device('/cpu:0'):
            utils.save_model(self.sess, FLAGS.model_path)

        train_writer.close()
        test_writer.close()

    def evaluate(self):
        pass


def main():
    # Delete old training log
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'clear':
        if tf.gfile.Exists(FLAGS.summaries_dir):
            tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
        tf.gfile.MakeDirs(FLAGS.summaries_dir)

    classifier = Classifier()
    with tf.device('/cpu:0'):
        classifier.create_network()
    classifier.training()
    # utils.restore_model(classifier._sess, FLAGS.model_path)
    classifier.evaluate()


if __name__ == '__main__':
    main()
