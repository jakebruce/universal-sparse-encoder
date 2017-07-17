#===============================================================================
# MIT License
#
# Copyright (c) 2017 Jake Bruce
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#===============================================================================

import numpy      as np
import tensorflow as tf

#===============================================================================
# MODEL

class UniversalSparseEncoder:
    def __init__(self,
                 input_size,              # dimensionality of the raw input data
                 bits,                    # number of bits in the resulting encoder
                 sparsity,                # desired sparsity, e.g. 0.1
                 batch_size,              # number of input examples per training batch
                 sparsity_weight = 100.0, # how much to weigh the sparsity constraint vs recon error
                 regular_weight  = 1e-5,  # strength of weight decay regularization
                 learn_rate      = 0.005, # learning rate of AdamOptimizer
                 dropout_input   = 0.8,   # probability to keep input units during training
                 dropout_hidden  = 0.5,   # probability to keep hidden units during training
                 input_noise     = 0.25): # scale of gaussian noise added to input during training

        # initialize constant object parameters
        self.input_size      = input_size
        self.hidden_size     = bits
        self.sparsity        = sparsity
        self.batch_size      = batch_size
        self.sparsity_weight = sparsity_weight
        self.regular_weight  = regular_weight
        self.dropout_input   = dropout_input
        self.dropout_hidden  = dropout_hidden
        self.input_noise     = input_noise

        # input fed in at runtime
        self.input_p        = tf.placeholder(shape=[self.batch_size, self.input_size], dtype=tf.float32)
        self.input_keep_p   = tf.placeholder(shape=[], dtype=tf.float32)
        self.hidden_keep_p  = tf.placeholder(shape=[], dtype=tf.float32)
        self.input_noise_p  = tf.placeholder(shape=[], dtype=tf.float32)

        # trainable weights and biases
        self.hidden_weights = tf.get_variable("hw",  shape=[self.input_size +1, self.hidden_size])
        self.output_weights = tf.get_variable("ow",  shape=[self.hidden_size+1, self.input_size])

        # add gaussian noise
        gaussian_noise      = tf.truncated_normal([self.batch_size, self.input_size], 0, self.input_noise_p)
        self.noisy_input    = tf.nn.dropout(self.input_p + gaussian_noise, keep_prob=self.input_keep_p)

        # forward pass
        bias_constants      = tf.constant  (np.ones((self.batch_size, 1), dtype=np.float32))
        augment_input       = tf.concat    ([self.noisy_input, bias_constants], axis=1)
        self.hidden         = tf.sigmoid   (tf.matmul(augment_input, self.hidden_weights))
        dropped_hidden      = tf.nn.dropout(self.hidden, keep_prob=self.hidden_keep_p)
        augment_hidden      = tf.concat    ([dropped_hidden, bias_constants], axis=1)
        self.reconstruction = tf.matmul    (augment_hidden, self.output_weights)

        # sparsity constraints
        self.layer_sparsity = tf.reduce_sum(self.hidden, axis=1) / self.hidden_size
        self.batch_sparsity = tf.reduce_sum(self.hidden, axis=0) / self.batch_size

        # reconstruction and sparsity losses
        self.recon_loss     = tf.reduce_mean(tf.squared_difference(self.reconstruction, self.input_p))
        self.layer_loss     = tf.reduce_mean(tf.squared_difference(self.layer_sparsity, sparsity))*self.sparsity_weight
        self.batch_loss     = tf.reduce_mean(tf.squared_difference(self.batch_sparsity, sparsity))*self.sparsity_weight
        self.regul_loss     = tf.nn.l2_loss (self.hidden_weights) * self.regular_weight
        self.total_loss     = self.recon_loss + self.layer_loss + self.batch_loss + self.regul_loss

        # optimization
        self.learn_rate     = tf.Variable(learn_rate, dtype=tf.float32, trainable=False)
        self.train_step     = tf.train.AdamOptimizer(self.learn_rate).minimize(self.total_loss)

    #------------------------------------------------------------

    def train(self, sess, batch):
        feed_dict={self.input_p       : batch,
                   self.input_keep_p  : self.dropout_input,
                   self.hidden_keep_p : self.dropout_hidden,
                   self.input_noise_p : self.input_noise}
        return sess.run([self.train_step, self.total_loss], feed_dict=feed_dict)

    #------------------------------------------------------------

    def test(self, sess, batch, noisy=False):
        feed_dict={self.input_p       : batch,
                   self.input_keep_p  : self.dropout_input  if noisy else 1,
                   self.hidden_keep_p : self.dropout_hidden if noisy else 1,
                   self.input_noise_p : self.input_noise    if noisy else 0}
        return sess.run([self.noisy_input, self.reconstruction, self.total_loss], feed_dict=feed_dict)

    #------------------------------------------------------------

    def encode(self, sess, batch):
        feed_dict={self.input_p       : batch,
                   self.input_keep_p  : 1,
                   self.hidden_keep_p : 1,
                   self.input_noise_p : 0}
        return sess.run([self.hidden], feed_dict=feed_dict)[0]

    #------------------------------------------------------------

    def save_model(self, sess, prefix):
        np.save(prefix+"-hidden.npy", self.hidden_weights.eval(session=sess))
        np.save(prefix+"-output.npy", self.output_weights.eval(session=sess))

    #------------------------------------------------------------

    def load_model(self, sess, prefix):
        sess.run(self.hidden_weights.assign(np.load(prefix+"-hidden.npy")))
        sess.run(self.output_weights.assign(np.load(prefix+"-output.npy")))

#===============================================================================

