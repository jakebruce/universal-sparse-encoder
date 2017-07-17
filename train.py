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
import sys

sys.dont_write_bytecode = True # keep python from creating .pyc file clutter

from model import UniversalSparseEncoder # imported from model.py

#===============================================================================
# PARAMS

INPUT_SIZE      = 784   # dimensionality of MNIST input data (28x28 pixels)
BITS            = 500   # number of bits in the encoder we're training
SPARSITY        = 0.05  # desired sparsity of our trained encoder

SPARSITY_WEIGHT = 100.0 # how heavily to weight the sparsity constraint
REGULAR_WEIGHT  = 1e-5  # strength of weight decay regularization
DROPOUT_INPUT   = 0.8   # keep input  units with this probability
DROPOUT_HIDDEN  = 0.5   # keep hidden units with this probability
INPUT_NOISE     = 0.25  # scale of gaussian noise added to inputs

LEARN_RATE      = 0.005 # initial  learning rate
LEARN_RATE_INC  = 0.5   # decay learning rate by this factor at regular intervals
LEARN_RATE_INT  = 1000  # learning rate decay interval

BATCH_SIZE      = 100   # number of training samples per gradient computation
TOTAL_BATCHES   = 10000 # stop training after this many batches and save the model

ENABLE_CV2_VIZ  = False  # use cv2 (python-opencv) for live visualization during training
TEST_INTERVAL   = 500   # test reconstructions and filters at regular intervals

MODEL_FILE      = "saved-model/mnist-encoder" # save model here after training

#===============================================================================
# DATA

def get_data():
    from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data
    return mnist_input_data.read_data_sets("MNIST_data/", one_hot=False)

#===============================================================================
# TRAIN

if ENABLE_CV2_VIZ:
    try:
        import cv2
        cv2.namedWindow("reconstructions", cv2.WINDOW_NORMAL)
        cv2.namedWindow("filters",         cv2.WINDOW_NORMAL)
    except Exception as e:
        print "Error using cv2 (python-opencv). Make sure cv2 is installed or set ENABLE_CV2_VIZ to False."

#----------------------------------------------------------------

with tf.Session() as sess:
    # build model network
    encoder = UniversalSparseEncoder(INPUT_SIZE, BITS,
                                     SPARSITY, BATCH_SIZE, SPARSITY_WEIGHT,
                                     REGULAR_WEIGHT, LEARN_RATE, DROPOUT_INPUT,
                                     DROPOUT_HIDDEN, INPUT_NOISE)

    # initialize all network variables
    sess.run(tf.global_variables_initializer())

    # get MNIST data
    train, valid, test = get_data()

    for batches in xrange(TOTAL_BATCHES):
        # training batch
        batch = train.images[np.random.randint(0,train.num_examples, size=(BATCH_SIZE))]

        # run input through network and do gradient descent update
        train_step, training_loss = encoder.train(sess, batch)

        if batches % TEST_INTERVAL == 0:
            # test batch
            batch = test.images[np.random.randint(0,test.num_examples, size=(BATCH_SIZE))]
            noisy, recons, test_loss = encoder.test(sess, batch, noisy=True)
            print "Batch %s:" % batches
            print "training loss: %s" % training_loss
            print "test     loss: %s" % test_loss

            if ENABLE_CV2_VIZ:
                # visualize reconstructions
                recons  = np.clip(recons, 0, 1)
                dim     = int(BATCH_SIZE**0.5)
                rec_img = np.zeros((dim*28, dim*3*28+dim), dtype=np.uint8)
                for i in range(dim**2):
                    xloc   = i%dim*3*28
                    yloc   = i/dim*28
                    offset = i%dim
                    rec_img[yloc:yloc+28,xloc+0*28+offset:xloc+1*28+offset] = np.clip(batch [i,...].reshape(28,28),0,1)*255
                    rec_img[yloc:yloc+28,xloc+1*28+offset:xloc+2*28+offset] = np.clip(noisy [i,...].reshape(28,28),0,1)*255
                    rec_img[yloc:yloc+28,xloc+2*28+offset:xloc+3*28+offset] = np.clip(recons[i,...].reshape(28,28),0,1)*255
                # vertical separators
                for i in range(dim-1):
                    rec_img[:,i*3*28+i+3*28] = 64
                cv2.imshow("reconstructions", rec_img)

                # visualize encoder filters
                filters  = encoder.hidden_weights.eval(session=sess).T
                dim      = int(BITS**0.5)
                filt_img = np.zeros((dim*28, dim*28), dtype=np.uint8)
                for i in range(dim**2):
                    filt = filters[i,:INPUT_SIZE]
                    filt = (filt-filt.min()) / (filt.max()-filt.min()+1e-8)
                    filt_img[i/dim*28:i/dim*28+28,i%dim*28:i%dim*28+28] = filt.reshape(28,28)*255
                cv2.imshow("filters", filt_img)
                cv2.waitKey(30)

        # decay learning rate at regular intervals
        if batches > 0 and batches % LEARN_RATE_INT == 0:
            sess.run(encoder.learn_rate.assign(encoder.learn_rate*LEARN_RATE_INC))

    # done training: save the model to disk
    print "Training complete. Saved model to %s" % MODEL_FILE
    encoder.save_model(sess, MODEL_FILE)

    if ENABLE_CV2_VIZ:
        print "Press any key to exit."
        cv2.waitKey(0)

