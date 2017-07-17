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
import sys, itertools

sys.dont_write_bytecode = True # don't clutter the directory with .pyc files
from model import UniversalSparseEncoder # imported from model.py

#===============================================================================
# PARAMS

INPUT_SIZE     = 784   # dimensionality of MNIST input data (28x28 pixels)
BITS           = 500   # number of bits in the encoder
SPARSITY       = 0.05  # sparsity of our trained encoder
ENABLE_CV2_VIZ = True  # use cv2 (python-opencv) to visualize learned encodings
NUM_PER_DIGIT  = 50    # test on this many copies of each digit
BATCH_SIZE     = 10*NUM_PER_DIGIT # number of samples per batch, N of each digit
MODEL_FILE     = "saved-model/mnist-encoder" # load trained model from here

#===============================================================================
# DATA

def get_data():
    from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data
    return mnist_input_data.read_data_sets("MNIST_data/", one_hot=False)

#===============================================================================
# ENCODE

if ENABLE_CV2_VIZ:
    try:
        import cv2
        cv2.namedWindow("digits",        cv2.WINDOW_NORMAL)
        cv2.namedWindow("encodings",     cv2.WINDOW_NORMAL)
        cv2.namedWindow("raw_confusion", cv2.WINDOW_NORMAL)
        cv2.namedWindow("sdr_confusion", cv2.WINDOW_NORMAL)
    except Exception as e:
        print "Error using cv2 (python-opencv). Make sure cv2 is installed or set ENABLE_CV2_VIZ to False."


#----------------------------------------------------------------

with tf.Session() as sess:
    # build and initialize model network
    encoder = UniversalSparseEncoder(INPUT_SIZE, BITS, SPARSITY, BATCH_SIZE)
    sess.run(tf.global_variables_initializer())

    # load pretrained model
    encoder.load_model(sess, MODEL_FILE)

    # get some MNIST data
    train, valid, test = get_data()

    # visualize encoding on N of each digit
    batch = np.zeros((10*NUM_PER_DIGIT,INPUT_SIZE), dtype=np.float32)
    for digit in range(10):
        for i in range(NUM_PER_DIGIT):
            # pull random digits until we get the label we want
            random_digit = np.random.randint(0,valid.num_examples)
            while valid.labels[random_digit] != digit:
                random_digit = np.random.randint(0,valid.num_examples)
            batch[digit*NUM_PER_DIGIT+i,...] = valid.images[random_digit]

    # encode our chosen digits
    encodings = encoder.encode(sess, batch)

    # K-winners-take-all on encodings to form binary SDRs
    K = int(SPARSITY*BITS)
    enc_sdrs   = np.zeros((10*NUM_PER_DIGIT, K),    dtype=np.int32)
    dense_sdrs = np.zeros((10*NUM_PER_DIGIT, BITS), dtype=np.int32)
    for digit in range(10):
        print "Digit %s encodings:" % digit
        for i in range(NUM_PER_DIGIT):
            bits_on = np.argpartition(encodings[digit*NUM_PER_DIGIT+i,...], -K)[-K:]
            print np.array(sorted(bits_on))
            enc_sdrs  [digit*NUM_PER_DIGIT+i,:] = bits_on
            dense_sdrs[digit*NUM_PER_DIGIT+i,bits_on] = 1

    # build confusion matrices to evaluate the quality of the SDRs
    raw_confusion = np.zeros((10*NUM_PER_DIGIT, 10*NUM_PER_DIGIT), dtype=np.float32)
    sdr_confusion = np.zeros((10*NUM_PER_DIGIT, 10*NUM_PER_DIGIT), dtype=np.float32)
    for (digitx, ix), (digity, iy) in itertools.combinations(itertools.product(range(10),range(NUM_PER_DIGIT)), 2):
        idxx = digitx*NUM_PER_DIGIT+ix
        idxy = digity*NUM_PER_DIGIT+iy
        raw_confusion[idxx,idxy] = np.abs(     batch[idxx,:] -      batch[idxy,:]).sum()
        raw_confusion[idxy,idxx] = np.abs(     batch[idxx,:] -      batch[idxy,:]).sum()
        sdr_confusion[idxx,idxy] = np.abs(dense_sdrs[idxx,:] - dense_sdrs[idxy,:]).sum()
        sdr_confusion[idxy,idxx] = np.abs(dense_sdrs[idxx,:] - dense_sdrs[idxy,:]).sum()

    # quantify the goodness of the SDR with a score:
    # difference between classes divided by difference within classes
    diff_within  = 0.0
    diff_between = 0.0
    for (digitx, ix), (digity, iy) in itertools.permutations(itertools.product(range(10),range(NUM_PER_DIGIT)), 2):
        diff = sdr_confusion[digitx*NUM_PER_DIGIT+ix,digity*NUM_PER_DIGIT+iy]
        if digitx == digity: diff_within  += diff
        else:                diff_between += diff
    goodness_score = diff_between/10 / diff_within
    print "Goodness score (diff between classes / diff within classes):", goodness_score

    #--------------------------------------------------------------------------
    # visualize with cv2

    if ENABLE_CV2_VIZ:
        # build an image showing all ten digits
        digit_img = np.zeros((10*28,NUM_PER_DIGIT*28), dtype=np.uint8)
        for digit in range(10):
            for i in range(NUM_PER_DIGIT):
                digit_img[digit*28:(digit+1)*28,i*28:(i+1)*28] = batch[digit*NUM_PER_DIGIT+i,...].reshape(28,28)*255
        cv2.imshow("digits", digit_img)

        # build an image showing all encodings
        enc_img = np.zeros((BITS, 10*NUM_PER_DIGIT+9), dtype=np.uint8)
        for digit in range(10):
            for i in range(NUM_PER_DIGIT):
                bits = enc_sdrs[digit*NUM_PER_DIGIT+i,:]
                enc_img[bits, digit*NUM_PER_DIGIT+i+digit] = 255
        # separator lines
        for digit in range(9):
            enc_img[:,digit*NUM_PER_DIGIT+i+digit+1] = 64
        cv2.imshow("encodings", enc_img)

        # show confusion images
        raw_confusion = 255-(raw_confusion-raw_confusion.min())/(raw_confusion.max()-raw_confusion.min()+1e-8)*255
        sdr_confusion = 255-(sdr_confusion-sdr_confusion.min())/(sdr_confusion.max()-sdr_confusion.min()+1e-8)*255
        cv2.imshow("raw_confusion", raw_confusion.astype(np.uint8))
        cv2.imshow("sdr_confusion", sdr_confusion.astype(np.uint8))

        # display images and wait for user input
        cv2.waitKey(0)

