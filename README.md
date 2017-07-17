Universal Sparse Encoder
========================

This repository contains python code to train a soft K-sparse autoencoder using
TensorFlow on MNIST digits.

Usage:

```
> python train.py  # train on MNIST example data
> python encode.py # test encoder on MNIST data
```

The autoencoder consists of a single hidden sigmoid layer that learns
representations for denoising arbitrary inputs through an output layer. In
addition to the traditional autoencoder reconstruction-based loss and L2
regularization loss, we implement two additional constraints in the loss
function to encourage sparsity:

Per-sample hidden-layer sparsity constraint
-------------------------------------------

```python
layer_sparsity = tf.reduce_sum(hidden, axis=1) / hidden_size
layer_loss     = tf.reduce_mean(tf.squared_difference(layer_sparsity, DESIRED_SPARSITY))
```

This constraint, based on the sum of the sigmoid units along the *hidden layer*
dimension, encourages representations for *individual input examples* to
approach the desired sparsity.


Per-unit batch sparsity constraint
-------------------------------------------

```python
batch_sparsity = tf.reduce_sum(hidden, axis=0) / batch_size
batch_loss     = tf.reduce_mean(tf.squared_difference(batch_sparsity, DESIRED_SPARSITY))
```

This constraint, based on the sum of the sigmoid units along the *batch*
dimension, encourages *each unit in the hidden layer* to be active for a
desired fraction of the examples in the training batch.

These two constraints together encourage 1) sparsity of representation, and 2)
equal division of representational capacity across the hidden layer.

Discussion
----------

This is a visualization of the reconstructions learned during training:

![reconstructions](https://raw.githubusercontent.com/jakebruce/universal-sparse-encoder/master/imgs/reconstructions.png "Reconstructions")

And here's a visualization of the filter for each unit in the hidden layer:

![filters](https://raw.githubusercontent.com/jakebruce/universal-sparse-encoder/master/imgs/filters.png "Filters")

We can encode inputs as sparse binary vectors with K-winners-take-all, which
produces vectors like this on digits from the MNIST test set (note that the
network has never seen these examples before):

![encodings](https://raw.githubusercontent.com/jakebruce/universal-sparse-encoder/master/imgs/encodings.png "Encodings")

The horizontal axis enumerates the digits fed to the network, and the vertical
axis corresponds to the units in the hidden layer.

It is not necessarily clear how to evaluate the quality of these encodings, but
notice that there is clear structure shared between digits of the same class
(separated by the gray lines), compared to the similarities between digits of
different classes. Note that the class labels were nowhere used during training.

One way to evaluate encoding quality is to look at the differences between vectors
of different training examples. Here's a confusion matrix of the sum of absolute pixel
differences between the same digits as we encoded above:

![raw-confusion](https://raw.githubusercontent.com/jakebruce/universal-sparse-encoder/master/imgs/raw_confusion.png "Raw Confusion")

Now compare against the sum of absolute differences between our encoded binary vectors:

![sdr-confusion](https://raw.githubusercontent.com/jakebruce/universal-sparse-encoder/master/imgs/sdr_confusion_constraint100.png "SDR Confusion")

Notice the clear main diagonal for digits of the same class. Much better!

How important are our two auxiliary sparsity constraints that we introduced at
the beginning? Well, we can try to evaluate their importance by quantifying the
"goodness" of the vectors represented above. As a goodness metric, we'll use the
mean difference between classes, divided by the mean difference within classes.
First let's look at the confusion matrix for an encoder trained without sparsity
constraints:

![sdr-confusion-nosparsity](https://raw.githubusercontent.com/jakebruce/universal-sparse-encoder/master/imgs/sdr_confusion_noconstraint.png "SDR Confusion, No sparsity constraint")

Visually, the confusion matrix seems to be less crisp in that there is more
structure between classes than there was with the sparsity constraints. Now
let's quantify.

For encodings trained without the sparsity constraints:

```
Goodness score (diff between classes / diff within classes): 1.105
```

And for encodings trained with the sparsity constraints:

```
Goodness score (diff between classes / diff within classes): 1.190
```

So the sparsity constraint seems to improve the quality of the resulting
binary encodings, which makes sense.

There is nothing MNIST-specific in the model (although train.py and encode.py
are specific to MNIST), so please feel free to train the network on any
high-dimensional data you like, and use for your own projects if useful.

