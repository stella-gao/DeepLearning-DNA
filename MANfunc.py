import numpy as np
import tensorflow as tf

def np_softmax(logits):
    Z = np.sum(np.exp(logits), 1, keepdims=True)
    return np.divide(np.exp(logits), Z)

def np_sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = np.random.uniform(0,1, shape)
    return -np.log(-np.log(U + eps) + eps)

def np_gumbel_softmax_sample(logits, temperature, inp, lam=0.5):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = lam*inp*logits + (1-lam)*np_sample_gumbel(np.shape(logits))
    return np_softmax( y / temperature)

def np_gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
    """
    y = np_gumbel_softmax_sample(logits, temperature)
    if hard:
        k = np.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = (np.equal(y,np.max(y,1,keepdims=True)))
#         y = tf.stop_gradient(y_hard - y) + y
    return y_hard-y+y

def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(inp, logits, temperature, lam=0.5):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = lam*inp*logits + (1-lam)*sample_gumbel(tf.shape(inp))
    return tf.nn.softmax( y / temperature, dim=1)

def gumbel_softmax(inp, logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(inp, logits, temperature)
    if hard:
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y

def get_batch(X_train, batch_size=50):
    ix = 0
    while True:
        if ix>=(X_train.shape[0]-batch_size):
            ix = 0
        ix+=batch_size
        yield  X_train[ix:(ix+batch_size)]