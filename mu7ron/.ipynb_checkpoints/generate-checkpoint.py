import numpy as np


def batch_to_sequence(batch, dtype=None):
    '''
    Converts a 3D tensor of shape=(n_sample, n_input, n_vocab)
    to a 1D tensor of shape=(n_sample + n_input - 1). Repeated values will
    be removed.
    '''
    n_sample, n_input, n_vocab = batch.shape
    dtype = batch.dtype if dtype is None else dtype
    seq   = np.empty((n_sample + n_input - 1, n_vocab), dtype=dtype)
    seq[: n_input, :] = batch[0, : n_input, :]
    for i in range(n_sample - 1):
        seq[n_input + i, :] = batch[i + 1, -1, :]
    return seq

def sequence_to_batch(seq, n_input, n_sample=None, dtype=None):
    '''
    Converts a 1D tensor of shape=(n_sample + n_input - 1) to a
    3D tensor of shape=(n_sample, n_input, n_vocab). 
    '''
    if not isinstance(seq, np.ndarray):
        seq  = np.array(seq)
    dtype    = seq.dtype if dtype is None else dtype
    n        = len(seq)
    n_vocab  = seq.shape[1]
    in_range = list(range(n_input))
    if n_sample is None:
        n_sample = 1 + n - n_input
    batch = np.empty((n_sample, n_input, n_vocab), dtype=dtype)
    batch[0, in_range, :] = seq[: n_input]
    for i in range(1, n_sample):
        batch[i, in_range, :] = seq[i : i + n_input]
    return batch

def append_prediction_to_batch(pred, batch):
    sample         = np.empty((1, *batch[-1].shape))
    sample[0, :-1] = batch[-1, 1:].copy()
    sample[0, -1]  = pred
    batch          = np.vstack([batch, sample])
    return batch

def heat(x, t, inplace=False):
    '''
    An attempt at implementing rnn decoding technique described here:
    https://nlp.stanford.edu/blog/maximum-likelihood-decoding-with-rnns-the-good-the-bad-and-the-ugly/
    Args:
        x - np.ndarray, 1D, probabilities
        t - temperature
    '''
    ret   = x if inplace else x.copy()
    ret **= 1. / np.clip(t, np.finfo(x.dtype).eps, 1.)
    return ret / np.sum(ret)
        
def semantic_temperature_sampling(x, map_, t1, t2, inplace=False):
    '''
    An attempt at implementing rnn decoding technique described here:
    https://nlp.stanford.edu/blog/maximum-likelihood-decoding-with-rnns-the-good-the-bad-and-the-ugly/
    Args:
        x      - np.ndarray, 1D, probabilities
        t1, t2 - temperature
    '''
    ret = x if inplace else x.copy()
    q   = np.empty(4, dtype=x.dtype)
    for i in range(4):
        q[i] = np.sum(ret[map_['sem'][i]])
    q_ = heat(x, t1)
    for i in range(4):
        ret[map_['sem'][i]] /= q[i]
        ret[map_['sem'][i]] = heat(ret[map_['sem'][i]], t2, inplace=True)
    ret *= q_
    ret /= np.sum(ret)
    return ret

def set_max_to_one(x, temperature=0.01):
    x = K.log(x) / temperature
    return K.round(K.softmax(x))

