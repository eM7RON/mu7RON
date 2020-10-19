import numpy as np

from mu7ron import utils


def initialize_weights(x, n_vocab=146, eps=None):
    '''
    Returns weights over a vector n_vocab in length.
    Is designed to allow weighted_categorical_crossentropy
    '''
    ret   = np.zeros(n_vocab, dtype='float32')
    eps   = 0. if eps is None else np.finfo('float32').eps
    ret += eps
    avnts = list(utils.flatten(exhaustive_transpose(x)))
    cnt   = utils.counter(avnts)
    n     = len(avnts)
    for i in range(n_vocab):
        if i in cnt:
            ret[i] = cnt[i] / n
    return ret

def vocab_wts(x:list, n_time:int, n_vocab:int,
              off_mode:bool, rho:float=0.4) -> np.ndarray:
    '''
    Returns weights that are different between each possible vocab
    '''
    ret   = np.zeros(n_vocab, dtype='float32')
    #ret   += np.finfo(x.dtype).eps
    avnts = list(utils.flatten(
        aug.exhaustive_transpose(x, n_time=n_time, off_mode=off_mode)))
    cnt   = utils.counter(avnts)
    n     = len(avnts)
    for i in range(n_vocab):
        if i in cnt:
            ret[i] = cnt[i] / n
    np.clip(ret, np.finfo(ret.dtype).eps, 1.)
    #ret /= np.max(ret)
    ret **= rho
    #ret /= np.max(ret)
    ret = 1. - ret
    np.clip(ret, np.finfo(ret.dtype).eps, 1.)
    return ret

def semantic_wts(x:list, n_time:int, n_vocab:int, 
                 map_:dict, off_mode:bool, rho=0.5) -> np.ndarray:
    '''
    Returns weights that are different across semantic partitions
    given by the 'map' argument
    '''
    ret   = np.zeros(n_vocab, dtype='float32')
    sums  = np.zeros(4, dtype='float32')
    avnts = list(utils.flatten(
        aug.exhaustive_transpose(x, n_time=n_time, off_mode=off_mode)))
    n     = len(avnts)
    for a in avnts:
        sums[map_['voc'][a]] += 1
    sums /= n
    sums /= np.max(sums)
    for i in range(len(map_['sem'].keys())):
        for j in map_['sem'][i]:
            ret[j] = sums[i]
    ret **= rho
    #ret /= np.max(ret)
    np.clip(ret, np.finfo(ret.dtype).eps, 1.)
    return ret