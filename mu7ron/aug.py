'''
Contains functions for augmenting encoded music data.

Please note that in this script an evnt (event) refers to a midi event or midi.events object
whereas an avnt (avent) refers to a single element of midi data that has been encoded or 
translated into an input format for a neural network. For example, an avnt may just be a 
single base 10 integer or a one hot encoded vector.
'''

import os
import pickle
import random

from tensorflow import __version__
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from packaging import version
import numpy as np

from mu7ron import maps
from mu7ron import temporal
from mu7ron import utils


def transpose(x, adj, n_t_p=None, n_time=64, inplace=False, off_mode=False):
    '''
    Shifts all of the pitches values of the note avnts in an encoded midi pattern by adj
    semitones
    '''
    ret     = x if inplace else x.copy()
    if n_t_p is None:
        n_t_p   = n_time + 128 * 2 if off_mode else n_time + 128
    idx     = (n_time <= x) & (x < n_t_p)
    if idx.sum():
        pitches = x[idx]
        if n_time <= np.min(pitches) + adj and np.max(pitches) + adj < n_t_p:
            ret[idx] += adj
    return ret

def exhaustive_transpose(x, range_=range(-4, 5), n_time=10, off_mode=False):
    '''
    Exhausts a list of transpose arguments
    '''
    ret = []
    for seq in x:
        for i in range_:
            if i:
                ret.append(transpose(seq, i, n_time=n_time, off_mode=off_mode))
            else:
                ret.append(seq)
    return ret

def random_choice_transpose(x, options=np.arange(-4, 5), wts=None, 
                            n_time=64, verbose=False, inplace=False, off_mode=False):
    '''
    Shifts all of the pitches values of the note avents in an encoded midi pattern by a
    random amount chosen from options argument.
    '''
    ret   = x if inplace else x.copy()
    n_t_p = n_time + 128 * 2 if off_mode else n_time + 128
    idx   = (n_time <= x) & (x < n_t_p)
    adj   = 0
    if any(idx):
        n       = options.shape[0]
        jdx     = np.zeros(n, dtype=bool)
        pitches = x[idx]
        min_    = np.min(pitches)
        max_    = np.max(pitches)
        for i in range(n):
            o = options[i]
            if not o or o < 0 and n_time <= min_ + o or o > o and max_ + o < n_t_p:
                jdx[i] = True
        if any(jdx):
            options = options[jdx]
            if wts is not None:
                wts     = wts[jdx]
                wts    /= np.sum(wts)
            adj     = np.random.choice(options, p=wts)
            ret[idx] += adj
    if verbose:
        print(f'transposed pitches {adj} steps')
    return ret

def random_bounded_transpose(x, lo_lim=40, hi_lim=100, n_time=64,
                             verbose=False, inplace=False, off_mode=False):
    '''
    Shifts all of the pitches values of the note avents in an encoded midi pattern by a
    random amount. The shift will occur within the bounds of feasibile pitch values 
    [0, 128] but can be further restricted by usage of lo_lim and hi_lim arguments.
    '''
    ret     = x if inplace else x.copy()
    n_t_p   = n_time + 128 * 2 if off_mode else n_time + 128
    lo_lim += n_time
    hi_lim += n_time
    idx     = (n_time <= x) & (x < n_t_p)
    if sum(idx):
        pitches = x[idx]
        min_    = max(np.min(pitches), lo_lim)
        max_    = min(np.max(pitches), hi_lim)
    else:
        min_, max_ = lo_lim, hi_lim
    hi = np.abs(max_ - hi_lim)
    lo = -np.abs(min_ - lo_lim)
    if lo < hi:
        adj       = np.random.randint(lo, hi)
        ret[idx] += adj
    else:
        adj = 0
    if verbose:
        print(f'transposed pitches {adj} steps', lo, hi)
    return ret

def bounded_exhaustive_transpose(x, lo_lim=40, hi_lim=100, n_time=64,
                             verbose=False, off_mode=False):
    '''
    Shifts all of the pitches values of the note avents in an encoded midi pattern by a
    random amount. The shift will occur within the bounds of feasibile pitch values 
    [0, 128] but can be further restricted by usage of lo_lim and hi_lim arguments.
    '''
    ret     = [x]
    n_t_p = n_time + 128 * 2 if off_mode else n_time + 128
    lo_lim += n_time
    hi_lim += n_time
    idx     = (n_time <= x) & (x < n_t_p)
    if sum(idx):
        pitches = x[idx]
        min_    = max(np.min(pitches), lo_lim)
        max_    = min(np.max(pitches), hi_lim)
    else:
        min_, max_ = lo_lim, hi_lim
    hi = np.abs(max_ - hi_lim) + 1
    lo = -np.abs(min_ - lo_lim)
    if lo < hi:
        for adj in range(lo, hi):
            if adj:
                clone = x.copy()
                clone[idx] += adj
                ret.append(clone)
    if verbose:
        print(f'transposed pitches {adj} steps', lo, hi)
    return ret

def permute_simultaneous_evnts(x:np.ndarray, n_time:int=64, n_t_p:int=253,
                               inplace:bool=False, off_mode=False) -> np.ndarray:
    '''
    Where x is an already mugen encoded input sequence, this function shuffles 
    the order of avents that occur simultaneously as to avoid the network learning
    their order when it yields no benefit but promotes learning order when it's 
    necessary. n_inst is the first number used to encode velocity avnts and n_time
    is the first number to encode pitch avnts.
    '''
    ret = x if inplace else x.copy()
    if n_t_p is None:
        n_t_p = n_time + 128 * 2 if off_mode else n_time + 128
    range0 = np.arange(len(x))
    # indices of time events
    idx0   = range0[x < n_time]
    split0 = np.split(range0, idx0)
    for i0 in split0:
        # indices not time events
        idx1   = range0[i0][x[i0] >= n_time]
        range1 = np.arange(len(idx1))
        idx2   = range1[x[idx1] >= n_t_p]
        split1 = np.split(idx1, idx2)
        for i1 in split1:
            i1 = i1[x[i1] < n_t_p]
            if len(i1):
                ret[i1] = np.random.permutation(x[i1])
    return ret

def sort_simultaneous_evnts(x:np.ndarray, n_time:int=64, n_t_p:int=253,
                               inplace:bool=False, off_mode=False) -> np.ndarray:
    '''
    Where x is an already mugen encoded input sequence, this function sorts
    the order of avents that occur simultaneously into ascending order. n_inst is
    the first number used to encode velocity avnts and n_time is the first number
    to encode pitch avnts.
    '''
    ret = x if inplace else x.copy()
    if n_t_p is None:
        n_t_p = n_time + 128 * 2 if off_mode else n_time + 128
    range0 = np.arange(len(x))
    # indices of time events
    idx0   = range0[x < n_time]
    split0 = np.split(range0, idx0)
    for i0 in split0:
        # indices not time events
        idx1   = range0[i0][x[i0] >= n_time]
        range1 = np.arange(len(idx1))
        idx2   = range1[x[idx1] >= n_t_p]
        split1 = np.split(idx1, idx2)
        for i1 in split1:
            idx3 = x[i1] < n_inst
            i1 = i1[idx3]
            if len(i1):
                ret[i1] = ret[i1[np.argsort(ret[i1])]]
    return ret

def time_warp(x, adj, n_req=None, n_time=64, asarray=True, dtype='int32',
              time_encoder=temporal.base_digits_encoder, ekwa=dict(b=64), 
              time_decoder=temporal.base_digits_decoder, dkwa=dict(b=64), **kwargs):
    '''
    Adjust the tempo of an already mugen encoded midi.Pattern/midi.Track. Here
    adj is a float representing the retained fraction of the original time.
    '''
    ret  = []
    idx  = x < n_time
    n     = 0
    if sum(idx):
        jdx   = np.where(idx)[0]
        cache = []
        for i in range(len(x)):
            if i in jdx:
                cache.append(x[i]) 
            else:
                if cache:
                    ms  = time_decoder(cache, **dkwa)
                    ms *= adj
                    ret.extend(time_encoder(ms, **ekwa))
                    cache = []
                ret.append(x[i])
            n += 1
            if n_req is not None and len(ret) >= n_req:
                break
    if asarray:
        ret = np.array(ret, dtype=dtype)
    return ret

def random_choice_time_warp(x, n_req=None, options=np.linspace(0.95, 1.05, num=10), wts=None, n_time=125,
                            asarray=True, dtype='int32', time_encoder=temporal.base_digits_encoder, ekwa=dict(b=64),
                             time_decoder=temporal.base_digits_decoder, dkwa=dict(b=64), **kwargs):
    '''
    Adjust the tempo of an already mugen encoded midi.Pattern/midi.Track. Here
    adj is a float representing the retained fraction of the original time.
    '''
    ret  = []
    idx  = x < n_time
    n     = 0
    if sum(idx):
        adj   = np.random.choice(options, p=wts)
        jdx   = np.where(idx)[0]
        cache = []
        for i in range(len(x)):
            if i in jdx:
                cache.append(x[i]) 
            else:
                if cache:
                    ms  = time_decoder(cache, **dkwa)
                    ms *= adj
                    ret.extend(time_encoder(ms, **ekwa))
                    cache = []
                ret.append(x[i])
            n += 1
            if n_req is not None and len(ret) >= n_req:
                break
    if asarray:
        ret = np.array(ret, dtype=dtype)
    return ret

def random_bounded_time_warp(x, n_req=None, lo_lim=0.95, hi_lim=1.05, n_time=64, asarray=True,
                             dtype='int32', time_encoder=temporal.base_digits_encoder, ekwa=dict(b=64),
                             time_decoder=temporal.base_digits_decoder, dkwa=dict(b=64),
                             **kwargs):
    '''
    Adjust the tempo of an already mugen encoded midi.Pattern/midi.Track. Here
    adj is a float representing the retained fraction of the original time.
    '''
    ret   = []
    idx   = x < n_time
    n     = 0
    if sum(idx):
        adj   = np.random.uniform(lo_lim, hi_lim)
        jdx   = np.where(idx)[0]
        cache = []
        for i in range(len(x)):
            if i in jdx:
                cache.append(x[i]) 
            else:
                if cache:
                    ms  = time_decoder(cache, **dkwa)
                    ms *= adj
                    ret.extend(time_encoder(ms, **ekwa))
                    cache = []
                ret.append(x[i])
            n += 1
            if n_req is not None and len(ret) >= n_req:
                break
    if asarray:
        ret = np.array(ret, dtype=dtype)
    return ret


class RandomDataAugGen(Sequence):
    def __init__(self, data, **kwargs):

        '''
        A generator that performs data augmentation on midi data that has been
        pre-encoded as input data.

        Args:
            input_size  - int, x < 0, the length of a  single input encoded midi pattern, 
                          i.e. the number of events.
            batch_size  - int, x < 0, number of samples to be included in a single batch of
                          input.
            funcs       - list([func1, func2]), lists that contain the functions that will 
                          augment the data. They should take a single input encoded midi
                          pattern as input.
        Yields:
            x - numpy.ndarray, shape=(batch_size, input_size, 6), the input data
            y - numpy.ndarray, shape=(batch_size, 6), the targets
        '''

        self.n_vocab    = kwargs.get('n_vocab', 252)
        self.n_input    = kwargs.get('n_input', 100)
        self.n_output   = kwargs.get('n_output', 1)
        self.n_sample   = kwargs.get('n_sample', 256)
        self.dtype      = kwargs.get('dtype', 'float32')
        self.aug_funcs  = kwargs.get('aug_funcs', [])
        self.buffer     = kwargs.get('buffer', 100)
        self.shuffle    = kwargs.get('shuffle', False)
        
        self.rtrn_mode  = True if version.parse(__version__) <= version.parse('2.1.0') else False
        self.n_io       = self.n_input + self.n_output
        self.in_range   = range(self.n_input)
        self.out_range  = range(self.n_output)
        self.batch_loop = range(self.n_sample)
        self.n_req      = self.n_input + self.n_output + self.n_sample - 1
        
        self.rtrn_map     = {
                             True: lambda x, y: (x, y), 
                             False: lambda x, y: (x, y, [None])
                             }
        self.rtrn_mode    = version.parse(__version__) >= version.parse("2.2.0")
        # purge sequences that are too short
        data               = [x for x in data if len(x) > self.n_req + self.buffer]
        self.data          = data
        self.len           = len(data)
        self.i             = 0

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        x      = np.zeros((self.n_sample, self.n_input, self.n_vocab), dtype='float32')
        y      = np.zeros((self.n_sample, self.n_vocab), dtype='float32')

        # select a random segment of the sample
        sampling = True

        while sampling:
            sample = self.data[idx].copy()
            n      = len(sample)
            try:
                i      = np.random.randint(n - self.n_req - self.buffer)
                sample = sample[i: i + self.n_req + self.buffer]
                # apply augmentation functions
                for f in self.aug_funcs:
                    sample = f(self, sample)
            except IndexError:
                continue
            else:
                if len(sample) >= self.n_req:
                    sampling = False
        sample = sample[:self.n_req]

        for j in self.batch_loop:
            sub = sample[j: j + self.n_io]
            x[j, self.in_range, sub[ :self.n_input]] = 1.
            y[j, sub[self.n_input: self.n_io]]       = 1.

        return self.rtrn_map[self.rtrn_mode](x, y)

    def __next__(self):
        if self.i + 1 >= len(self.data):
            self.i  = 0
        else:
            self.i += 1
        return self.__getitem__(self.i)
    
    def __rand__(self):
        i = np.random.randint(self.len)
        return self.__getitem__(i)
    
    def _shuffle(self):        
        self.data = np.random.permutation(self.data)

    def on_epoch_end(self):
        if self.shuffle:
            self._shuffle()
            

class MappedDataAugGen(Sequence):
    def __init__(self, data, **kwargs):
        '''
        A generator that performs data augmentation on midi data that has been
        pre-encoded as input data. The main idea is that instead of computing all
        augmentations beforehand and holding them in memory, we sacrifice some speed
        by computing them as and when they are needed and by doing so allow us to 
        use a dataset that would otherwise not fit in memory.
        
        This generator calculates all possible augmentations and then stores the information
        required to quickly generate them in the dictionary attribute 'self.aug_map'.

        Args:
            n_input - int, number of timesteps used as input
            n_output - int, number of timesteps to predict
            n_sample - int, batch/mini batch size, number of samples to predict at once
            n_time   - int, number of bits that represent time in one-hot encoding
            n_vocab  - int, number of categorical possibilities, number of bits that
                       represent an avnt in one-hot encoding
            n_teach  - int, number of teacher forced samples per original sample
            buffer   - int, default=100, number of avnts to use as a buffer for time warp
                       augmentations because these can reduce the number of avnts in a
                       sample
            dtype           - str, default='float32' data type in the output array
            off_mode        - bool, default=True, True if NoteOffEvents are to be considered
            time_encoder    - callable, a function to encode milliseconds into a time avnt
            time_decoder    - callable, a function to decode a time avnt into milliseconds
            ekwa, dkwa      - dict, time_encoder and time_decoder keyword arguments respectively
            shuffle         - bool, default=False, if True the input data will be shuffled
                              at the end of each epoch
            teacher_forcing - bool, default=False, whether to use teacher_forcing
            lo_lim, hi_lim  - int, lower and higher midi pitch limits respectively; for when
                              considering pitch augmentations
            time_aug_range  - list[float], default=[0.94, 0.97, 1., 1.03, 1.06], time warp
                              augmentations

        Yields:
            x - numpy.ndarray, shape=(n_sample, n_input, n_vocab), the input data
            y - numpy.ndarray, shape=(n_sample, n_vocab), the targets
        '''

        self.n_input         = kwargs.get('n_input', 100)
        self.n_output        = kwargs.get('n_output', 1)
        self.n_sample        = kwargs.get('n_sample', 256)
        self.n_time          = kwargs.get('n_time', 64)
        self.n_vocab         = kwargs.get('n_vocab', 252)
        self.n_teach         = kwargs.get('n_teach', 100)
        self.n_step          = kwargs.get('n_step', self.n_input + self.n_output + self.n_sample - 2)
        self.buffer          = kwargs.get('buffer', 75)
        self.cache           = kwargs.get('cache', None)
        self.cache_mode      = kwargs.get('cache_mode', False)

        self.dtype           = kwargs.get('dtype', 'float32')
        self.off_mode        = kwargs.get('off_mode', True)
        self.time_encoder    = kwargs.get('time_encoder', temporal.base_digits_encoder)
        self.time_decoder    = kwargs.get('time_decoder', temporal.base_digits_decoder)
        self.ekwa            = kwargs.get('ekwa', dict(b=self.n_time))
        self.dkwa            = kwargs.get('dkwa', dict(b=self.n_time))
        self.shuffle         = kwargs.get('shuffle', False)
        self.teacher_forcing = kwargs.get('teacher_forcing', False)

        self.lo_lim          = kwargs.get('lo_lim', 60)
        self.hi_lim          = kwargs.get('hi_lim', 80)
        self.time_aug_range  = kwargs.get('time_aug_range', [0.95, 0.975, 1., 1.025, 1.05])

        self.n_io         = self.n_input + self.n_output
        self.in_range     = range(self.n_input)
        self.out_range    = range(self.n_output)
        self.batch_loop   = range(self.n_sample)
        self.n_req        = self.n_input + self.n_output + self.n_sample - 1
        self.prev_map     = None
        
        self.rtrn_map     = {
                             True: lambda x, y: (x, y), 
                             False: lambda x, y: (x, y, [None])
                             }
        self.rtrn_mode    = version.parse(__version__) >= version.parse("2.2.0")

        # purge sequences that are too short
        original_len = len(data)
        if self.teacher_forcing:
            data          = [x for x in data if len(x) > self.n_req + self.n_teach * self.n_step + self.buffer]
        else:
            data          = [x for x in data if len(x) > self.n_req + self.buffer]
        after_len = len(data)
        print(f'{original_len - after_len} samples were removed because they are too short. There are {after_len} remaining. \nIf this is not enough, try reducing n_step, n_teach and/or buffer and create a new instance of MappedDataAugGen')

        self.aug_map      = maps.augmentation_map(data,
                                                  self.n_time,
                                                  self.off_mode,
                                                  lo_lim=self.lo_lim,
                                                  hi_lim=self.hi_lim,
                                                  t_range=self.time_aug_range
                                                  )
        if self.teacher_forcing:
            self.aug_map  = maps.teacher_forcing_map(
                                                    data, 
                                                    self.aug_map, 
                                                    self.n_time, 
                                                    self.n_vocab, 
                                                    self.n_input, 
                                                    self.n_output,
                                                    self.n_sample,
                                                    self.n_teach,
                                                    self.n_step, 
                                                    self.buffer
                                                    )

        self.__getitem__  = self._tf_getitem if self.teacher_forcing else self._aug_getitem

        self.data         = data
        self.len          = len(self.aug_map)
        self.indices      = np.random.permutation(list(range(self.len)))
        self.i            = 0
        
        if self.cache_mode:
            range_len = range(self.len)
            self.pkl_range = list(zip(range_len, [repr(i) for i in range_len]))
            if self.cache is None:
                self.init_cache()
                
            self.pickle_data()
            self.__getitem__ = self._from_cache

    def __len__(self):
        return self.len
    
    def _tf_getitem(self, idx):

        map_   = self.aug_map[self.indices[idx]]

        if self.prev_map is None or map_['id'] != self.prev_map['id']:
            self.prev_map = map_
            sample        = self.data[map_['orig_sample_idx']].copy()
            if map_['time_aug'] != 1.:
                # apply time warp
                sample = time_warp(
                                   sample,
                                   map_['time_aug'],
                                   time_encoder=self.time_encoder,
                                   ekwa=self.ekwa,
                                   time_decoder=self.time_decoder,
                                   dkwa=self.dkwa,
                                   n_req=len(sample),
                                   n_time=self.n_time
                                   )
            sample = transpose(sample, map_['pitch_aug'], n_time=self.n_time, off_mode=self.off_mode)
            self.current_sample = sample[map_['strt_idx']: map_['end_idx']]

        sample = self.current_sample[map_['teach_idx']: map_['teach_idx'] + self.n_req].copy()
        x      = np.zeros((self.n_sample, self.n_input, self.n_vocab), dtype=self.dtype)
        y      = np.zeros((self.n_sample, self.n_vocab), dtype=self.dtype)

        for i in self.batch_loop:
            sub = sample[i: i + self.n_io]
            x[i, self.in_range, sub[ :self.n_input]] = 1.
            y[i, sub[self.n_input: self.n_io]]       = 1.
            
        return self.rtrn_map[self.rtrn_mode](x, y)

    def _aug_getitem(self, idx):

        # 3-tuple (data[i], pitch adjustment, time adjustment)
        map_   = self.aug_map[self.index[idx]]
        x      = np.zeros((self.n_sample, self.n_input, self.n_vocab), dtype=self.dtype)
        y      = np.zeros((self.n_sample, self.n_vocab), dtype=self.dtype)

        # select a random segment of the sample
        sampling = True

        while sampling:
            # we use try accept block because some augmentations such as time warp may
            # change the length of a sequence - this is why we use a buffer
            sample = self.data[map_['orig_sample_idx']].copy()
            n      = len(sample)
            i      = np.random.randint(n - self.n_req - self.buffer)
            sample = sample[i: i + self.n_req + self.buffer]

            if map_['time_aug'] != 1.:
                try:
                    # apply time warp
                    sample = time_warp(sample,
                                           map_['time_aug'], 
                                           time_encoder=self.time_encoder,
                                           ekwa=self.ekwa,
                                           time_decoder=self.time_decoder,
                                           dkwa=self.dkwa,
                                           n_req=self.n_req, 
                                           n_time=self.n_time)
                except IndexError:
                    continue
                else:
                    if len(sample) >= self.n_req:
                        sampling = False
            else:
                break
      
        sample = transpose(sample, map_['pitch_aug'], n_time=self.n_time, off_mode=self.off_mode)
        sample = sample[: self.n_req + 1]
        
        for j in self.batch_loop:
            sub = sample[j: j + self.n_io]
            x[j, self.in_range, sub[ :self.n_input]] = 1.
            y[j, sub[self.n_input: self.n_io]]       = 1.
            
        return self.rtrn_map[self.rtrn_mode](x, y)
    
    def pickle_data(self):
        print('pickling augmentations...')
        for n, s in tqdm(self.pkl_range):
            with open(os.path.join(self.cache, s), 'wb') as file:
                pickle.dump(self.__getitem__(n), file)
                
    def _from_cache(self, idx):
        idx = repr(self.indices[idx])
        with open(os.path.join(self.cache, idx), 'rb') as file:
            return pickle.load(file)

    def __getitem__(self, idx):
        return self._tf_getitem(idx)
    
    def __next__(self):
        if self.i + 1 >= self.len:
            self.i  = 0
        else:
            self.i += 1
        return self.__getitem__(self.i)
    
    def __rand__(self):
        return self.__getitem__(np.random.randint(self.len))
    
    def _shuffle(self):        
        self.indices = np.random.permutation(self.indices)

    def on_epoch_end(self):
        if self.shuffle:
            self._shuffle()
            
    def init_cache(self):
        dirs = 'data', 'cache'
        self.cache = ''
        for d in dirs:
            self.cache = os.path.join(self.cache, d)
            if not os.path.exists(self.cache):
                os.mkdir(self.cache)


class MappedBalancedDataGen(Sequence):
    def __init__(self, data, **kwargs):
        '''
        A generator that samples the data in a balanced way and creates a map to generate those 
        samples sequentialy, avoiding holding them all in memory.
        
        The sampler will attempt to find n_example sequences for each possible output in the 
        vocabulary. So n_example * n_vocab maximum. But some possible outputs may not occur 
        in a sample or in the data. In these cases the sampler will attempt to find as many 
        as possible but may return zero of certain outputs. If this becomes a problem, it may 
        be beneficial to reduce n_examples or increase the number of samples in the data.

        Args:
            n_input - int, number of timesteps used as input
            n_output - int, number of timesteps to predict
            n_sample - int, batch/mini batch size, number of samples to predict at once
            n_vocab  - int, number of categorical possibilities, number of bits that
                       represent an avnt in one-hot encoding
            shuffle  - bool, default=False, whether to shuffle the order of batches on
                       epoc end
            equal_bs - bool, default=True, whether to ensure that every batch is the 
                       same size, if True excess samples will be discarded
            dtype    - str, default='float32' data type in the output array
            off_mode - bool, default=True, True if NoteOffEvents are to be considered

        Yields:
            x - numpy.ndarray, shape=(n_sample, n_input, n_vocab), the input data
            y - numpy.ndarray, shape=(n_sample, n_vocab), the targets
        '''

        self.n_input   = kwargs.get('n_input', 100)
        self.n_output  = kwargs.get('n_output', 1)
        self.n_sample  = kwargs.get('n_sample', 256)
        self.n_vocab   = kwargs.get('n_vocab', 252)
        self.n_example = kwargs.get('n_example', 5)
        self.shuffle   = kwargs.get('shuffle', False)
        self.equal_bs  = kwargs.get('equal_bs', True)

        self.dtype    = kwargs.get('dtype', 'float32')
        self.off_mode = kwargs.get('off_mode', True)

        self.n_io       = self.n_input + self.n_output
        self.in_range   = range(self.n_input)
        self.out_range  = range(self.n_output)
        self.batch_loop = range(self.n_sample)
        self.n_req      = self.n_input + self.n_output + self.n_sample - 1
        
        self.rtrn_map     = {
                             True: lambda x, y: (x, y), 
                             False: lambda x, y: (x, y, [None])
                             }
        self.rtrn_mode    = version.parse(__version__) >= version.parse("2.2.0")
        
        # purge sequences that are too short
        data = [x for x in data if len(x) >= self.n_req]

        self.map = maps.balanced_sampling_map(data, self.n_input, self.n_output, 
                                              self.n_example, self.n_vocab)
        
        self.map = [self.map[x :x + self.n_sample] for x in range(0, len(self.map), self.n_sample)]
        
        if self.equal_bs:
            self.map = [x for x in self.map if len(x) == self.n_sample]

        self.data         = data
        self.len          = len(self.map)
        self.i            = 0

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        map_   = self.map[idx]
        x      = np.zeros((self.n_sample, self.n_input, self.n_vocab), dtype=self.dtype)
        y      = np.zeros((self.n_sample, self.n_vocab), dtype=self.dtype)
        
        for i in self.batch_loop:
            sample = self.data[map_[i]['orig_sample_idx']].copy()[map_[i]['strt_idx']: map_[i]['end_idx']]
            x[i, self.in_range, sample[:self.n_input]] = 1.
            y[i, sample[self.n_input: self.n_io]]      = 1.

        return self.rtrn_map[self.rtrn_mode](x, y)
    
    def __next__(self):
        if self.i + 1 >= self.len:
            self.i  = 0
        else:
            self.i += 1
        return self.__getitem__(self.i)
    
    def __rand__(self):
        return self.__getitem__(np.random.randint(self.len))
    
    def _shuffle(self):        
        self.map = np.random.permutation(self.map)

    def on_epoch_end(self):
        if self.shuffle:
            self._shuffle()


class MappedSamplingGen(Sequence):
    def __init__(self, data, **kwargs):
        '''
        A generator that performs data augmentation on midi data that has been
        pre-encoded as input data. The main idea is that instead of computing all
        augmentations beforehand and holding them in memory, we sacrifice some speed
        by computing them as and when they are needed and by doing so allow us to 
        use a dataset that would otherwise not fit in memory.
        
        This generator calculates all possible augmentations and then stores the information
        required to quickly generate them in the dictionary attribute 'self.aug_map'.

        Args:
            n_input - int, number of timesteps used as input
            n_output - int, number of timesteps to predict
            n_sample - int, batch/mini batch size, number of samples to predict at once
            n_time   - int, number of bits that represent time in one-hot encoding
            n_vocab  - int, number of categorical possibilities, number of bits that
                       represent an avnt in one-hot encoding
            dtype           - str, default='float32' data type in the output array
            off_mode        - bool, default=True, True if NoteOffEvents are to be considered
            shuffle         - bool, default=False, if True the input data will be shuffled
                              at the end of each epoch

        Yields:
            x - numpy.ndarray, shape=(n_sample, n_input, n_vocab), the input data
            y - numpy.ndarray, shape=(n_sample, n_vocab), the targets
        '''

        self.n_input         = kwargs.get('n_input', 100)
        self.n_output        = kwargs.get('n_output', 1)
        self.n_sample        = kwargs.get('n_sample', 5)
        self.n_time          = kwargs.get('n_time', 64)
        self.n_vocab         = kwargs.get('n_vocab', 252)
        self.n_teach         = kwargs.get('n_teach', 100)
        self.n_step          = kwargs.get('n_step', self.n_sample)

        self.dtype           = kwargs.get('dtype', 'float32')
        self.off_mode        = kwargs.get('off_mode', True)
        self.shuffle         = kwargs.get('shuffle', False)

        self.n_io         = self.n_input + self.n_output
        self.in_range     = range(self.n_input)
        self.out_range    = range(self.n_output)
        self.batch_loop   = range(self.n_sample)
        self.n_req        = self.n_input + self.n_output + self.n_sample - 1
        
        self.rtrn_map     = {
                             True: lambda x, y: (x, y), 
                             False: lambda x, y: (x, y, [None])
                             }
        self.rtrn_mode    = version.parse(__version__) >= version.parse("2.2.0")

        # purge sequences that are too short
        original_len = len(data)
        data         = [x for x in data if len(x) > self.n_req]
        after_len    = len(data)
        print(f'{original_len - after_len} samples were removed because they are too short. There are {after_len} remaining. \nIf this is not enough, try reducing n_step, n_teach and/or buffer and create a new instance of MappedDataAugGen')

        self.map     = maps.sampling_map(data, self.n_req, self.n_step)
        self.data    = data
        self.i       = 0
        self.len     = len(self.map)
        self.indices = list(range(self.len))

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):

        map_        = self.map[self.indices[idx]]
        sequence    = self.data[map_[0]]
        subsequence = sequence[map_[1]: map_[2]] #.copy()
        x           = np.zeros((self.n_sample, self.n_input, self.n_vocab), dtype=self.dtype)
        y           = np.zeros((self.n_sample, self.n_vocab), dtype=self.dtype)

        for i in self.batch_loop:
            sample = subsequence[i: i + self.n_io]
            x[i, self.in_range, sample[ :self.n_input]] = 1.
            y[i, sample[self.n_input: self.n_io]]       = 1.
            
        return self.rtrn_map[self.rtrn_mode](x, y)
    
    def __next__(self):
        if self.i + 1 >= self.len:
            self.i  = 0
        else:
            self.i += 1
        return self.__getitem__(self.i)
    
    def __rand__(self):
        return self.__getitem__(np.random.randint(self.len))
    
    def _shuffle(self):        
        self.indices = np.random.permutation(self.indices)

    def on_epoch_end(self):
        if self.shuffle:
            self._shuffle()