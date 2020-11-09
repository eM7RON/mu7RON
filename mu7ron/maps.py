import os
import json

import numpy as np
from tqdm import tqdm

from mu7ron import utils


def load_zip_map(dirs):
    """
    Returns a mapping of all the zip files in a list of directories 
    to their containing directories
    """
    zip_map = {d: [] for d in dirs}
    for d in tqdm(dirs):
        for fn in os.listdir(d):
            if fn.lower().endswith(".zip"):
                zip_map[d].append(fn)
    return zip_map


def load_midi_map():
    """
    Returns a dictionary that contains 4 nested dictionaries that map the midi
    standard. Maps:
    'inst' - instrument numbering system
    'drm'  - drum instrument numbering system
    'grp'  - maps instrument numbers to instrument group e.g. violin -> strings
    'tsig' - maps timesignature denominators are encoded as negative powers
             by python-midi
    """
    midi_map = {}
    fnames = (fn + "_map" for fn in ["grp", "inst", "drm", "tsig"])
    for fn in fnames:
        with open(
            os.path.join(os.path.split(__file__)[0], "data", "maps", fn + ".json"), "r"
        ) as file:
            midi_map[fn[:-4]] = json.load(file)
    return midi_map


def create_q_map(x: int, q: int, encode: bool = True, decode: bool = True) -> dict:
    """
    Creates a mapping of the numbered steps of a quantized range.
    For example if the range(0, 10) is quantized into steps [0, 4, 8]
    they will be numbered [0, 1, 2] an encode mapping will be the dictionary
    {0: 0, 4: 1, 8: 2} and a decode mapping will be {0: 0, 1: 4, 2: 8}.
    """
    q_map = {}
    if encode:
        q_map["encode"] = {o: i for i, o in enumerate(utils.dynamic_range(x, q))}
    if decode:
        q_map["decode"] = {i: o for i, o in enumerate(utils.dynamic_range(x, q))}
    return q_map


def semantic_map(n_time, n_pitch, n_vel, n_pulse, off_mode=False):
    """
    Creates a python dictionary that contains two nested dictionaries:
    'voc' maps avnts into partitions. Partitions have been created manually
    by grouping by type e.g. velocity, time or pitch. The second dictionary
    'sem' maps the semantic partitions to lists of avnts that belong to 
    those partitions
    """

    def to_map(n, map_, lo, hi):
        range_ = list(range(lo, hi))
        map_["sem"][n] = range_
        for i in range_:
            map_["voc"][i] = n
        n += 1
        return n, map_

    ret = {"voc": {}, "sem": {}}
    n = 0
    if off_mode:
        n_pitch = 128 * 2
    else:
        n_pitch = 128
    n_t_p = n_time + n_pitch
    n_t_p_v = n_time + n_pitch + n_vel
    n_vocab = n_time + n_pitch + n_vel + n_pulse

    n, ret = to_map(n, ret, 0, n_time)
    if off_mode:
        n_on = n_time + 128
        n, ret = to_map(n, ret, n_time, n_on)
        n, ret = to_map(n, ret, n_on, n_t_p)
    else:
        n, ret = to_map(n, ret, n_time, n_t_p)
    n, ret = to_map(n, ret, n_t_p, n_t_p + 1)
    n, ret = to_map(n, ret, n_t_p + 1, n_t_p_v)
    n, ret = to_map(n, ret, n_t_p_v, n_vocab)
    return ret


def augmentation_map(
    data, n_time, off_mode, lo_lim=40, hi_lim=100, t_range=[0.95, 1.0, 1.05]
):
    """
    Creates a list where each element is a 3-tuple containing an index (i), pitch 
    adjustment and time adjustment describing a augmentations to be performend on
    data[i]. The idea is that augmentations can be performed on the fly which reduces
    mempry footprint.
    """
    ret = []
    n_t_p = n_time + 128
    lo_lim += n_time
    hi_lim += n_time
    for i, x in enumerate(data):
        idx = (n_time <= x) & (x < n_t_p)
        if sum(idx):
            pitches = x[idx]
            min_ = np.maximum(np.min(pitches), lo_lim)
            max_ = np.minimum(np.max(pitches), hi_lim)
        else:
            min_, max_ = lo_lim, hi_lim
        hi = np.abs(max_ - hi_lim) + 1
        lo = -np.abs(min_ - lo_lim)
        if lo < hi:
            for p_aug in range(lo, hi):
                for t_aug in t_range:
                    ret.append(
                        {"orig_sample_idx": i, "pitch_aug": p_aug, "time_aug": t_aug}
                    )
    return ret


def teacher_forcing_map(
    data, aug_map, n_time, n_vocab, n_input, n_output, n_sample, n_teach, n_step, buffer
):
    """
    Expands an augmentation map to produce indexes that will allow 
    targets values of previous outputs to be used as inputs
    """
    n_io = n_input + n_output
    n_req = n_io + n_sample - 1
    teach_range = range(0, n_teach * n_step, n_step)
    tf_map = []

    for map_ in aug_map:
        sample = data[map_["orig_sample_idx"]]
        n = len(sample)
        i = np.random.randint(n - n_req - n_teach * n_step - buffer)
        j = i + n_req + n_teach * n_step + buffer
        for k in teach_range:
            new_map_ = {"strt_idx": i, "end_idx": j, "teach_idx": k, **map_}
            new_map_["id"] = (
                new_map_["orig_sample_idx"],
                map_["pitch_aug"],
                map_["time_aug"],
            )
            tf_map.append(new_map_)

    return tf_map


def sub_sample_map(data, aug_map, n_input, n_output, n_teach, buffer):
    """
    Expands an augmentation map to produce indexes that will allow 
    targets values of previous outputs to be used as inputs
    """
    n_io = n_input + n_output
    n_req = n_io
    teach_range = range(n_teach)
    tf_map = []

    for map_ in aug_map:
        sample = data[map_["orig_sample_idx"]]
        n = len(sample)
        i = np.random.randint(n - n_io - n_teach - buffer)
        j = i + n_req + n_teach + buffer
        new_map_ = {"strt_idx": i, "end_idx": j, **map_}
        tf_map.append(new_map_)

    return tf_map


def stateless_balanced_sampling_map(
    data, n_input, n_output, n_sample, n_example, n_vocab
):
    """
    Returns a map of subsequences in the data with targets that represent each possible
    outcome/choice in equal proportions.
    Args:
        n_input   - int, number of input times steps
        n_output  - int, number of output times steps
        n_sample  - int, batch size, number of samples in a batch
        n_example - int, number of examples of each avnt/choice
        n_vocab   - int, number of bits used to encode a single avnt, number of possible
                    outcomes/choices
    """
    n_req = n_input + n_output + n_sample - 1
    ret = []
    data = [x for x in data if len(x) > n_req]

    for i, x in enumerate(data):
        n = len(x)
        a_range = np.arange(n)
        for j in range(n_vocab):
            avnt_idxs = x[n_req:] == j
            n_choice = np.sum(avnt_idxs)
            if n_choice:
                n_choice = min(n_choice, n_example)
                idx = a_range[n_req:]  # [idx]
                idx = idx[avnt_idxs]
                idx = np.random.choice(idx, n_choice, replace=False)
                for k in idx:
                    ret.append(
                        {
                            "orig_sample_idx": i,
                            "strt_idx": k - n_req + 1,
                            "end_idx": k + 1,
                        }
                    )
    return ret


def balanced_sampling_map(data, n_input, n_output, n_example, n_vocab):
    """
    Returns a map of subsequences in the data with targets that represent each possible
    outcome/choice in equal proportions.
    Args:
        n_input   - int, number of input times steps
        n_output  - int, number of output times steps
        n_example - int, number of examples of each avnt/choice
        n_vocab   - int, number of bits used to encode a single avnt, number of possible
                    outcomes/choices
    """
    n_req = n_input + n_output
    ret = []
    data = [x for x in data if len(x) > n_req]

    for i, x in enumerate(data):
        n = len(x)
        a_range = np.arange(n)
        for j in range(n_vocab):
            avnt_idxs = x[n_req:] == j
            n_choice = np.sum(avnt_idxs)
            if n_choice:
                n_choice = min(n_choice, n_example)
                idx = a_range[n_req:]  # [idx]
                idx = idx[avnt_idxs]
                idx = np.random.choice(idx, n_choice, replace=False)
                for k in idx:
                    ret.append(
                        {
                            "orig_sample_idx": i,
                            "strt_idx": k - n_req + 1,
                            "end_idx": k + 1,
                        }
                    )
    return ret


def sampling_map(data, n_req, n_step):
    """
    Creates a list containing 3-tuples where each element can be used to
    index a specific sequence and then index start and end positions for 
    a sample of that sequence. Each sample will be n_req in length which
    should be long enough to create a batch. We sample a sequence at 
    intervals of n_step apart which, as a default, could be n_sample
    (batch size) apart.
    """
    map_ = []
    for i, sequence in enumerate(data):
        for j in range(0, len(sequence) - n_req, n_step):
            map_.append((i, j, j + n_req))
    return map_
