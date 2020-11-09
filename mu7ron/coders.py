from collections import defaultdict
import operator

import numpy as np
import midi

from tensorflow.keras.models import load_model
from tqdm import tqdm

from mu7ron import maps
from mu7ron import notes
from mu7ron import temporal
from mu7ron import utils


def process_timestep(
    master_sequence,
    timestep,
    tick,
    bpm,
    mspt,
    q,
    n_time,
    n_t_p,
    n_t_p_v,
    q_map,
    p_map,
    time_encoder,
    sort_velocity,
    sort_pitch,
    sort_pulse,
):
    """
    An auxiliary function for 'categorize_input'. Once all events, which occur simultaneously, at 
    a particular timestep have been collected by encode_input, 'process_timestep' will process
    these events, encoding them as a sequence of integers. The master function 'encode_input' 
    then continues collecting events that occur simultaneously in the next timestep.
    """
    bpm_flag = False
    v_map = defaultdict(list)
    pulses = []  # Added note division pulses
    for evnt in timestep:
        if isinstance(evnt, midi.SetTempoEvent):
            bpm = evnt.get_bpm()
            bpm_flag = True
        elif p_map is not None and isinstance(evnt, notes.Note):
            pulses.append(n_t_p_v + pulse_map[type(evnt)])
        else:  #
            pitch = n_time + evnt.data[0]
            if isinstance(evnt, midi.NoteOffEvent):
                pitch += 128
            v_map[n_t_p + q_map["encode"][utils.quantize(evnt.data[1], q)]].append(
                pitch
            )

    # velocity will be in ascending order as to make it easier to predict
    velocities = list(v_map.keys())
    if sort_velocity:
        velocities.sort()
    if sort_pulse:
        pulses.sort()
    timestep = pulses
    for v in velocities:
        # we will now process each velocity step or 'velocity bubble'
        timestep.append(v)
        # we also sort pitches and eliminate duplicates
        for pitch in sorted(set(v_map[v])) if sort_pitch else set(v_map[v]):
            timestep.append(pitch)

    if mspt is None or bpm_flag:
        mspt = utils.bpm_to_mspt(bpm)
    if timestep:
        master_sequence.extend(time_encoder(tick * mspt))  # tick * (ms / tick) = ms
        master_sequence.extend(timestep)
        timestep = []
    return [master_sequence, timestep, tick, bpm, mspt]


def categorize_input(x, q, n_time, off_mode, time_encoder, ekwa, **kwargs):
    """
    This function is the inverse of categorize_output.
    This function will encode a midi sequence in the form of a midi.Track/midi.Pattern
    (with 1 track) or a MuGen.utils.MidiObj into a encoded/serialized sequence of integers. 
    A midi.events object, often referred to in this code as 'evnt' will be encoded as an 
    integer, an 'avnt'. The evnt/avnt terminology is used to imitate efferent/afferent neuron
    terminology in neuroscience.

    Args:
        x             - list/midi.Track/midi.Pattern, iterable containing midi.Events
        q             - int, quantization factor of velocity
        n_time        - int, number of bits in one-hot encoding that are used for time slips
        off_mode      - bool, True if NoteOffEvents are included in input sequence
        time_encoder  - int, a function to encode milliseconds
        ekwa          - dict, keyword arguments for the encoder
        
        n_vel         - int, number of bits in one-hot encoding that are used for velocity, 
                        this value is dependent on q
        q_map         - dict, maps quantized velocities to encoding
        p_map         - dict, maps added pulses/note divisions to encoding
        asarray       - bool, default=False, whether to return output as a numpy array
        dtype         - str, default='int', data type of the returned array
        bpm           - int, default=120, tempo in Beats Per Minute, initial tempo of the input
                        sequence, is updated dynamically as tempo changed in input sequence
        sort_velocity - bool, default=False, whether velocity values should be in sorted order 
        sort_pitch    = bool, default=False, whether pitch values should be in sorted order 
        sort_pulse    = bool, default=False, whether pulse values should be in sorted order 

    Vars:
        n_pitch       - int, number of bits in one-hot encoding that are used for pitch 
                        i.e. NoteOnEvents, if off_mode is True an additional 128 bits are
                        used for NoteOffEvents otherwise x should only contain NoteOnEvents
                        with velocity=0 to signify NoteOffEvents
        n_t_p         - int, n_time + n_pitch
        n_t_p_v       - int, n_time + n_pitch + n_vel
        n_vocab       - int, n_time + n_pitch + n_vel + n_pulse, total number of bits used 
                        for one-hot encoding
        mspt          - float, default=None, milliseconds per tick, updated dynamically as we iterate 
                        through the input sequence (x)
        tick          - int, holds the current tick as we iterate through the input sequence (x)
        timestep      - list, as we iterate through the input sequence all events that occur 
                        simultaneously, in a single 'timestep', will be collected here before being
                        processed/encoded/serialized

    Returns:
        ret           - list[int]/np.ndarray[int] - encoded and serialized midi sequence
    """

    if isinstance(x, utils.MidiObj):
        x = x.ptrn

    n_vel = kwargs.get("n_vel", utils.dynamic_order(128, q))
    q_map = kwargs.get("q_map", maps.create_q_map(128, q))
    p_map = kwargs.get("p_map", None)
    asarray = kwargs.get("asarray", False)
    dtype = kwargs.get("dtype", "int")
    sort_velocity = kwargs.get("sort_velocity", False)
    sort_pitch = kwargs.get("sort_pitch", False)
    sort_pulse = kwargs.get("sort_pulse", False)
    bpm = kwargs.get("bpm", 120)  # beats per minute

    n_pitch = 128 * 2 if off_mode else 128
    n_pulse = len(p_map.keys()) if p_map is not None else 0
    n_t_p = n_time + n_pitch
    n_t_p_v = n_time + n_pitch + n_vel
    n_vocab = n_time + n_pitch + n_vel + n_pulse  # n_t_p_v_p

    mspt = None  # miliseconds per tick
    tick = 0
    timestep = []  # timestep/time bubble
    ret = []  # output sequence

    t_encoder = lambda x: time_encoder(x, **ekwa)
    args = [ret, timestep, tick, bpm, mspt]
    static_args = (
        q,
        n_time,
        n_t_p,
        n_t_p_v,
        q_map,
        p_map,
        t_encoder,
        sort_velocity,
        sort_pitch,
        sort_pulse,
    )
    _processor = lambda x: process_timestep(*x, *static_args)

    for evnt in utils.evnt_gen(x):
        if evnt.tick != 0:
            args = _processor(args)
            args[2] = tick = evnt.tick
        args[1].append(evnt)
    ret = _processor(args)[0]
    if asarray:
        ret = np.array(ret, dtype=dtype)
    return ret


def decategorize_output(x, q, n_time, off_mode, time_decoder, dkwa, **kwargs):
    """
    This function is the inverse of categorize_input . 
    Decodes an iterable containing avnts into a midi.Pattern containing midi.events,
    'evnts'.

    Args:
        x             - list[int, ..., int], containing encoded/serialized midi sequence
        q             - int, quantization factor of velocity
        n_time        - int, number of bits used to encode time slips
        off_mode      - bool, True if NoteOffEvents are included in input sequence
        time_decoder  - int, a function to decode to milliseconds
        dkwa          - dict, keyword arguments for the decoder

        n_vel         - int, number of bits in one-hot encoding that are used for velocity, 
                        this value is dependent on q
        q_map         - dict, maps quantized velocities to encoding
        p_map         - dict, maps added pulses/note divisions to encoding
        dtype         - str, default='int', data type of the returned array
        bpm           - int, default=120, tempo in Beats Per Minute, initial tempo of the input
                        sequence, is updated dynamically as tempo changed in input sequence

    Vars:
        n_pitch       - int, number of bits in one-hot encoding that are used for pitch 
                        i.e. NoteOnEvents, if off_mode is True an additional 128 bits are
                        used for NoteOffEvents otherwise x should only contain NoteOnEvents
                        with velocity=0 to signify NoteOffEvents
        n_t_p         - int, n_time + n_pitch
        n_t_p_v       - int, n_time + n_pitch + n_vel
        n_vocab       - int, n_time + n_pitch + n_vel + n_pulse, total number of bits used 
                        for one-hot encoding
        mspt          - float, default=None, milliseconds per tick, updated dynamically as we iterate 
                        through the input sequence (x)
        tick          - int, holds the current tick as we iterate through the input sequence (x)
    
    Returns:
        ptrn          - midi.Pattern, a playable midi sequence
    """
    n_vel = kwargs.get("n_vel", utils.dynamic_order(128, q))
    q_map = kwargs.get("q_map", maps.create_q_map(128, q, encode=False, decode=True))
    p_map = kwargs.get("p_map", None)
    dtype = kwargs.get("dtype", "int")
    bpm = kwargs.get("bpm", 120)
    include_end = kwargs.get("include_end", False)

    n_pitch = 128 * 2 if off_mode else 128
    n_on = n_time + 128
    n_pulse = len(p_map.keys()) if p_map is not None else 0
    n_t_p = n_time + n_pitch
    n_t_p_v = n_time + n_pitch + n_vel
    n_vocab = n_time + n_pitch + n_vel + n_pulse  # n_t_p_v_p

    istime = lambda avnt: avnt < n_time
    isvelocity = lambda avnt: n_t_p <= avnt < n_t_p_v
    ispitch = lambda avnt: n_time <= avnt < n_t_p
    isnoteon = lambda avnt: n_time <= avnt < n_time + 128

    tick = 0
    velocity = 0
    tick_flag = False
    mspt = utils.bpm_to_mspt(bpm)
    cache = []
    ptrn = midi.Pattern(format=1, tick_relative=True, resolution=480)
    trck = midi.Track(tick_relative=True)
    ptrn.append(trck)

    for avnt in x:
        if isvelocity(avnt):
            velocity = q_map["decode"][avnt - n_t_p]
            velocity_flag = True
        elif ispitch(avnt):
            if tick_flag:
                tick = int(round(time_decoder(cache, **dkwa) / mspt))
                cache = []
                tick_flag = False
            else:
                tick = 0
            if isnoteon(avnt):
                evnt = midi.NoteOnEvent(tick=tick, data=[avnt - n_time, velocity])
            else:
                evnt = midi.NoteOffEvent(tick=tick, data=[avnt - n_on, velocity])
            trck.append(evnt)
            velocity_flag = False
        elif istime(avnt):
            cache.append(avnt)
            tick_flag = True
        else:
            pass  # potential to add percussion or 'pulse' events
    if include_end:
        trck.append(midi.EndOfTrackEvent(tick=200))
    return ptrn


def beam_search(
    sequence, n_beam, model=None, path_to_model=None, custom_objs=None, n_iter=3
):
    """
    Beam search is a heuristic search algorithm that explores a graph by expanding the most
    promising node in a limited set. Beam search is an optimization of best-first search that 
    reduces its memory requirements.
    https://en.wikipedia.org/wiki/Beam_search

    Args:
        sequence      - categorized midi sequence to be used as seed 
        n_beam        - int, search depth, number of solutions to explore
        model         - tensorflow, LSTM/GRU
        path_to_model - str, location of the model in storage
        custom_objs   - object, any non-standard objects required by the model such as weights or a custom loss
                        function
        n_iter        - int, default=3, number of iterations to run search
    Returns:
        main_beam     - list[[probability, one_hot_sequence], ..., [probability, one_hot_sequence]], shape=(n_beam, 2)
                        n_beam best sequences discovered
    """
    n_input, n_vocab = sequence.shape
    assert (
        n_beam < n_vocab
    ), f"n_beam is not smaller than n_vocab i.e. {n_beam} not smaller than {n_vocab}"
    b_range = range(n_beam)
    in_range = range(n_input)
    v_range = range(n_vocab)
    main_beam = []
    model = load_model(path_to_model, custom_objects=custom_objs)

    y_pred = model.predict(sequence.reshape(1, n_input, n_vocab))
    idx = y_pred.argsort()[0, -n_beam:]
    one_hot = np.zeros((n_beam, n_vocab), dtype="float32")
    one_hot[b_range, idx] = 1.0
    probs = y_pred[0, idx]

    # main_beam = [[probability, one_hot_sequence], ..., [probability, one_hot_sequence]]

    for i in b_range:
        main_beam.append([probs[i], np.vstack([sequence.copy(), one_hot[i]])])

    for i in tqdm(range(1, n_iter)):
        temp_beam = []
        batch = np.array([seq[i:] for p, seq in main_beam])
        y_pred = model.predict(batch)
        sorted_p = y_pred.argsort()

        for j in b_range:
            idx = sorted_p[j, -n_beam:]
            one_hot = np.zeros((n_beam, n_vocab), dtype="float32")
            one_hot[b_range, idx] = 1.0
            probs = y_pred[j, idx]
            #  make extra beam with 3-tuple = (prob., next avnt, also keep j to later match up with parent sequence)
            for k in b_range:
                temp_beam.append([main_beam[j][0] + probs[k], one_hot[k], j])
        # keep n_beam best
        temp_beam = sorted(temp_beam, key=operator.itemgetter(0))[-n_beam:]
        # match up with parent sequences
        main_beam = [
            [
                temp_beam[j][0],
                np.vstack([main_beam[temp_beam[j][2]][1].copy(), temp_beam[j][1]]),
            ]
            for j in b_range
        ]

        min_ = min(main_beam, key=operator.itemgetter(0))[0]
        for j in b_range:
            main_beam[j][0] -= min_

    return main_beam


#
# Experimental !
#

def process_timestep_with_drums(
    master_sequence,
    timestep,
    tick,
    bpm,
    mspt,
    q,
    n_time,
    n_pitch,
    n_drum,
    n_note,
    n_t_n,
    n_t_n_v,
    p_map,
    q_map,
    time_encoder,
    sort_velocity,
    sort_pitch,
    sort_pulse,
):
    """
    An auxiliary function for 'categorize_input'. Once all events, which occur simultaneously, at 
    a particular timestep have been collected by encode_input, 'process_timestep' will process
    these events, encoding them as a sequence of integers. The master function 'encode_input' 
    then continues collecting events that occur simultaneously in the next timestep.
    """
    bpm_flag = False
    v_map = defaultdict(list)
    pulses = []  # Added note division pulses
    for evnt in timestep:
        if isinstance(evnt, midi.SetTempoEvent):
            bpm = evnt.get_bpm()
            bpm_flag = True
        elif p_map is not None and isinstance(evnt, notes.Note):
            pulses.append(n_t_n_v + pulse_map[type(evnt)])
        else:
            if evnt.channel == 9:
                note = n_time + n_pitch + evnt.data[0] - 35
            else:
                note = n_time + evnt.data[0]
            if isinstance(evnt, midi.NoteOffEvent):
                note += 175
            v_map[n_t_n + q_map["encode"][utils.quantize(evnt.data[1], q)]].append(note)

    # velocity will be in ascending order as to make it easier to predict
    velocities = list(v_map.keys())
    if sort_velocity:
        velocities.sort()
    if sort_pulse:
        pulses.sort()
    timestep = pulses
    for v in velocities:
        # we will now process each velocity step or 'velocity bubble'
        timestep.append(v)
        # we also sort pitches and eliminate duplicates
        for pitch in sorted(set(v_map[v])) if sort_pitch else set(v_map[v]):
            timestep.append(pitch)

    if mspt is None or bpm_flag:
        mspt = utils.bpm_to_mspt(bpm)
    if timestep:
        master_sequence.extend(time_encoder(tick * mspt))  # tick * (ms / tick) = ms
        master_sequence.extend(timestep)
        timestep = []
    return [master_sequence, timestep, tick, bpm, mspt]


def categorize_input_with_drums(
    x, q, n_time, off_mode, drm_mode, time_encoder, ekwa, **kwargs
):
    """
    This function is the inverse of categorize_output.
    This function will encode a midi sequence in the form of a midi.Track/midi.Pattern
    (with 1 track) or a MuGen.utils.MidiObj into a encoded/serialized sequence of integers. 
    A midi.events object, often referred to in this code as 'evnt' will be encoded as an 
    integer, an 'avnt'. The evnt/avnt terminology is used to imitate efferent/afferent neuron
    terminology in neuroscience.

    Args:
        x             - list/midi.Track/midi.Pattern, iterable containing midi.Events
        q             - int, quantization factor of velocity
        n_time        - int, number of bits in one-hot encoding that are used for time slips
        off_mode      - bool, True if NoteOffEvents are included in input sequence
        time_encoder  - int, a function to encode milliseconds
        ekwa          - dict, keyword arguments for the encoder
        
        n_vel         - int, number of bits in one-hot encoding that are used for velocity, 
                        this value is dependent on q
        q_map         - dict, maps quantized velocities to encoding
        p_map         - dict, maps added pulses/note divisions to encoding
        asarray       - bool, default=False, whether to return output as a numpy array
        dtype         - str, default='int', data type of the returned array
        bpm           - int, default=120, tempo in Beats Per Minute, initial tempo of the input
                        sequence, is updated dynamically as tempo changed in input sequence
        sort_velocity - bool, default=False, whether velocity values should be in sorted order 
        sort_pitch    = bool, default=False, whether pitch values should be in sorted order 
        sort_pulse    = bool, default=False, whether pulse values should be in sorted order 

    Vars:
        n_pitch       - int, number of bits in one-hot encoding that are used for pitch 
                        i.e. NoteOnEvents, if off_mode is True an additional 128 bits are
                        used for NoteOffEvents otherwise x should only contain NoteOnEvents
                        with velocity=0 to signify NoteOffEvents
        n_drum        - int, number of bits used for one-hot encoding drum data
        n_t_n         - int, n_time + n_note
        n_t_n_v       - int, n_time + n_note + n_vel
        n_vocab       - int, n_time + n_note + n_vel + n_pulse, total number of bits used 
                        for one-hot encoding
        mspt          - float, default=None, milliseconds per tick, updated dynamically as we iterate 
                        through the input sequence (x)
        tick          - int, holds the current tick as we iterate through the input sequence (x)
        timestep      - list, as we iterate through the input sequence all events that occur 
                        simultaneously, in a single 'timestep', will be collected here before being
                        processed/encoded/serialized

    Returns:
        ret           - list[int]/np.ndarray[int] - encoded and serialized midi sequence
    """

    if isinstance(x, utils.MidiObj):
        x = x.ptrn

    n_vel = kwargs.get("n_vel", utils.dynamic_order(128, q))
    p_map = kwargs.get("p_map", None)
    q_map = kwargs.get("q_map", maps.create_q_map(128, q))
    asarray = kwargs.get("asarray", False)
    dtype = kwargs.get("dtype", "int")
    sort_velocity = kwargs.get("sort_velocity", False)
    sort_pitch = kwargs.get("sort_pitch", False)
    sort_pulse = kwargs.get("sort_pulse", False)
    bpm = kwargs.get("bpm", 120)  # beats per minute

    n_pitch = 128
    n_drum = 47 if drm_mode else 0
    n_note = 2 * (n_pitch + n_drum) if off_mode else n_pitch + n_drum
    n_pulse = len(p_map.keys()) if p_map is not None else 0
    n_t_n = n_time + n_note
    n_t_n_v = n_time + n_note + n_vel
    n_vocab = n_time + n_note + n_vel + n_pulse  # n_t_n_v_p

    mspt = None  # miliseconds per tick
    tick = 0
    timestep = []  # timestep/time bubble
    ret = []  # output sequence

    t_encoder = lambda x: time_encoder(x, **ekwa)
    args = [ret, timestep, tick, bpm, mspt]
    static_args = (
        q,
        n_time,
        n_pitch,
        n_drum,
        n_note,
        n_t_n,
        n_t_n_v,
        p_map,
        q_map,
        t_encoder,
        sort_velocity,
        sort_pitch,
        sort_pulse,
    )
    _processor = lambda x: process_timestep_with_drums(*x, *static_args)

    for evnt in utils.evnt_gen(x):
        if evnt.tick != 0:
            args = _processor(args)
            args[2] = tick = evnt.tick
        args[1].append(evnt)
    ret = _processor(args)[0]
    if asarray:
        ret = np.array(ret, dtype=dtype)
    return ret


def decategorize_output_with_drums(
    x, q, n_time, off_mode, drm_mode, time_decoder, dkwa, **kwargs
):
    """
    This function is the inverse of categorize_input . 
    Decodes an iterable containing avnts into a midi.Pattern containing midi.events,
    'evnts'.

    Args:
        x             - list[int, ..., int], containing encoded/serialized midi sequence
        q             - int, quantization factor of velocity
        n_time        - int, number of bits used to encode time slips
        off_mode      - bool, True if NoteOffEvents are included in input sequence
        time_decoder  - int, a function to decode to milliseconds
        dkwa          - dict, keyword arguments for the decoder

        n_vel         - int, number of bits in one-hot encoding that are used for velocity, 
                        this value is dependent on q
        q_map         - dict, maps quantized velocities to encoding
        p_map         - dict, maps added pulses/note divisions to encoding
        dtype         - str, default='int', data type of the returned array
        bpm           - int, default=120, tempo in Beats Per Minute, initial tempo of the input
                        sequence, is updated dynamically as tempo changed in input sequence

    Vars:
        n_pitch       - int, number of bits in one-hot encoding that are used for pitch 
                        i.e. NoteOnEvents, if off_mode is True an additional 128 bits are
                        used for NoteOffEvents otherwise x should only contain NoteOnEvents
                        with velocity=0 to signify NoteOffEvents
        n_t_n         - int, n_time + n_note
        n_t_n_v       - int, n_time + n_note + n_vel
        n_vocab       - int, n_time + n_note + n_vel + n_pulse, total number of bits used 
                        for one-hot encoding
        mspt          - float, default=None, milliseconds per tick, updated dynamically as we iterate 
                        through the input sequence (x)
        tick          - int, holds the current tick as we iterate through the input sequence (x)
    
    Returns:
        ptrn          - midi.Pattern, a playable midi sequence
    """
    n_vel = kwargs.get("n_vel", utils.dynamic_order(128, q))
    q_map = kwargs.get("q_map", maps.create_q_map(128, q, encode=False, decode=True))
    p_map = kwargs.get("p_map", None)
    dtype = kwargs.get("dtype", "int")
    bpm = kwargs.get("bpm", 120)
    include_end = kwargs.get("include_end", False)

    n_pitch = 128
    n_drum = 47 if drm_mode else 0
    n_note = 2 * (n_pitch + n_drum) if off_mode else n_pitch + n_drum
    n_pulse = len(p_map.keys()) if p_map is not None else 0
    n_t_p = n_time + n_pitch
    n_t_p_d = n_time + n_pitch + n_drum
    n_t_p_d_p = n_t_p_d + n_pitch

    n_t_n = n_time + n_note
    n_t_n_v = n_time + n_note + n_vel
    n_vocab = n_time + n_note + n_vel + n_pulse  # n_t_n_v_p

    istime = lambda avnt: avnt < n_time
    isvelocity = lambda avnt: n_t_n <= avnt < n_t_n_v
    isnote = lambda avnt: n_time <= avnt < n_t_n
    ison = lambda avnt: n_time <= avnt < n_t_p_d

    if off_mode:
        ispitch = lambda avnt: n_time <= avnt < n_t_p or n_t_p_d <= avnt < n_t_p_d_p
    else:
        ispitch = lambda avnt: n_time <= avnt < n_t_p

    tick = 0
    velocity = 0
    tick_flag = False
    mspt = utils.bpm_to_mspt(bpm)
    cache = []
    ptrn = midi.Pattern(format=1, tick_relative=True, resolution=480)
    trck = midi.Track(tick_relative=True)
    ptrn.append(trck)

    for avnt in x:
        if isvelocity(avnt):
            velocity = q_map["decode"][avnt - n_t_n]
            velocity_flag = True
        elif isnote(avnt):
            if tick_flag:
                tick = int(round(time_decoder(cache, **dkwa) / mspt))
                cache = []
                tick_flag = False
            else:
                tick = 0
            if ispitch(avnt):
                channel = 0
                adj_on = n_time
                adj_off = n_t_p_d
            else:
                channel = 9
                adj_on = n_t_p - 35
                adj_off = n_t_p_d_p - 35
            if ison(avnt):
                evnt = midi.NoteOnEvent(
                    tick=tick, data=[avnt - adj_on, velocity], channel=channel
                )
            else:
                evnt = midi.NoteOffEvent(
                    tick=tick, data=[avnt - adj_off, velocity], channel=channel
                )
            trck.append(evnt)
            velocity_flag = False

        elif istime(avnt):
            cache.append(avnt)
            tick_flag = True
        else:
            pass  # potential to add percussion or 'pulse' events
    if include_end:
        trck.append(midi.EndOfTrackEvent(tick=200))
    return ptrn
