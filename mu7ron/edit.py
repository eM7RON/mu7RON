import copy

import midi

from mu7ron import analyze
from mu7ron import utils


def copy_ptrn(old_ptrn, trck=None):
    """
    Returns a shallow cop of a midi.Pattern
    """
    if trck is None:
        new_ptrn = midi.Pattern(
            format=old_ptrn.format,
            resolution=old_ptrn.resolution,
            tick_relative=old_ptrn.tick_relative,
        )
    else:
        new_ptrn = midi.Pattern(
            [trck],
            format=old_ptrn.format,
            resolution=old_ptrn.resolution,
            tick_relative=old_ptrn.tick_relative,
        )
    return new_ptrn


def finalize_midi_sequence(
    ptrn, name="Track 1", inst=0, channel=0, tsig=(4, 4), bpm=120
):
    """
    Makes a geenrated midi sequnce valid for most players 
    by adding TimeSignature, SetTempo and EndOfTrack events.
    """
    has_tsig = False
    has_tempo = False
    has_end = False
    for evnt in utils.evnt_gen(ptrn):
        if isinstance(evnt, midi.SetTempoEvent):
            has_tsig = True
        if isinstance(evnt, midi.TimeSignatureEvent):
            has_tempo = True
        if isinstance(evnt, midi.EndOfTrackEvent):
            has_end = True
    if not has_tsig:
        ptrn[0].insert(0, midi.TimeSignatureEvent(tick=0))
        ptrn[0][0].data[:2] = tsig
        ptrn[0][0].data[2:] = 24, 8
    if not has_tempo:
        ptrn[0].insert(0, midi.SetTempoEvent(tick=0))
        ptrn[0][0].set_bpm(bpm)
    if not has_end:
        ptrn[0].append(midi.EndOfTrackEvent(tick=0))
    ptrn[0].insert(0, midi.TrackNameEvent(tick=0, text=name, data=[]))
    ptrn[0].insert(0, midi.ProgramChangeEvent(tick=0, channel=channel, data=[inst]))
    return ptrn


def filter_ptrn_of_evnt_typs(old_ptrn, typs_2_keep=tuple(), f=copy.deepcopy):
    """
    Filter out unwanted event types from a midi.Pattern
    Returns a new midi.Pattern (not in-place) exclusively containing events 
    types contained in typs_2_keep. 
    """
    # It's necessary to convert ticks to abs. when removing
    # events otherwise timing issues occur
    old_ptrn.make_ticks_abs()
    new_ptrn = copy_ptrn(old_ptrn)
    for old_trck in old_ptrn:
        new_trck = midi.Track(tick_relative=False)  # old_trck.tick_relative
        for old_evnt in old_trck:
            if isinstance(old_evnt, typs_2_keep):
                new_evnt = f(old_evnt)
                new_trck.append(new_evnt)
        if len(new_trck):
            new_ptrn.append(new_trck)
    # We convert to rel. so that playback sounds correct
    old_ptrn.make_ticks_rel()
    new_ptrn.make_ticks_rel()
    return new_ptrn


def filter_ptrn_of_insts(
    old_ptrn, condition=lambda x: x.data[0] == 47 or x.data[0] > 95
):
    """
    Filter out tracks that contain unwanted instruments i.e. 47 for timpani
    """
    new_ptrn = copy_ptrn(old_ptrn)
    for trck in old_ptrn:
        keep = True
        for evnt in trck:
            if isinstance(evnt, midi.ProgramChangeEvent) and condition(evnt):
                keep = False
                break
        if keep:
            new_ptrn.append(trck)
    return new_ptrn


def filter_ptrn_of_percussion(
    x, typs=(midi.NoteOnEvent, midi.NoteOffEvent), channel=9, res=480
):
    """
    Filter out note events that use channel 9 which corresponds to percussion
    """
    if isinstance(x, utils.MidiObj):
        x = x.ptrn
    x.make_ticks_abs()
    if isinstance(x, midi.Pattern):
        new_ptrn = copy_ptrn(x)
    else:
        new_ptrn = midi.Pattern(tick_relative=False, resolution=res)

    for old_trck in x:
        new_trck = midi.Track(tick_relative=False)
        new_ptrn.append(new_trck)
        for old_evnt in old_trck:
            keep = True
            if isinstance(old_evnt, typs) and old_evnt.channel == channel:
                keep = False
            if keep:
                new_evnt = copy.deepcopy(old_evnt)
                new_trck.append(new_evnt)
    x.make_ticks_rel()
    new_ptrn.make_ticks_rel()
    return new_ptrn


def filter_data_of_insts(data, to_remove=set()):
    """
    Remove midi.Pattern or MidiObj from an iterable data container if they contain
    instrument numbers in 'to_remove'. These numbers are same as midi standard -1.
    """
    ret = []
    for x in data:
        keep = True
        for evnt in utils.evnt_gen(x):
            if isinstance(evnt, midi.ProgramChangeEvent) and evnt.data[0] in to_remove:
                keep = False
                break
        if keep:
            ret.append(x)
    return ret


def filter_data_of_empty_ptrn(data):
    """
    Filters an iterable containing MidiObj of MidiObj which have ptrn with no
    midi.NoteOn or midi.NoteOff events
    """
    return [
        ptrn
        for ptrn in data
        if analyze.has_evnt_typ(ptrn, (midi.NoteOnEvent, midi.NoteOffEvent))
    ]


def filter_data_of_ptrn_of_length(x, l=10):
    """
    Filter a data iterable of ptrns shorter than l
    """
    if isinstance(x[0], utils.MidiObj):
        func = lambda v: v.ptrn[0]
    elif isinstance(x[0], midi.Pattern):
        func = lambda v: v[0]
    else:
        func = lambda v: v
    return [o for o in x if len(func(o)) > l]


def filter_data_custom(data, condition=lambda x: True):
    """
    A custom filter for filtering a iterable data container which can be given a
    condition function as 'condition'
    """
    ret = []
    for x in data:
        if condition(x):
            ret.append(x)
    return ret


def quantize_typ_attr(x, q, typ, func):
    """
    Reduces the number of different available velocity values
    Args:
        x     - MidiObj or midi.Pattern
        typ   - midi.events type, i.e. midi.NoteOnEvent
        q     - int, quantization factor
    """
    for evnt in utils.evnt_gen(x):
        if isinstance(evnt, typ):
            evnt.data[1] = utils.quantize(func(evnt), q)
    return x


def normalize_resolution(ptrn, res=480):
    """
    Normalizes the resolution and adjusts the tempo of a midi.Pattern to a given value whilst
    preserving the speed/rate of the music.
    """
    adj = res / ptrn.resolution
    ptrn.resolution = int(round(ptrn.resolution * adj))
    for trck in ptrn:
        for evnt in trck:
            if isinstance(evnt, midi.SetTempoEvent):
                evnt.set_bpm(int(round(evnt.get_bpm() / adj)))
    return ptrn


def time_split(x):
    """
    Breaks up a list or midi.Track containing midi.events into a nested list
    where each sublist represents events which occur simultaneously.
    """
    if isinstance(x, midi.Pattern) or any(
        [isinstance(trck, (list, midi.Track)) for trck in x]
    ):
        x = [
            trck
            for trck in x
            if utils.safe_len(trck) and isinstance(trck, (list, midi.Track))
        ]
        x = consolidate_trcks(x)
    if not len(x):
        return []
    ret = []
    sub = []
    for evnt in utils.evnt_gen(x):
        if evnt.tick != 0 and sub:
            ret.append(sub)
            sub = []
        sub.append(evnt)
    ret.append(sub)
    return ret


def aux_split_timestep(output, new_trck, cache, tempo, tmp_flag, typ_flag, include_end):
    """
    Auxillary function for split_on_timesignature_change. Once all simultaneous events 
    are checked and collected in the cache, we call this function to create the 
    splits/decide whether these events go in the currently generated track or in a new
    track.
    """
    new_flag = len(new_trck) != 0

    if typ_flag and new_flag:
        if include_end:
            new_trck.append(midi.EndOfTrackEvent(tick=new_trck[-1].tick + 1000))
        new_trck = midi.Track(tick_relative=False)
        output.append(new_trck)

    new_trck.extend(cache)
    cache = []

    if typ_flag and new_flag and not tmp_flag:
        tempo.tick = new_trck[-1].tick
        new_trck.insert(1, copy.copy(tempo))

    tmp_flag = False
    typ_flag = False

    return [output, new_trck, cache, tempo, tmp_flag, typ_flag]


def split_on_timesignature_change(x, typ, include_end=True):
    """
    Partitions a midi.Track into different midi.Tracks depending on the occurance of 
    typ (a particular midi.events type).
    For example, one may split a single track into smaller tracks that have different
    time signatures.
    """
    if isinstance(x, midi.Pattern):
        trck = x[0]
        mode = "ptrn"
    elif isinstance(x, utils.MidiObj):
        assert len(x.ptrn) == 1
        trck = x.ptrn[0]
        mode = "midi_obj"
    else:
        trck = x
        mode = "trck"
    trck.make_ticks_abs()
    new_trck = midi.Track(tick_relative=False)
    tempo = midi.SetTempoEvent()
    tmp_flag = False
    typ_flag = False
    output = [new_trck]
    cache = []
    tick = trck[0].tick
    args = [output, new_trck, cache, tempo, tmp_flag, typ_flag]
    for evnt in trck:
        if evnt.tick != tick:
            tick = evnt.tick
            args = aux_split_timestep(*args, include_end)
        args[2].append(copy.copy(evnt))
        # encountered type on which to split?
        if isinstance(evnt, typ):
            args[5] = typ_flag = True
        # does the tempo change and therefore need updating?
        if isinstance(evnt, midi.SetTempoEvent):
            args[4] = tmp_flag = True
            args[3] = tempo = evnt
    output = aux_split_timestep(*args, include_end)[0]
    ret = []
    for trck in output:
        trck.make_ticks_rel()
        trck[0].tick = 0
        if mode == "ptrn":
            ptrn = copy_ptrn(x)
            ptrn.append(trck)
            ret.append(ptrn)
        elif mode == "midi_obj":
            ptrn = copy_ptrn(x.ptrn)
            ptrn.append(trck)
            midi_obj = utils.MidiObj(ptrn)
            ret.append(midi_obj)
        else:
            ret = output
    return ret


def _dedupe_timestep(new_trck, cache, criteria, omits=None):
    """
    Auxillary function for MuGen.edit.dedupe.
    Processes evnts in a timestep.
    """
    n = len(cache)
    n_range = range(n)
    isdupe = [False] * n
    for i in n_range:
        if not isdupe[i]:
            if omits is None or not isinstance(cache[i], omits):
                evnt1 = cache[i]
                typ = type(evnt1)
                for j in n_range[i + 1 :]:
                    evnt2 = cache[j]
                    if isinstance(evnt2, typ) and criteria(evnt1, evnt2):
                        isdupe[j] = True
    for i in n_range:
        if not isdupe[i]:
            new_trck.append(cache[i])
    cache = []
    return new_trck, cache


def _default_dedupe_criteria(e0, e1):
    """
    An auxikkary function for MuGen.edit.dedupe.
    Returns True if e0 and e1 are midi.events with equal data and tick attributes. Unless
    both events are NoteOn or NoteOff events, in which case if at least one is for channel 9,
    we consider whether they have an equal channel attribute.
    """
    if isinstance(e0, (midi.NoteOnEvent, midi.NoteOffEvent)) and any(
        [e.channel == 9 for e in [e0, e1]]
    ):
        return e0.channel == e1.channel and e0.data == e1.data and e0.tick == e1.tick
    else:
        return e0.data == e1.data and e0.tick == e1.tick


def dedupe(x, **kwargs):
    """
    Removes duplicate midi.events from a midi.Track. Some event types may be omitted by
    passing them as elements in the 'omits' argument. Custom criteria may be used by passing
    a function, 'criteria', that returns True if two evnts match.
    """

    if isinstance(x, midi.Pattern):
        assert (
            len(x) == 1
        ), "Pattern must be uni-tracked, try using edit.consolidate_trcks beforehand"
        old_ptrn = x
        old_trck = x[0]
        old_ptrn.make_ticks_abs()
        mode = "ptrn"
    else:
        old_trck = x
        mode = "trck"

    criteria = kwargs.get("criteria", _default_dedupe_criteria)
    omits = kwargs.get("omits", None)
    verbose = kwargs.get("verbose", False)
    old_trck.make_ticks_abs()
    new_trck = midi.Track(tick_relative=False)
    cache = []
    tick = 0
    for evnt in old_trck:
        if evnt.tick != tick:
            new_trck, cache = _dedupe_timestep(new_trck, cache, criteria, omits)
            tick = evnt.tick
        cache.append(copy.deepcopy(evnt))
    new_trck = _dedupe_timestep(new_trck, cache, criteria, omits)[0]
    new_trck.make_ticks_rel()
    if verbose:
        print(f"Removed {len(old_trck) - len(new_trck)} duplicate events...")
    if mode == "ptrn":
        new_ptrn = midi.Pattern()
        new_ptrn.append(new_trck)
        old_ptrn.make_ticks_rel()
        ret = new_ptrn
    else:
        ret = new_trck
        old_trck.make_ticks_rel()
        new_trck.make_ticks_rel()
    old_trck.make_ticks_rel()
    return ret


def consolidate_trcks(x, include_end=False):
    """
    Consolidates all midi.events in a midi.Pattern into a single track
    """
    if isinstance(x, midi.Pattern):
        old_ptrn = x
    else:
        old_ptrn = midi.Pattern(format=1, resolution=480, tick_relative=True,)
        for trck in x:
            if isinstance(trck, list):
                trck = midi.Track(trck)
            old_ptrn.append(trck)
    old_ptrn.make_ticks_abs()
    new_ptrn = midi.Pattern(
        format=old_ptrn.format,
        resolution=old_ptrn.resolution,
        tick_relative=old_ptrn.tick_relative,
    )
    temp_trck = []
    for old_trck in old_ptrn:
        # old_trck.make_ticks_abs()
        for old_evnt in old_trck:
            new_evnt = copy.deepcopy(old_evnt)
            temp_trck.append(new_evnt)
    temp_trck = sorted(temp_trck, key=lambda x: x.tick)
    new_trck = midi.Track(tick_relative=old_ptrn.tick_relative)
    new_ptrn.append(new_trck)
    for new_evnt in temp_trck:
        new_trck.append(new_evnt)
    # We convert to rel. so that playback sounds correct
    old_ptrn.make_ticks_rel()
    new_ptrn.make_ticks_rel()
    if include_end:
        new_trck.append(midi.EndOfTrackEvent(tick=1000))
    return new_ptrn


def replace_evnt_typ(
    old_ptrn, old_typ, new_typ, condition=lambda x: True, copy_func=None
):
    """
    Replace a midi.events type with another midi.events type.
    tick, data and channel attributes are copied accross. 
    A function may be passed (custom_condition) to identify instances
    of the event to be copied
    """
    old_ptrn.make_ticks_abs()
    new_ptrn = copy_ptrn(old_ptrn)
    for old_trck in old_ptrn:
        new_trck = midi.Track(tick_relative=False)
        new_ptrn.append(new_trck)
        for old_evnt in old_trck:
            if isinstance(old_evnt, old_typ) and condition(old_evnt):
                if copy_func is not None:
                    new_evnt = copy_func(old_evnt, new_typ)
                else:
                    new_evnt = new_typ(
                        tick=old_evnt.tick,
                        data=old_evnt.data,
                        channel=old_evnt.channel,
                    )
            else:
                new_evnt = copy.deepcopy(old_evnt)

            new_trck.append(new_evnt)
    old_ptrn.make_ticks_rel()
    new_ptrn.make_ticks_rel()
    return new_ptrn


def apply_on_pulse(x, div, apply_func, gen_func):
    """
    Will insert an object e.g. midi.NoteOneEvent, 'on a pulse' within a midi sequence. An nd =
    1, 4, 8 will create a pulse of whole
    At the moment this function is quite naive and makes the assumption that the first beat
    occurs at the very start of a track or on the same tick that a TimeSignatureEvent occurs
    and does not consider the situation when there is an anacrusis or incorrectly written 
    time signature. 
    Args:
        iterable of midi.events e.g. list/midi.Pattern/midi.Track/utils.MidiObj
    Returns:
        list - containing utils.Beat objects representing where the ebats occur in a track
    """
    if isinstance(x, utils.MidiObj):
        x = x.ptrn
    x.make_ticks_abs()

    chgs = analyze.get_evnt_info(x, midi.TimeSignatureEvent, attrs=["tick", "data"])[1]
    chgs.append([x[-1][-1].tick, chgs[-1][-1]])
    tick = 0

    ret = []

    for i in range(len(chgs) - 1):
        tick, tsig = chgs[i][0], chgs[i][1][:2]
        tpp = gen_func(x.resolution, **dict(tsig=tsig, raw=1, div=div))
        on = True
        while tick <= chgs[i + 1][0]:
            ret = apply_func(
                **dict(alist=ret, tick=tick, on=on, res=x.resolution, tsig=tsig)
            )
            tick += tpp
            on = not on
    x.make_ticks_rel()
    return ret


def add_pulse(x, div, apply_func, gen_func):
    """
    Apply 'func' to x at the divisions (div)
    
    For example:
    
    apply_func may be
    
    def add_bass_drum(alist, beat, on):
        if on:
            alist.append(midi.NoteOnEvent(tick=beat, channel=9, data=[35, 75]))
        return alist
        
    gen_func is used to generate steps in ticks
    """
    if isinstance(x, utils.MidiObj):
        x = x.ptrn
    tmp_trck = apply_on_pulse(x, div, apply_func, gen_func)
    x.make_ticks_abs()
    new_ptrn = copy_ptrn(x)
    tmp_trck.extend([copy.copy(e) for e in x[0]])
    x.make_ticks_rel()
    tmp_trck = sorted(tmp_trck, key=lambda x: x.tick)
    new_trck = midi.Track(tick_relative=False)
    for evnt in tmp_trck:
        new_trck.append(evnt)
    new_ptrn.append(new_trck)
    x.make_ticks_rel()
    new_ptrn.make_ticks_rel()
    return new_ptrn
