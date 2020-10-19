import numpy as np
import midi

from mu7ron import maps
from mu7ron import utils


def display_evnt_typs(x):
    '''
    A quick count of each unique event type in a midi.Track or midi.Pattern
    '''
    typs = [type(evnt) for evnt in utils.evnt_gen(x)]
    return counter(typs)

def cnt_evnt_typ(ptrn, typ):
    '''
    Args:
        ptrn   - midi.Pattern
        typ    - midi.events Type
    Returns:
        n_evnt - int, number of typ evnts in ptrn
    '''
    n_evnt = 0
    for trck in ptrn:
        for evnt in trck:
            if isinstance(evnt, typ):
                n_evnt += 1
    return n_evnt

def has_tsig(x, tsigs=[(4, 4)]) -> bool:
    '''
    Args:
        x       - list/midi.Pattern/midi.Track/MidiObj
        tsigs   - list[tuple(int, int),... tuple(int, int)],
                  time signatures encoded as a 2-tuple(numerator, denominator)
    Returns:
        boolean - True if ptrn contains any time signatures contained in tsigs
    '''
    for evnt in utils.evnt_gen(x):
        if isinstance(evnt, midi.TimeSignatureEvent):
            data = list(evnt.data[: 2])
            data[1] = utils.nptf(data[1])
            data = tuple(data)
            if any([ts == data for ts in tsigs]):
                return True
            
def has_insts(x, insts: tuple) -> bool:
    '''
    Args:
        x       - list/midi.Pattern/midi.Track/MidiObj
        insts   - int, midi standard numbers for instruments
    Returns:
        boolean - True if ptrn contains any types in typ
    '''
    insts = [i - 1 for i in insts]
    for evnt in utils.evnt_gen(x):
        if isinstance(evnt, midi.ProgramChangeEvent) and evnt.data[0] in insts:
            return True
        
def which_insts(x, insts: tuple, uniq: bool=True) -> list:
    '''
    Args:
        x       - list/midi.Pattern/midi.Track/MidiObj
        insts   - int, midi standard numbers for instruments
    Returns:
        boolean - True if ptrn contains any types in typ
    '''
    ret = []
    insts = [i - 1 for i in insts]
    for evnt in utils.evnt_gen(x):
        if isinstance(evnt, midi.ProgramChangeEvent):
            for i in insts:
                if i == evnt.data[0]:
                    ret.append(i + 1)
    if uniq:
        ret = list(set(ret))
    return ret

def has_only_inst(x, to_keep:set) -> bool:
    '''
    Args:
        x         - list/midi.Pattern/midi.Track/MidiObj
        to_keep   - set{int} midi standard numbers for desired instruments
                    all shifted by -1
    Returns:
        bool      - True if isinstance(evnt, typ) and not condition(evnt)
    '''
    all_insts = set(range(128))
    to_remove = all_insts - to_keep
    for evnt in utils.evnt_gen(x):
        if isinstance(evnt, midi.ProgramChangeEvent) and evnt.data[0] in to_remove:
            return False
    return True
    
def has_evnt_typ(x, typ=()) -> bool:
    '''
    Args:
        x       - list/midi.Pattern/midi.Track/MidiObj
        typ     - tuple(type(midi.event),..., type(midi.event))
    Returns:
        boolean - True if ptrn contains any types in typ
    '''
    for evnt in utils.evnt_gen(x):
        if isinstance(evnt, typ):
            return True
    return False
        
def has_channels(x, typs=(midi.NoteOnEvent, midi.NoteOffEvent), channels=[9]):
    '''
    Returns True if any midi.events types in the tuple 'typs' have channels contained
    in the tuple 'channels'
    '''
    for evnt in utils.evnt_gen(x):
        if isinstance(evnt, typs) and evnt.channel in channels:
            return True

def is_evnt_typ_uniq(x, typ):
    '''
    Returns True if x contains a single instance of typ
    '''
    uniq = False
    for evnt in utils.evnt_gen(x):
        if isinstance(evnt, typ):
            if uniq:
                return not uniq
            uniq = True
    return uniq

def is_equal_midi_sequences(a, b):
    '''
    Returns True if a and b are sequences containing 
    equivalent midi.events objects else False.
    The type, .data and .tick of the evnts are compared.
    '''
    if type(a) != type(b):
        return False
    a = list(utils.evnt_gen(a))
    b = list(utils.evnt_gen(b))
    lena, lenb = len(a), len(b)
    if lena != lenb:
        return False
    else:
        for i in range(lena):
            evnt0, evnt1 = a[i], b[i]
            if type(evnt0) != type(evnt1) \
            or evnt0.data != evnt1.data \
            or evnt0.tick != evnt1.tick:
                return False
    return True
        
def uniq_pgm_chk(ptrn):
    '''
    Which tracks have a unique, solitary ProgramChangeEvent
    '''
    return map(lambda x: is_evnt_typ_uniq(x, midi.ProgramChangeEvent), ptrn)

def max_simultaneous_notes(x):
    '''
    Returns the maximum number of simultaneous notes in a ptrn
    '''
    state = {}
    n_max  = 0
    for evnt in utils.evnt_gen(x):
        on = off = False
        if  isinstance(evnt, midi.NoteOnEvent):
            on  = True
        elif isinstance(evnt, midi.NoteOffEvent):
            off = True
        if on or off:
            note, vel = evnt.data[:2]
            if on and not vel or off:
                state[note] = max(state.get(note, 0) - 1, 0)
            elif on:
                state[note] = state.get(note, 0) + 1
            n     = sum(list(state.values()))
            n_max = max(n, n_max)
    return n_max

def get_evnt_info(x, evnt_typ, attrs=['tick', 'data'], func=None):
    '''
    Returns a 2-tuple containing the count (as an int) and a list containing the value
    of each entry for an attribute of a particular event type.
    '''
    count, data = 0, []
    for evnt in utils.evnt_gen(x):
        if isinstance(evnt, evnt_typ):
            datum = [getattr(evnt, attr) for attr in attrs]
            if func is not None:
                datum = func(datum)
            data.append(datum)
            count += 1
    return count, data

def get_min_max(x, typs=(), func=lambda x: x.tick):
    '''
    Get the minimum and maximum values for an event type.
    Args:
        x    - midi.Pattern/midi.Track or python sequence containing
               midi.events
        typs - tuple, contains types of events to analyze
        func - callable function to expose event attribute, default is
               lambda x: x.tick
    Returns:
          min, max - int, int
    '''
    min_ = np.inf 
    max_ = -np.inf
    for evnt in utils.evnt_gen(x):
        if isinstance(evnt, typs):
            min_ = min(min_, func(evnt))
            max_ = max(max_, func(evnt))
    return min_, max_