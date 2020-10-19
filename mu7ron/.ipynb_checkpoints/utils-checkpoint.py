from zipfile import ZipFile
import os
import copy
import time
import datetime
import math

import numpy as np
import pygame
import midi

from mu7ron import analyze
from mu7ron import maps


def extract_zip_files(zip_map: dict) -> None:
    # extract all zipped files to repsective directories
    for dir_ in tqdm(zip_map):
        for fn in zip_map[dir_]:
            with ZipFile(os.path.join(dir_, fn), 'r') as file:
                try:
                    file.extractall(dir_)
                except NotImplementedError:
                    print(f'Warning: cannot open: {dir_}')
                    continue

def open_all_files_with_ext(dirs, ext: str) -> list:
    '''
    Retrieve all filenames ending ext
    '''
    fnames = []
    for dir_ in tqdm(dirs):
        for root, subs, files in os.walk(dir_):
            for fn in files:
                if fn.lower().endswith(ext):
                    fnames.append(os.path.join(root, fn))
    return fnames

def tstamp(name:str='mugen', fmt:str='%d_%b_%Y_%H-%M-%S'):
    '''
    Concatenates the current date & time to a string.
    Format: '%d_%b_%Y_%H-%M-%S'
    '''
    return f"{name}_{datetime.datetime.now().strftime(fmt)}"

def safe_len(x):
    '''
    safely returns the length of an object without throwing an exception
    if the object is a number
    '''
    try:
        ret = len(x)
    except TypeError:
        ret = False
    return ret

def flatten(alist, depth=0):
    '''
    A generator that flattens nested containers (list, tuple, set, np.ndarray) of any nested degree
    '''
    if depth is 1:
        for sublist in alist:
            for item in sublist:
                yield item
    else:
        for item in alist:
            if isinstance(item, (list, tuple, set, np.ndarray)) and not isinstance(item, (str, bytes)):
                yield from flatten(item)
            else:
                yield item
    
def play(x, t=None):
    '''
    A quick way to play a midi.Track or midi.Pattern. t=number
    of seconds to play the sequence.
    '''
    if isinstance(x, str):
        sname = x
    else:
        if isinstance(x, midi.Track):
            ptrn = midi.Pattern(format=1, resolution=480, tick_relative=True)
            ptrn.append(x)
        elif isinstance(x, midi.Pattern):
            ptrn = x
        else:
            raise TypeError
            
        working_dir = ''
        for s in ['data', 'midi', 'temp', 'working']:
            working_dir = os.path.join(working_dir, s)
            if not os.path.isdir(working_dir):
                os.mkdir(working_dir)
        valid = False
        while not valid:
            i = 0
            sname = os.path.join(working_dir, f'temp{i}.mid')
            try:
                if os.path.exists(sname):
                    os.remove(sname)
            except PermissionError:
                i += 1
            else:
                break
        midi.write_midifile(sname, ptrn)
        
    pygame.init()
    pygame.mixer.music.load(sname)
    if t is not None:
        t_end = time.time() + t
        pygame.mixer.music.play()
        while time.time() < t_end:
            pass
        pygame.mixer.music.stop()
    else:
        pygame.mixer.music.play()

def trck_gen(x):
    '''
    loops through each track in x
    '''
    if isinstance(x, midi.Pattern):
        for trck in x:
            yield trck
    elif isinstance(x, MidiObj):
        for trck in x.ptrn:
            yield trck
    elif isinstance(x[0], MidiObj):
        for obj in x:
            for trck in obj.ptrn:
                yield trck
    else:
        for ptrn in x:
            for trck in ptrn:
                yield trck

def evnt_gen(x):
    '''
    loops through each event in x
    ''' 
    if isinstance(x, midi.Track):
        for evnt in x:
            yield evnt
    else:
        for trck in trck_gen(x):
            for evnt in trck:
                yield evnt

def counter(alist, func=None):
    '''
    - counts the number of things in a list
    - can apply a function (func) to item
    '''
    adict = {}
    for item in alist:
        if func is not None:
            item = func(item)
        if item is not None:
            adict[item] = adict.get(item, 0) + 1
    return adict

def nptf(x):
    '''
    Negative Power To Fraction
    For converting the second value in midi.TimesignatureEvent data from 
    a negative power to a fraction
    '''
    return round(1 // 2 ** -x)

def ftnp(x):
    '''
    Fraction to Negative Power
    Converts the denominator in a fraction representation of a time signature
    into a value used by midi.TimesignatureEvent data[1] i.e. negative power
    '''
    return round(math.log2(x))

def bpm_to_mspt(bpm, res=480):
    '''
    Coverts an integer value of beats per minute to miliseconds per quarter note
    '''
    return 60000 / res / bpm

def mspt_to_bpm(mspt, res=480):
    '''
    Coverts miliseconds per quarter note to an integer value of beats per minute
    '''
    return 60000 / res / mspt

def ticks_per_bar_division(res, **kwargs):
    '''
    Resolution and Time Signature to Ticks Per Measure
    Converts a resolution and time signature to ticks per beat
    if raw=1 the the denominator of the tsig is in the python-midi format 
    i.e. a negative power of 2: 2 = quarter, 3 = eight notes etc...
    The bar division (bdiv) is tied to frequency of the pulse. For example,
    values of 1., 2., and 4., will return tpb required to feel pulses every
    whole, half and quarter of a bar respectively. Please note this is 
    independent of note divisions.
    '''
    tsig = kwargs.get('tsig', [4, 4])
    raw  = kwargs.get('raw', 0)
    div  = kwargs.get('div', 1.)
    tsig = tsig[0] / (utils.nptf(tsig[1]) if raw else tsig[1])
    return round(res * tsig * 4. / div)

def ticks_per_note_division(res, **kwargs) -> int:
    '''
    Args:
        res  - int, midi.Pattern.resolution, ticks per quarter note
        div  - int, a note division e.g. 1, 2, 4 or 8 for whole, half
               quarter or eight notes respectively. Unlike real music
               it is possible to use a number that is not 1 or 2 and
               not divisible by 4, e.g. 3 or 6.
    Returns:
        int, ticks per note division
    '''
    div = kwargs.get('div', 1.)
    return round(res * 4. / div)

def translate_tsig(tsig_data):
    '''
    Translates the data from a midi.TimeSignatureEvent into a string 
    representation.
    '''
    tsig_data = list(tsig_data)
    tsig_data[1] = nptf(tsig_data[1])
    tsig_data = '[' + ', '.join(map(str, tsig_data)) + ']'
    return tsig_data

def quantize(x: int, q: int) -> int:
    '''
    Will quantize a continuous or discrete range into steps of q
    where anything between 0 and q will be clipped to q.
    '''
    return q * (x // q) if not 0 < x <= q else q

def dynamic_order(x: int, q: int) -> int:
    '''
    Returns the number of steps that will exist if a range(0, x)
    is quantized into steps of q.
    '''
    return math.ceil(x / q)

def dynamic_range(x: int, q: int) -> list:
    '''
    Returns a list of values if range(0, x) is quantized into steps of q.
    '''
    return sorted(set(quantize(n, q) for n in range(x)))


class MidiObj:
    '''
    A simple wrapper for a midi.Pattern for simple analysis, manipulations and playback etc...
    '''
    def __init__(self, path=None, ptrn=None, sn=None):
        '''
        Args:
            ptrn - midi.Pattern()
            path - str, relative path to midi file
        '''

        if not 'midi_map' in globals():
            global midi_map
            midi_map = maps.load_midi_map()
        
        has_path = path is not None
        has_ptrn = ptrn is not None
        
        assert any([has_path, has_ptrn]), 'MidiObj requires midi.Pattern or path to a saved midi.Pattern as input to __init__'
        
        working_dir = os.path.split(__file__)[0]
        for s in ['data', 'midi', 'temp', 'working']:
            working_dir = os.path.join(working_dir, s)
            if not os.path.isdir(working_dir):
                os.mkdir(working_dir)
        
        if has_ptrn and not has_path:
            assert sn is not None, 'A midi.ptrn without a path requires a serial number (sn) argument to act as an identifier and generate a unique path'
            revert_dir = os.path.split(__file__)[0]
            for s in ['data', 'midi', 'temp', 'working']:
                revert_dir = os.path.join(revert_dir, s)
                if not os.path.isdir(revert_dir):
                    os.mkdir(revert_dir)
            
            if not sn.endswith('.mid'):
                sn += '.mid'
            self.fn       = sn
            self.org_path = os.path.join(revert_dir, sn)
            self.path     = os.path.abspath(os.path.join(working_dir, self.fn))
            self.dir      = os.path.split(self.path)[0]

        elif has_path:
            path = os.path.normpath(path)

            self.fn       = os.path.split(path)[-1]
            self.org_path = path
            self.path     = os.path.abspath(os.path.join(working_dir, self.fn))
            self.dir      = os.path.split(self.path)[0]

        if ptrn is not None:
            self.ptrn = ptrn
        else:
            self.load(path) 
        
        self._init()
        
    def _init(self):
        
        self.ptrn.make_ticks_abs()
        self.n_tsig, self.tsig_data = analyze.get_evnt_info(self.ptrn, midi.TimeSignatureEvent)
        self.n_tmpo, self.tmpo_data = analyze.get_evnt_info(self.ptrn, midi.SetTempoEvent, ['tick', 'get_bpm'], lambda x: [x[0], x[1]()])
        self.inst_data              = analyze.get_evnt_info(self.ptrn, midi.ProgramChangeEvent, ['data'])[1]
        self.ptrn.make_ticks_rel()
        
        self.uniq_pgm_trcks = list(analyze.uniq_pgm_chk(self.ptrn))
        self.inst_data      = list(flatten(self.inst_data))
        self.n_uinst        = len(set(self.inst_data))
        self.n_vce          = analyze.max_simultaneous_notes(self.ptrn)
        
        self.save()

    def __repr__(self):
        '''
        Returns a clearly formatted string for displaying the 
        midi objects attributes
        '''
        repr_str = '<%s.%s object at %s>' % (
                    self.__class__.__module__,
                    self.__class__.__name__,
                    hex(id(self))
                )
        sep        = '\n' + 24 * ' '
        
        inst_data = ['i  | Group     | Instrument', '-' * 25]
        
        inst_data.extend([f"{str(x + 1)}{' ' * (3 - len(str(x + 1)))}| " + \
                          f"{midi_map['grp'][str(x + 1)]}{' ' * (10 - len(midi_map['grp'][str(x + 1)]))}| {midi_map['inst'][str(x + 1)]}"
                          for x in self.inst_data])
        inst_data = sep.join(inst_data)
        
        tsig_data = ['tick      | data', '-' * 25]
        tsig_data.extend([f"{x[0]}{' ' * (10 - len(str(x[0])))}| {translate_tsig(x[1])}" for x in self.tsig_data])
        tsig_data = sep.join(tsig_data)

        tmpo_data = ['tick      | bpm', '-' * 25]
        tmpo_data.extend([f"{x[0]}{' ' * (10 - len(str(x[0])))}| {round(x[1], 3)}" for x in self.tmpo_data])
        tmpo_data = sep.join(tmpo_data)
        
        attr_str = f'''              
        __________________________________________________________________________
        
        File          : {self.fn}
        Location      : {self.dir}
        Repr          : {repr_str}
        Resolution    : {self.ptrn.resolution}
        ---------------------------------------------------------------------------
        
        Voices        :
            n. voice  : {self.n_vce}
            n. u_inst : {self.n_uinst}
            data.     : {inst_data}
            u. trck   : {self.uniq_pgm_trcks}

        Time sig.     :
            n.        : {self.n_tsig}
            data      : {tsig_data}

        Tempo         :
            n.        : {self.n_tmpo}
            data      : {tmpo_data}
        __________________________________________________________________________
        
        '''
        return attr_str
    
    def play(self, t=None):
        '''
        Play audio of self.ptrn
        '''
        play(self.ptrn, t)
        
    @staticmethod
    def stop():
        '''
        Stop audio of self.ptrn
        '''
        pygame.mixer.music.stop()
        
    def save(self, path=None):
        '''
        Store self at self.path
        '''
        if path is not None:
            self.path = path
        midi.write_midifile(self.path, self.ptrn)
        
    def load(self, path=None):
        '''
        Load from backed up MidiObj
        '''
        self.ptrn = midi.read_midifile(path if path is not None else self.path)
    
    def revert(self):
        '''
        Load up the original midi file and start fresh
        '''
        self.load(self.org_path)
        self.save()
            
    def chge_inst(self, idx, new_inst):
        '''
        Change the instruments of some voices (ProgramChangeEvents)
        
        Args:
            idx      - list, numbers of voices/insts to change, please note that they
                       are numbered in the order in which they appear when looping
                       ptrn > trck > evnt
            new_inst - int, the number of the new midi instrument
        '''
        i = 0
        for trck in self.ptrn:
            for evnt in trck:
                if isinstance(evnt, midi.ProgramChangeEvent):
                    if i in idx:
                        evnt.data[0] = new_inst
                        i += 1
        self.save()