import numpy as np

from mu7ron import utils

def timeslips_encoder(x:float, t:int, n_time:int) -> list:
    '''
    The inverse function of from_timeslips. Encodes milliseconds 
    as a list of ints; chunks of n_time - 1 plus a remainder: 
    r < n_time - 1.
    '''
    n_time -= 1
    w, r    = divmod(x / t, n_time)
    ret     = [n_time for _ in range(int(w))]
    ret.append(int(round(r)))
    return ret

def timeslips_decoder(x:list, t:int) -> float:
    '''
    The inverse function of to_timeslips. Decodes a list of ints (timeslips)
    into a single integer representing milliseconds of time.
    '''
    return sum(x) * t

def base_digits_encoder(n:int, b:int) -> list:
    '''
    The inverse function of 'from_base'. This 
    function converts a base 10 integer to a list of 
    integers that represent the original number in
    another base 'b'. 
    '''
    if n == 0:
        return [0]
    ret = []
    while n:
        ret.append(int(n % b))
        n //= b
    return ret[::-1]

def base_digits_decoder(alist:list, b:int) -> int:
    '''
    The inverse function of 'to_base'. This
    function will take a list of integers, where each 
    element is a digit in a number encoded in base 'b',
    and return an integer in base 10
    '''
    p   = 0
    ret = 0
    for n in alist[::-1]:
        ret += n * b ** p
        p   += 1
    return ret

def ticks_per_bar_division(res, **kwargs):
    '''
    Resolution and Time Signature to Ticks Per Measure
    Converts a resolution and time signature to ticks per beat
    if raw=1 the the denominator of the tsig is in the python-midi format 
    i.e. a negative power of 2: 2 = quarter, 3 = eight notes etc...
    The bar division (div) is tied to frequency of the pulse. For example,
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