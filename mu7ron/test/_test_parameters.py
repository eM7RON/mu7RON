import midi

from mu7ron import maps
from mu7ron import temporal
from mu7ron import utils


def p0():
    typs_2_keep  = (midi.NoteOnEvent, midi.NoteOffEvent, midi.SetTempoEvent)
    off_mode     = True #True  # whether to encode NoteOffEvents or just NoteOnEvents with 0 velocity
    q            = 8     # quantization factor of velocity
    q_map        = maps.create_q_map(128, q, encode=True, decode=True)
    t            = 8     # the smallest timestep in milliseconds
    n_vel        = utils.dynamic_order(128, q) # No. of different available levels of velocity
    n_time       = 48   # Available timesteps
    n_pitch      = 128 * 2 if off_mode else 128 # Available pitches
    n_pulse      = 0     # Number of added pulses
    n_vocab      = n_time + n_pitch + n_vel + n_pulse # Available choices
    time_encoder = temporal.base_digits_encoder #temporal.timeslips_encoder
    time_decoder = temporal.base_digits_decoder #temporal.timeslips_decoder
    ekwa         = dict(b=n_time) #dict(t=t, n_time=n_time)
    dkwa         = dict(b=n_time)
    return locals()
    
def p1():
    typs_2_keep  = (midi.NoteOnEvent, midi.NoteOffEvent, midi.SetTempoEvent)
    off_mode     = False #True  # whether to encode NoteOffEvents or just NoteOnEvents with 0 velocity
    q            = 18     # quantization factor of velocity
    q_map        = maps.create_q_map(128, q, encode=True, decode=True)
    t            = 8     # the smallest timestep in milliseconds
    n_vel        = utils.dynamic_order(128, q) # No. of different available levels of velocity
    n_time       = 24   # Available timesteps
    n_pitch      = 128 * 2 if off_mode else 128 # Available pitches
    n_pulse      = 0     # Number of added pulses
    n_vocab      = n_time + n_pitch + n_vel + n_pulse # Available choices
    time_encoder = temporal.base_digits_encoder #temporal.timeslips_encoder
    time_decoder = temporal.base_digits_decoder #temporal.timeslips_decoder
    ekwa         = dict(b=n_time) #dict(t=t, n_time=n_time)
    dkwa         = dict(b=n_time)
    return locals()
    
def p2():
    typs_2_keep  = (midi.NoteOnEvent, midi.NoteOffEvent, midi.SetTempoEvent)
    off_mode     = True #True  # whether to encode NoteOffEvents or just NoteOnEvents with 0 velocity
    q            = 8     # quantization factor of velocity
    q_map        = maps.create_q_map(128, q, encode=True, decode=True)
    t            = 8     # the smallest timestep in milliseconds
    n_vel        = utils.dynamic_order(128, q) # No. of different available levels of velocity
    n_time       = 144   # Available timesteps
    n_pitch      = 128 * 2 if off_mode else 128 # Available pitches
    n_pulse      = 0     # Number of added pulses
    n_vocab      = n_time + n_pitch + n_vel + n_pulse # Available choices
    time_encoder = temporal.timeslips_encoder
    time_decoder = temporal.timeslips_decoder
    ekwa         = dict(t=t, n_time=n_time)
    dkwa         = dict(t=t)
    return locals()
    
def p3():
    typs_2_keep  = (midi.NoteOnEvent, midi.NoteOffEvent, midi.SetTempoEvent)
    off_mode     = False #True  # whether to encode NoteOffEvents or just NoteOnEvents with 0 velocity
    q            = 18     # quantization factor of velocity
    q_map        = maps.create_q_map(128, q, encode=True, decode=True)
    t            = 10     # the smallest timestep in milliseconds
    n_vel        = utils.dynamic_order(128, q) # No. of different available levels of velocity
    n_time       = 192   # Available timesteps
    n_pitch      = 128 * 2 if off_mode else 128 # Available pitches
    n_pulse      = 0     # Number of added pulses
    n_vocab      = n_time + n_pitch + n_vel + n_pulse # Available choices
    time_encoder = temporal.timeslips_encoder
    time_decoder = temporal.timeslips_decoder
    ekwa         = dict(t=t, n_time=n_time)
    dkwa         = dict(t=t)
    return locals()

