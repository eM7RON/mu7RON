max simultaneous voices = 1

off_mode     = False   # whether to encode NoteOffEvents or just NoteOnEvents with 0 velocity
q            = 8       # quantization factor of velocity
q_map        = maps.create_q_map(128, q, encode=True, decode=True)
t            = 24     # the smallest timestep in milliseconds
n_vel        = utils.dynamic_order(128, q) # No. of different available levels of velocity
n_time       = 48   # Available timesteps
n_pitch      = 128 * 2 if off_mode else 128 # + Available pitches
n_pulse      = 0     # Number of added pulses
n_vocab      = n_time + n_pitch + n_vel + n_pulse # Available choices
n_sample     = 48    # batch size
n_input      = 120   # time steps - the temporal dimension
n_output     = 1     # number of timesteps to predict into the future
n_teach      = 12
n_step       = n_sample
n_example    = 1
buffer       = 150
random_state = 117

time_encoder = temporal.base_digits_encoder #time.timeslips_encoder
time_decoder = temporal.base_digits_decoder #time.timeslips_decoder
ekwa         = dict(b=n_time) #dict(t=t, n_time=n_time)
dkwa         = dict(b=n_time)

model = Sequential()
model.add(LSTM(480, 
               input_shape=(n_input, n_vocab),
               return_sequences=True,
               ))

model.add(LSTM(480,
               ))

