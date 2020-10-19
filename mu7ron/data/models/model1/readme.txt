off_mode     = False   # whether to encode NoteOffEvents or just NoteOnEvents with 0 velocity
q            = 8       # quantization factor of velocity
q_map        = maps.create_q_map(128, q, encode=True, decode=True)
t            = 8     # the smallest timestep in milliseconds
n_vel        = utils.dynamic_order(128, q) # No. of different available levels of velocity
n_time       = 144   # Available timesteps
n_pitch      = 128 * 2 if off_mode else 128 # + Available pitches
n_pulse      = 0     # Number of added pulses
n_vocab      = n_time + n_pitch + n_vel + n_pulse # Available choices
n_sample     = 24    # batch size
n_input      = 120   # time steps - the temporal dimension
n_output     = 1     # number of timesteps to predict into the future
n_teach      = 12
n_step       = n_sample
n_example    = 1
random_state = 117

model = Sequential()
model.add(LSTM(720, 
               input_shape=(n_input, n_vocab),
               return_sequences=True,
               ))

model.add(LSTM(720, 
               ))

model.add(Dense(n_vocab))
model.add(Activation('softmax'))

model.compile(
              loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['categorical_accuracy'],
)
model.summary()