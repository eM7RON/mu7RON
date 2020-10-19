import copy
import operator
import os
import pickle
import random
import sys

from tqdm import tqdm
import numpy as np
import midi

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, LayerNormalization
from tensorflow.compat.v1 import ConfigProto, Session
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import CategoricalAccuracy, CategoricalCrossentropy
from tensorflow.keras import backend as K

from MuGen import analyze
from MuGen import aug
from MuGen import edit
from MuGen import generate
from MuGen import learn 
from MuGen import maps
from MuGen import coders
from MuGen import temporal
from MuGen import utils
from MuGen import visualize

print('imports successful...')

WORKING_DIR = os.path.join('gdrive', 'My Drive', 'backups') 

config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)
set_session(session)
n_arg = len(sys.argv) - 1

if n_arg > 1:
    print('Invalid number of arguments', end='')
    sys.exit()
        
elif n_arg:
    with open(os.path.join(WORKING_DIR, 'params_train_valid.pkl'), 'rb') as file:
        params, train, valid = pickle.load(file)
        for name, value in params.items():
            try:
                exec(f'{name} = {value}')
            except:
                print('Parameter loading error... name={name}, value={value}.')
                pass
    
    model = load_model(os.path.join(WORKING_DIR, sys.argv[1]))
else:

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
    buffer       = 150
    random_state = 117

    time_encoder = temporal.timeslips_encoder #temporal.base_digits_encoder #temporal.timeslips_encoder
    time_decoder = temporal.timeslips_decoder # temporal.base_digits_decoder #temporal.timeslips_decoder
    ekwa         = dict(t=t, n_time=n_time) # dict(b=n_time) #dict(t=t, n_time=n_time)
    dkwa         = dict(t=t) # dict(b=n_time) #dict(t=t)
    
    params = dict(
              off_mode     = off_mode,
              q            = q,
              q_map        = q_map,
              t            = t,
              n_vel        = n_vel,
              n_time       = n_time,
              n_pitch      = n_pitch,
              n_pulse      = n_pulse,
              n_vocab      = n_vocab,
              n_sample     = n_sample,
              n_input      = n_input,
              n_output     = n_output,
              n_teach      = n_teach,
              n_step       = n_step,
              n_example    = n_example,
              buffer       = buffer,
              time_encoder = time_encoder,
              time_decoder = time_decoder,
              ekwa         = dict(t=t, n_time=n_time),
              dkwa         = dict(t=t) 
              )

    data_dir = os.path.join('data', 'midi', 'medren')
#     sub_dirs = [
#                 "Dave's", 
#     #             'Lakh v0.1', 
#     #             'bachcentral', 
#     #             'mfiles',
#     #             'classicalarchives_com',
#     #             'gasilvis_net',
#                ]
    
    sub_dirs = [
            'anonymous', 
            'assorted',
            'carols', 
            'Conrad Paumann', 
            'davidbellugi',
            'english',
            'irish',
            'italian',
            'praetorius',
            'scottish',
            'shakespearean',
            'maucamedus',
            'medieval_org',
            'davidbellugi',
            'curtisclark_org',
            'midiworld_com',
            'standingstones_com',
            'figshare_com',
           ]

    dirs = [os.path.join(data_dir, dir_) for dir_ in sub_dirs]

    print("retrieving all filenames ending '.mid'|'.midi'...")
    fnames = []
    for dir_ in tqdm(dirs):
        for root, subs, files in os.walk(dir_):
            for fn in files:
                if fn.lower().endswith(('.mid', '.midi')):
                    fnames.append(os.path.join(root, fn))

    print("reading midi files...")
    data   = []
    for fn in tqdm(fnames):
        try:
            data.append(midi.read_midifile(fn))
        except (TypeError, AssertionError):
            pass
    
    ###################
    #### Data Prep ####
    ###################

    insts_to_keep = {
        *range(46),
        *range(48, 112),
    #     *range(56, 105),
    #     *range(113, 128),
    }

    data = [x for x in data if analyze.has_only_inst(x, insts_to_keep)]

    #print("filtering by instruments...")
    #data = [x for x in tqdm(data) if analyze.has_only_inst(x, insts_to_keep)]
    print(f'{len(data)} samples remaining...')

    typs_2_keep=(
                 midi.NoteOffEvent,
                 midi.NoteOnEvent,
                 midi.SetTempoEvent,
                 midi.TimeSignatureEvent,
    )

    print('cleaning and preparing data...')

    def copy_func(old_evnt, new_typ):
        new_evnt = new_typ(
                           tick=old_evnt.tick,
                           data=[old_evnt.data[0], 0],
                           channel=old_evnt.channel,
                           )
        return new_evnt

    for i in tqdm(range(len(data))):

        # remove redundant events types
        data[i] = edit.filter_ptrn_of_evnt_typs(data[i], typs_2_keep)

        # filter out NoteOnEvents on channel 9 (10); which is used exclusively for percussion
        data[i] = edit.filter_ptrn_of_percussion(data[i])

        # combine all Tracks in the Pattern into a single Track
        data[i] = edit.consolidate_trcks(data[i])

        # adjust resolution to 480 but also adjust all tempo evnts as to preserve the sounding
        # speed of the music
        data[i] = edit.normalize_resolution(data[i], res=480)

        # we will quantize velocity into a fewer possible values
        data[i] = edit.quantize_typ_attr(data[i], q, (midi.NoteOnEvent, midi.NoteOffEvent), lambda x: x.data[1])

        # consolidating trcks leads to a lot of redundent/duplicate evnts which we can remove
        data[i] = edit.dedupe(data[i])

        #convert all NoteOff to NoteOn with 0 velocity
        if not off_mode:
            data[i] = edit.replace_evnt_typ(data[i], midi.NoteOffEvent, midi.NoteOnEvent, copy_func=copy_func)

        # split each ptrn where a timesignature event occurs
        data[i] = edit.split_on_timesignature_change(data[i], midi.TimeSignatureEvent)

    data = list(utils.flatten(data, depth=1))
    print(f'{len(data)} samples remaining...')

    typs_2_keep=(
                 midi.NoteOffEvent,
                 midi.NoteOnEvent,
                 midi.SetTempoEvent,
    )

    for i in tqdm(range(len(data))):
        # remove redundant events types
        data[i] = edit.filter_ptrn_of_evnt_typs(data[i], typs_2_keep)

    data = edit.filter_data_of_empty_ptrn(data)

    print(f'{len(data)} samples remaining...')

    print("filtering by max simultaneous voices...")
    data = [x for x in tqdm(data) if analyze.max_simultaneous_notes(x) < 3]
    print(f'{len(data)} samples remaining...')

    print(f'{len(data)} samples remaining...')

    data = [ptrn[0] for ptrn in data]

    print('encoding data...')

    input_ = []

    for trck in tqdm(data):
        input_.append(coders.categorize_input(trck, 
                                     q=q,
                                     q_map=q_map,
                                     time_encoder=time_encoder,
                                     ekwa=ekwa,
                                     n_time=n_time,
                                     n_vel=n_vel,
                                     sort_velocity=True,
                                     sort_pitch=True,
                                     sort_pulse=False,
                                     asarray=True, 
                                     dtype='int',
                                     off_mode=off_mode,
                                    ))
    
    for sequence in input_:
        assert max(sequence) < n_vocab, 'Encoding error: encoded sequences contain vocabulary that is out of bounds.'
    
    print(r'creating train/test splits...')

    np.random.seed(random_state)
    idx = np.random.permutation(range(len(input_)))
    input_ = np.array(input_)[idx]

    test_frac = 0.1
    n         = len(input_)
    idx       = int(round(n * (1. - test_frac)))
    train, valid = input_[: idx],  input_[idx: ]
    print(f'{len(train)} train samples...')
    print(f'{len(valid)} valid samples...')
    print('saving splits...')
    with open(os.path.join(WORKING_DIR, 'params_train_valid.pkl'), 'wb') as file:
        pickle.dump([params, train, valid], file)
    
    model = Sequential()
    model.add(LSTM(720, 
                   input_shape=(n_input, n_vocab),
                   #return_sequences=True,
                   ))

    # model.add(Dropout(0.24))
    # model.add(LSTM(720, 
    #                #return_sequences=True,
    #                ))

    # model.add(Dropout(0.24))
    # model.add(LSTM(480,
    #                ))

    # model.add(Dropout(0.24))

    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    
print('n_vocab: ', n_vocab)



train_gen = aug.MappedSamplingGen(train,
                                 shuffle=True,
                                 n_step=n_step,
                                 n_time=n_time,
                                 n_vocab=n_vocab,
                                 n_sample=n_sample, 
                                 n_input=n_input, 
                                 n_output=n_output,
                                 off_mode=off_mode,
                                 )

valid_gen = aug.MappedBalancedDataGen(valid, n_input=n_input, n_output=n_output, n_example=n_example, n_sample=n_sample, n_vocab=n_vocab)


print('spinning up tensorflow...')

set_session(session)

opt = optimizers.Nadam(
                      learning_rate=0.0001,
                      beta_1=0.9,
                      beta_2=0.999,
                      epsilon=1e-07,
                      name="Nadam",
                      clipnorm=True,
                      )

class CustomCheckpoint(Callback):

    def __init__(self, **kwargs):
        super().__init__()
        
        self.save_dir  = kwargs.get('save_dir', os.path.join(os.getcwd(), 'models'))
        self.monitor   = kwargs.get('monitor', 'loss')
        self.verbose   = kwargs.get('verbose', True)
        self.save_mode = kwargs.get('save_mode', 'save') # save_weights
        
        if self.monitor.endswith('loss'):
            self.best  = np.inf
            self.oper  = operator.lt
        else:
            self.best  = -np.inf
            self.oper  = operator.gt
    
    def on_epoch_end(self, epoch, logs=None):
        imp = False
        if self.oper(logs[self.monitor], self.best):
            imp = True
            val = str(logs[self.monitor]).replace('.', '_')
            save_path = os.path.join(self.save_dir, f"{utils.tstamp(f'model_{val}')}.h5")

        if self.verbose:
            if imp:
                msg = f"{self.monitor} improved from {self.best} to {logs[self.monitor]}, saving model to {save_path}" 
            else:
                msg = f"val_loss did not improve from {self.best}"
            print(f"epoch: {epoch} {msg} loss: {round(logs['loss'], 4)} acc: {round(logs['categorical_accuracy'], 4)} val_acc: {round(logs['val_categorical_accuracy'], 4)} val_loss: {round(logs['val_loss'], 4)}")
            
        if imp:
            getattr(self.model, self.save_mode)(save_path, overwrite=True)
            self.best = logs[self.monitor]
            
autosaver = CustomCheckpoint(save_dir=WORKING_DIR)

model.compile(
              loss='categorical_crossentropy', 
              optimizer=opt,
              metrics=['categorical_accuracy'],
)
model.summary()

model.fit(train_gen, 
          epochs=1000, 
          verbose=True, 
          callbacks=[autosaver], 
          validation_data=valid_gen, 
          use_multiprocessing=True, workers=-1)

