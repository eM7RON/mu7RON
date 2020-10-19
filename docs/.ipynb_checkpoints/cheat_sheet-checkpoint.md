# MIDI Format Cheat sheet for python-midi
---

## midi.Pattern

#### .format

A Pattern with format 0 contains a single Track. The events in this Track will be in the order in which they occur in time. For example, time signature changes will be scattered throughout the Track in locations where they occur in time.

In format 1, the very first Track should consist of only the time signature and tempo events so that it could be read by some device capable of generating a "tempo map". Normally, this would also be the Track that contains meta-events. The multiple Tracks in this Pattern would normally be played simultaneously. 

In format 2, each Track should begin with at least one initial time signature and tempo event. Each Track can be thought of as an independent sequence.

## midi.events

#### Quick Lookup Table of Unique event types:  
---
|Event Name|  Purpose     |
|----------|-------------------------|
|MetaEvent   | 
|NoteEvent   | Parent class of NoteOnEvent and NoteOffEvent
|NoteOnEvent | Indicates the sounding of a note with pitch, velocity and tick (**.data[0]**, **.data[1]** and **.tick** respectively)
|NoteOffEvent| Indicates the muting of a note with pitch, velocity and tick (**.data[0]**, **.data[1]** and **.tick** 
|AfterTouchEvent| The AfterTouchEvent is used to indicate a pressure change on one of the currently pressed MIDI keys. It has two parameters. The note number of which key's pressure is changing and the aftertouch value which specifies amount of pressure being applied (0 = no pressure, 127 = full pressure). Note Aftertouch is used for extra expression of particular notes, often introducing or increasing some type of modulation during the instrument's sustain phase.
|ControlChangeEvent| This message is sent when a controller value changes. Controllers include devices such as pedals and levers. Controller numbers 120-127 are reserved as "Channel Mode Messages" (below). **.data[0]** is the controller number (0-119). **.data[1]** is the controller value (0-127).
|ProgramChangeEvent | Indicates the instrument/voice of its containing midi.Track after this point (**.tick**) onward (**.data[0]** controls instrument, see midi standard)
|ChannelAfterTouchEvent|  
|PitchWheelEvent| Encodes pitch bending/vibrato
|SysexEvent|
|SequenceNumberMetaEvent|
|MetaEventWithText|
|TextMetaEvent|
|CopyrightMetaEvent| 
|TrackNameEvent|
|InstrumentNameEvent|
|LyricsEvent|
|MarkerEvent|
|CuePointEvent|
|ProgramNameEvent|
|UnknownMetaEvent|
|ChannelPrefixEvent |
|PortEvent|
|TrackLoopEvent|
|EndOfTrackEvent| Signifies the end of the midi sequence with **.tick**
|SetTempoEvent | Indicates a change at **.tick** in the rate of music per time. Methods: **.get_bpm**, **.set_bpm**, **.get_mpqn**, & **.set_mqpn** may be used. Where bpm and mqpn are beats per minute and milliseconds per quarter note.
|SmpteOffsetEvent|
|TimeSignatureEvent | Does not necessarily affect how music is played back unless the player does something unusual. Signifies the timesignature from **.tick** onwards and is mostly meant just for the sake of a human user.
|KeySignatureEvent  | Indicates to a human user what key signature in which the music is. Does not affect playback.
|SequencerSpecificEvent|  

## midi.NoteOnEvent/midi.NoteOffEvent

Sets a pitch to on/off.

#### .channel

`channel`  
&nbsp;&nbsp;&nbsp;&nbsp;type(channel) = int [0, 16)

##### notes:  
###### The MIDI standard reserves channel 10 for percussion. This is channel 9 in python-midi which starts its numbering from 0.

#### .data

`data`  
&nbsp;&nbsp;&nbsp;&nbsp;type(data) = list  
&nbsp;&nbsp;&nbsp;&nbsp;len(data) = 2  
&nbsp;&nbsp;&nbsp;&nbsp;data = [pitch, velocity]    
`pitch`  
&nbsp;&nbsp;&nbsp;&nbsp;type(pitch) = int, [0, 128)  
`velocity`  
&nbsp;&nbsp;&nbsp;&nbsp;type(velocity) = int, [0, 128)

##### notes:  

###### Purpose of NoteOneEvent.data found [here](https://stackoverflow.com/questions/29718532/how-to-interpret-values-of-parameters-of-midi-file-analysis-especially-the-data).

###### `It is common for a note on with 0 velocity to be interpreted as note off` [here](https://stackoverflow.com/questions/48687756/midi-note-on-event-without-off-event).

###### `NoteOffEvents sometimes have non-zero velocity`. I found an explaination [here](https://stackoverflow.com/questions/3306866/why-is-there-a-velocity-parameter-for-note-off-events).

###### "... I'm not sure of the intention behind its origins but the classic use-case for note off velocity is a harpsichord — the hammer falls differently depending on the speed of the release and changes the sound. Some sample libraries (usually 'complete'ish ones for specific instruments like piano or violin) include separate "key off" samples. Release velocity can be used to vary the level and length of those samples... "

## midi.ProgramChangeEvent

ProgramChangeEvent sets the instrument used for the playback of the Track in which it occurs from that point onwards. The data value for one of these events indicates the instrument. The numbers form part of the [General Midi standard](https://en.wikipedia.org/wiki/General_MIDI). I copied all of the intrument numbering information from tables on a website into some text documents and saved them in the data folder.

#### .data

`data`  
&nbsp;&nbsp;&nbsp;&nbsp;type(data) = int  
&nbsp;&nbsp;&nbsp;&nbsp;data = instrument  
`instrument`  
&nbsp;&nbsp;&nbsp;&nbsp;if midi.ProgramChangeEvent.channel == 9:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;data is in [34, 81) (for percussion)  
&nbsp;&nbsp;&nbsp;&nbsp;else data in [0, 128)

## midi.SetTempoEvent

FF 51 03 tt tt tt

tt tt tt is a 24-bit value specifying the tempo as the number of microseconds per quarter note.

Specifying tempos as time per beat, rather than the more usual (musically) beat per time, ensures precise long-term synchronisation with a time-based synchronisation protocol such as SMPTE or MIDI time code.

Ideally, Tempo events should only occur where MIDI clocks would be located – a convention intended to maximise compatibility with other synchronisation devices, thus allowing easy transfer of time signature / tempo map information between devices.

There should generally be a Tempo event at the beginning of a track (at time = 0), otherwise a default tempo of 120 bpm will be assumed. Thereafter they can be used to effect an immediate tempo change at any point within a track.

For a format 1 MIDI file, Tempo Meta events should only occur within the first MTrk chunk (i.e. the tempo track).

## midi.TimeSignatureEvent  

FF 58 04 nn dd cc bb

nn is a byte specifying the numerator of the time signature (as notated).
dd is a byte specifying the denominator of the time signature as a negative power of 2 (i.e. 2 represents a quarter-note, 3 represents an eighth-note, etc).
cc is a byte specifying the number of MIDI clocks between metronome clicks.
bb is a byte specifying the number of notated 32nd-notes in a MIDI quarter-note (24 MIDI Clocks). The usual value for this parameter is 8, though some sequencers allow the user to specify that what MIDI thinks of as a quarter note, should be notated as something else.
Examples

A time signature of 4/4, with a metronome click every 1/4 note, would be encoded :
FF 58 04 04 02 18 08
There are 24 MIDI Clocks per quarter-note, hence cc=24 (0x18).

A time signature of 6/8, with a metronome click every 3rd 1/8 note, would be encoded :
FF 58 04 06 03 24 08
Remember, a 1/4 note is 24 MIDI Clocks, therefore a bar of 6/8 is 72 MIDI Clocks.
Hence 3 1/8 notes is 36 (=0x24) MIDI Clocks.

There should generally be a Time Signature Meta event at the beginning of a track (at time = 0), otherwise a default 4/4 time signature will be assumed. Thereafter they can be used to effect an immediate time signature change at any point within a track.

For a format 1 MIDI file, Time Signature Meta events should only occur within the first MTrk chunk.