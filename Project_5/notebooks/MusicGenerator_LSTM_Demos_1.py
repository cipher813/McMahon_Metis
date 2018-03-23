
# coding: utf-8

# In[1]:


import pretty_midi
import numpy as np
# For plotting
import mir_eval.display
import librosa.display
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# For putting audio in the notebook
import IPython.display


# ### Sample Training Midi Files

# In[4]:


fs = 44100
rate = 44100
filepath = '/home/ubuntu/GitClones/McMahon_Metis/Project_5/data/Mozart-MuseData/M-k450-01.mid'
pm = pretty_midi.PrettyMIDI(filepath)
IPython.display.Audio(pm.synthesize(fs=fs),rate=rate)


# In[19]:


def plot_piano_roll(pm_o1, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm_o1.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))

def display_midi(filepath,start_pitch, end_pitch, fs,rate):
    pm = pretty_midi.PrettyMIDI(filepath)
    length = pm.get_end_time()
#     adj_length = 10
#     pm.adjust_times([0,length],[0,length*1.0])
#     print("Midi file original length: {} Adjusted length: {}".format(length,adj_length))
    plt.figure(figsize=(16, 4))
    plot_piano_roll(pm, start_pitch, end_pitch, fs)
    return IPython.display.Audio(pm.synthesize(fs=fs), rate=rate)


# In[20]:


fs = 44100
rate = 16000
start_pitch = 24 
end_pitch = 84


# ### Input File Samples

# In[8]:


filepath = '/home/ubuntu/GitClones/McMahon_Metis/Project_5/data/Mozart-MuseData/M-k450-01.mid'
display_midi(filepath,start_pitch, end_pitch, fs,rate)


# ### Output File Samples

# In[ ]:


filepath = '/home/ubuntu/GitClones/McMahon_Metis/Project_5/output/200.mid'
display_midi(filepath,start_pitch, end_pitch, fs,rate)


# In[13]:


filepath = '/tmp/melody_rnn/generated/2018-03-18_221631_01.mid'
display_midi(filepath,start_pitch, end_pitch, fs,rate)


# In[14]:


filepath = '/tmp/melody_rnn/generated/2018-03-18_221631_02.mid'
display_midi(filepath,start_pitch, end_pitch, fs,rate)


# In[15]:


filepath = '/tmp/melody_rnn/generated/2018-03-18_221631_10.mid'
display_midi(filepath,start_pitch, end_pitch, fs,rate)


# In[16]:


filepath = '/home/ubuntu/Music/melody_rnn/generated/2018-03-18_224915_01.mid'
display_midi(filepath,start_pitch, end_pitch, fs,rate)


# In[17]:


filepath = '/home/ubuntu/Music/melody_rnn/generated/2018-03-18_224915_10.mid'
display_midi(filepath,start_pitch, end_pitch, fs,rate)


# In[ ]:


pm1 = pretty_midi.PrettyMIDI('/home/ubuntu/GitClones/McMahon_Metis/Project_5/data/Mozart-MuseData/M-k450-01.mid')
pm2 = pretty_midi.PrettyMIDI('/home/ubuntu/GitClones/McMahon_Metis/Project_5/data/Mozart-MuseData/M-k450-02.mid')
pm3 = pretty_midi.PrettyMIDI('/home/ubuntu/GitClones/McMahon_Metis/Project_5/data/Mozart-MuseData/M-k450-03.mid')


# In[ ]:


IPython.display.Audio(pm1.synthesize(fs=16000), rate=16000)


# In[ ]:


IPython.display.Audio(pm2.synthesize(fs=16000), rate=16000)


# In[ ]:


IPython.display.Audio(pm3.synthesize(fs=16000), rate=16000)


# ### Sample Output Files

# In[ ]:


pm_o1 = pretty_midi.PrettyMIDI('/home/ubuntu/GitClones/McMahon_Metis/Project_5/output/20.mid')


# In[ ]:


IPython.display.Audio(pm_o1.synthesize(fs=16000), rate=16000)


# In[ ]:


def plot_piano_roll(pm_o1, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm_o1.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))

plt.figure(figsize=(8, 4))
plot_piano_roll(pm, 56, 70)
# Note the blurry section between 1.5s and 2.3s - that's the pitch bending up!


# In[ ]:


# Construct a PrettyMIDI object.
# We'll specify that it will have a tempo of 80bpm.
pm = pretty_midi.PrettyMIDI(initial_tempo=80)


# In[ ]:


# The instruments list from our PrettyMIDI instance starts empty
print(pm.instruments)


# In[ ]:


# Let's add a Cello instrument, which has program number 42.
# pretty_midi also keeps track of whether each instrument is a "drum" instrument or not
# because drum/non-drum instruments share program numbers in MIDI.
# You can also optionally give the instrument a name,
# which corresponds to the MIDI "instrument name" meta-event.
inst = pretty_midi.Instrument(program=42, is_drum=False, name='my cello')
pm.instruments.append(inst)


# In[ ]:


audio_file = '/home/ubuntu/GitClones/McMahon_Metis/Project_5/output/test_output2.mid'

librosa.core.load(midi)


# In[ ]:


# audio_path = librosa.util.example_audio_file()
audio_path = '/home/ubuntu/GitClones/McMahon_Metis/Project_5/output/test_output2.mid'

# or uncomment the line below and point it at your favorite song:
#
# audio_path = '/path/to/your/favorite/song.mp3'

y, sr = librosa.load(audio_path)


# In[ ]:


# Let's make and display a mel-scaled power (energy-squared) spectrogram
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

# Convert to log scale (dB). We'll use the peak power (max) as reference.
log_S = librosa.power_to_db(S, ref=np.max)

# Make a new figure
plt.figure(figsize=(12,4))

# Display the spectrogram on a mel scale
# sample rate and hop length parameters are used to render the time axis
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

# Make the figure layout compact
plt.tight_layout()


# In[ ]:


y_harmonic, y_percussive = librosa.effects.hpss(y)


# In[ ]:


# What do the spectrograms look like?
# Let's make and display a mel-scaled power (energy-squared) spectrogram
S_harmonic   = librosa.feature.melspectrogram(y_harmonic, sr=sr)
S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)

# Convert to log scale (dB). We'll use the peak power as reference.
log_Sh = librosa.power_to_db(S_harmonic, ref=np.max)
log_Sp = librosa.power_to_db(S_percussive, ref=np.max)

# Make a new figure
plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
# Display the spectrogram on a mel scale
librosa.display.specshow(log_Sh, sr=sr, y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram (Harmonic)')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

plt.subplot(2,1,2)
librosa.display.specshow(log_Sp, sr=sr, x_axis='time', y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram (Percussive)')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

# Make the figure layout compact
plt.tight_layout()


# In[ ]:


# We'll use a CQT-based chromagram here.  An STFT-based implementation also exists in chroma_cqt()
# We'll use the harmonic component to avoid pollution from transients
C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

# Make a new figure
plt.figure(figsize=(12,4))

# Display the chromagram: the energy in each chromatic pitch class as a function of time
# To make sure that the colors span the full range of chroma values, set vmin and vmax
librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)

plt.title('Chromagram')
plt.colorbar()

plt.tight_layout()


# In[ ]:


# Next, we'll extract the top 13 Mel-frequency cepstral coefficients (MFCCs)
mfcc        = librosa.feature.mfcc(S=log_S, n_mfcc=13)

# Let's pad on the first and second deltas while we're at it
delta_mfcc  = librosa.feature.delta(mfcc)
delta2_mfcc = librosa.feature.delta(mfcc, order=2)

# How do they look?  We'll show each in its own subplot
plt.figure(figsize=(12, 6))

plt.subplot(3,1,1)
librosa.display.specshow(mfcc)
plt.ylabel('MFCC')
plt.colorbar()

plt.subplot(3,1,2)
librosa.display.specshow(delta_mfcc)
plt.ylabel('MFCC-$\Delta$')
plt.colorbar()

plt.subplot(3,1,3)
librosa.display.specshow(delta2_mfcc, sr=sr, x_axis='time')
plt.ylabel('MFCC-$\Delta^2$')
plt.colorbar()

plt.tight_layout()

# For future use, we'll stack these together into one matrix
M = np.vstack([mfcc, delta_mfcc, delta2_mfcc])


# In[ ]:


# Now, let's run the beat tracker.
# We'll use the percussive component for this part
plt.figure(figsize=(12, 6))
tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)

# Let's re-draw the spectrogram, but this time, overlay the detected beats
plt.figure(figsize=(12,4))
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

# Let's draw transparent lines over the beat frames
plt.vlines(librosa.frames_to_time(beats),
           1, 0.5 * sr,
           colors='w', linestyles='-', linewidth=2, alpha=0.5)

plt.axis('tight')

plt.colorbar(format='%+02.0f dB')

plt.tight_layout()


# In[ ]:


print('Estimated tempo:        %.2f BPM' % tempo)

print('First 5 beat frames:   ', beats[:5])

# Frame numbers are great and all, but when do those beats occur?
print('First 5 beat times:    ', librosa.frames_to_time(beats[:5], sr=sr))

# We could also get frame numbers from times by librosa.time_to_frames()


# In[ ]:


# feature.sync will summarize each beat event by the mean feature vector within that beat

M_sync = librosa.util.sync(M, beats)

plt.figure(figsize=(12,6))

# Let's plot the original and beat-synchronous features against each other
plt.subplot(2,1,1)
librosa.display.specshow(M)
plt.title('MFCC-$\Delta$-$\Delta^2$')

# We can also use pyplot *ticks directly
# Let's mark off the raw MFCC and the delta features
plt.yticks(np.arange(0, M.shape[0], 13), ['MFCC', '$\Delta$', '$\Delta^2$'])

plt.colorbar()

plt.subplot(2,1,2)
# librosa can generate axis ticks from arbitrary timestamps and beat events also
librosa.display.specshow(M_sync, x_axis='time',
                         x_coords=librosa.frames_to_time(librosa.util.fix_frames(beats)))

plt.yticks(np.arange(0, M_sync.shape[0], 13), ['MFCC', '$\Delta$', '$\Delta^2$'])             
plt.title('Beat-synchronous MFCC-$\Delta$-$\Delta^2$')
plt.colorbar()

plt.tight_layout()


# In[ ]:


# Beat synchronization is flexible.
# Instead of computing the mean delta-MFCC within each beat, let's do beat-synchronous chroma
# We can replace the mean with any statistical aggregation function, such as min, max, or median.

C_sync = librosa.util.sync(C, beats, aggregate=np.median)

plt.figure(figsize=(12,6))

plt.subplot(2, 1, 1)
librosa.display.specshow(C, sr=sr, y_axis='chroma', vmin=0.0, vmax=1.0, x_axis='time')

plt.title('Chroma')
plt.colorbar()

plt.subplot(2, 1, 2)
librosa.display.specshow(C_sync, y_axis='chroma', vmin=0.0, vmax=1.0, x_axis='time', 
                         x_coords=librosa.frames_to_time(librosa.util.fix_frames(beats)))


plt.title('Beat-synchronous Chroma (median aggregation)')

plt.colorbar()
plt.tight_layout()

