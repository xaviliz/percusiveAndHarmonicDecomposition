import numpy as np
import subprocess
import math, copy, sys, os
from scipy.io.wavfile import write, read
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift
import scipy.signal as signal
import dftModel as DFT
import utilFunctions as UF

def fadeOut(dur, fs, audioLength):
	nSamp = np.floor(dur * fs)
	ar = np.linspace(-10,0, nSamp-2)
	out = np.exp(1-ar)
	out = out/max(out)
	out = np.append(0, out)
	out = np.append(out, 0)
	out = np.concatenate((np.ones(audioLength-nSamp),out))
	return out

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

def wavread(filename):
	"""
	Read a sound file and convert it to a normalized floating point array
	filename: name of file to read
	returns fs: sampling rate of file, x: floating point array
	"""
	if (os.path.isfile(filename) == False):                  # raise error if wrong input file
		raise ValueError("Input file is wrong")

	fs, x = read(filename)
	"""
	if (len(x.shape) !=1):                                   # raise error if more than one channel
		raise ValueError("Audio file should be mono")

	if (fs !=44100):                                         # raise error if more than one channel
		raise ValueError("Sampling rate of input sound should be 44100")
	"""

	#scale down and convert audio into floating point number in range of -1 to 1
	x = np.float32(x)/norm_fact[x.dtype.name]
	return fs, x

def wavwrite(y, fs, filename):
	"""
	Write a sound file from an array with the sound and the sampling rate
	y: floating point array of one dimension, fs: sampling rate
	filename: name of file to create
	"""

	x = copy.deepcopy(y)                         # copy array
	x *= INT16_FAC                               # scaling floating point -1 to 1 range signal to int16 range
	x = np.int16(x)                              # converting to int16 type
	write(filename, fs, x)

def wavplay(filename):
	"""
	Play a wav audio file from system using OS calls
	filename: name of file to read
	"""
	if (os.path.isfile(filename) == False):                  # raise error if wrong input file
		print("Input file does not exist. Make sure you computed the analysis/synthesis")
	else:
		if sys.platform == "linux" or sys.platform == "linux2":
		    # linux
		    subprocess.call(["aplay", filename])

		elif sys.platform == "darwin":
			# OS X
			subprocess.call(["afplay", filename])
		elif sys.platform == "win32":
			if winsound_imported:
				winsound.PlaySound(filename, winsound.SND_FILENAME)
			else:
				print("Cannot play sound, winsound could not be imported")
		else:
			print("Platform not recognized")

def stftMedianFiltering(x, fs, w, N, H, filter):
	"""
	Apply a filter to a sound by using the STFT
	x: input sound, w: analysis window, N: FFT size, H: hop size
	filter: magnitude response of filter with frequency-magnitude pairs (in dB)
	returns y: output sound
	"""
	M = w.size                                     # size of analysis window
	hM1 = int(math.floor((M+1)/2))                 # half analysis window size by rounding
	hM2 = int(math.floor(M/2))                     # half analysis window size by floor
	x = np.append(np.zeros(hM2),x)                 # add zeros at beginning to center first window at sample 0
	x = np.append(x,np.zeros(hM1))                 # add zeros at the end to analyze last sample
	pin = hM1                                      # initialize sound pointer in middle of analysis window       
	pend = x.size-hM1                              # last sample to start a frame
	w = w / sum(w)                                 # normalize analysis window
	y = np.zeros(x.size)                           # initialize output array
	while pin<=pend:                               # while sound pointer is smaller than last sample      
	#-----analysis-----  
		x1 = x[pin-hM1:pin+hM2]                    # select one frame of input sound
		mX, pX = DFT.dftAnal(x1, w, N)             # compute dft
	#------transformation-----
		mY = mX + filter                           # filter input magnitude spectrum
	#-----synthesis-----
		y1 = DFT.dftSynth(mY, pX, M)               # compute idft
		y[pin-hM1:pin+hM2] += H*y1                 # overlap-add to generate output sound
		pin += H                                   # advance sound pointer
	y = np.delete(y, range(hM2))                   # delete half of first window which was added in stftAnal
	y = np.delete(y, range(y.size-hM1, y.size))    # add zeros at the end to analyze last sample
	return y

# Start processing
print "Reading audio files... "
# Read audio file
wavefile = "billie_jean3.wav"
fs, x = wavread(wavefile)
if x.shape[1]>1:
	x = x[:,0]
#fs, x = wavread("Downloads/trumpet16bitsmono.wav")
# Define FFTsize
N = 4096
N2 = N/2
overlap = N/4;
step_size = N-overlap

# Adapt audio length to step_size
pad = step_size- round(float(x.size-N)%step_size)
x = np.concatenate((x,np.zeros(pad)))

"""
# Display waveform
plt.figure(figsize=(20,5))
plt.plot(x)
plt.axis([0, x.size, -1.1, 1.1])
plt.title("Original signal")
plt.ylabel("Amplitude[V]")
plt.xlabel("Time[s]")
plt.grid()
plt.show()
"""

# Compute STFT
print "Computing STFT... "
w = signal.blackmanharris(N);	# without zero padding
M = w.size                                     # size of analysis window
hM1 = int(math.floor((M+1)/2))                 # half analysis window size by rounding
hM2 = int(math.floor(M/2))                     # half analysis window size by floor
x = np.append(np.zeros(hM2),x)                 # add zeros at beginning to center first window at sample 0
x = np.append(x,np.zeros(hM1))                 # add zeros at the end to analyze last sample
pin = hM1                                      # initialize sound pointer in middle of analysis window       
pend = x.size-hM1                              # last sample to start a frame
w = w / sum(w)                                 # normalize analysis window
yp = np.zeros(x.size)                           # initialize output array
yh = np.zeros(x.size)                           # initialize output array
nframes = (x.size-M)/overlap+1
S = np.zeros(shape=(nframes, hM2+1))  # initialize magnitude of spectrogram array
Ph = np.zeros(shape=(nframes, hM2+1))  # initialize phases of spectrogram array
ct = 0;
while pin<=pend:                               # while sound pointer is smaller than last sample      
	#-----analysis-----  
		x1 = x[pin-hM1:pin+hM2]                    # select one frame of input sound
		mX, pX = DFT.dftAnal(x1, w, N)             # compute dft
		S[ct, :] = 10.0**(mX/20.0);
		Ph[ct,:] = pX
		ct += 1;
		pin += overlap                                   # advance sound pointer

# Display spectrogram S
xa, ya = np.mgrid[:nframes, :hM2+1]
#UF.plotspecgram(20.0*np.log10(S),xa,ya,nframes,hM2, "Original Spectrogram [S]")

# Compute Median filters
print "Computing Median filters... "
k = 17 # median filter length
kb = (k-1)/2

print "\t---> Computing Percussive components... "
P = np.zeros(shape=(nframes, hM2+1))  # initialize percussive components array
for frames in range(0,nframes):
	l = 0
	# Display frame iterations
	sys.stdout.write("\r\t\tfor frame: %i/%i" % (frames,nframes-1))
	sys.stdout.flush()
	while l<hM2+1:
		# Define computation on boundaries
		if l<kb+1:
			P[frames, l] = np.median(S[frames,0:l+kb])
		elif l>hM2+1-kb:
			P[frames, l] = np.median(S[frames,l-kb:])
		else:
			P[frames, l] = np.median(S[frames,l-kb-1:l+kb])
		l += 1
print ""
# Clipping zero values to epsilon on P
P[P==0] = np.finfo(float).eps 
# Display spectrogram P
#UF.plotspecgram(20.0*np.log10(P),xa,ya,nframes,hM2, "Percussive Spectrogram [P]")

print "\t---> Computing Harmonic components... "
H = np.zeros(shape=(nframes, hM2+1))  # initialize harmonic components array
for freqs in range(0,hM2+1):
    m = 0
    # Display frame iterations
    sys.stdout.write("\r\t\tfor frequency slice: %i/%i" % (freqs, hM2))
    sys.stdout.flush()
    while m<nframes-1:
    	# Define computation on boundaries
		if m<kb+1:
			H[m,freqs] = np.median(S[0:m+kb,freqs])
		elif m>nframes-kb-1:
			H[m,freqs] = np.median(S[m-kb:,freqs])
		else:
			H[m,freqs] = np.median(S[m-kb-1:m+kb,freqs])
		m += 1
print ""
# Clipping zero values to epsilon
H[H==0] = np.finfo(float).eps 
# Display spectrogram H
#UF.plotspecgram(20.0*np.log10(H),xa,ya,nframes,hM2, "Harmonic Spectrogram [H]")

# Define P and H
print "Defining P and H with bynary masking... "
bMh = np.zeros(shape=(nframes, hM2+1))  # initialize harmonic components array
bMh = np.greater(H,P)
# Display binart mask bMh
#UF.plotspecgram(bMh,xa,ya,nframes,hM2, "Harmonic binary mask [bMh]")

bMp = np.zeros(shape=(nframes, hM2+1))  # initialize harmonic components array
bMp = np.greater(P,H)

print "Defining P and H with Wiener Filtering... "
# Computing Wiener Filter for H
p = 2.0
wMh = np.zeros(shape=(nframes, hM2+1))  # initialize harmonic components array
wMh = H**p/(H**p+P**p)
# Display Wiener Filtering on P
#UF.plotspecgram(20.0*np.log10(wMh),xa,ya,nframes,hM2, "Harmonic Wiener Filter [Mh]")

# Computing Wiener Filter for P
wMp = np.zeros(shape=(nframes, hM2+1))  # initialize harmonic components array
wMp = P**p/(H**p+P**p)
# Display Wiener Filtering on P
#UF.plotspecgram(20.0*np.log10(wMp),xa,ya,nframes,hM2, "Percussive Wiener Filter [Mp]")

# Applying masks on original spectrogram [S]
# Compute iSTFT

#-----synthesis-----
print "Computing iSTFT... "
pin = hM1
ct = 0
while pin<=pend:
		y1 = DFT.dftSynth(20.0*np.log10(S[ct,:]), Ph[ct,:], wMp[ct,:], N)    # compute idft on transformation
		y2 = DFT.dftSynth(20.0*np.log10(S[ct,:]), Ph[ct,:], wMh[ct,:], N)    # compute idft on transformation
		yp[pin-hM1:pin+hM2] += overlap*y1                 # overlap-add to generate output sound
		yh[pin-hM1:pin+hM2] += overlap*y2
		pin += overlap
		ct += 1                                   # advance sound pointer
yp = np.delete(yp, range(hM2))                   # delete half of first window which was added in stftAnal
yp = np.delete(yp, range(yp.size-hM1, yp.size))    # add zeros at the end to analyze last sample
yh = np.delete(yh, range(hM2))
yh = np.delete(yh, range(yh.size-hM1, yh.size))

# Write wave file
print "Writing audio files... "
wavwrite(yp, fs, "percussive_" + wavefile + ".wav")
wavwrite(yh, fs, "harmonic_" + wavefile + ".wav")

# Listening percussive components
print "Playing percussive extraction... "
wavplay("percussive_" + wavefile + ".wav")
print "Playing harmonic extraction... "
wavplay("harmonic_" + wavefile + ".wav")

print "Done!"
