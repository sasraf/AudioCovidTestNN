import librosa
import torch
import matplotlib as plt

# Librosa preprocessing example


fn = 'data/breathing-deep.wav'

librosa_audio, librosa_sample_rate = librosa.load(fn)
mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc = 40)


print(mfccs.shape)
print(mfccs)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

exit(0)
