import librosa 
import csv
from os import walk 

#feature 1 : zero-crossing 
def extract_zero_crossing(file_path):
    n0 = 9000
    n1 = 9100
    x, sr = librosa.load(file_path)
    zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
    return [sum(zero_crossings)]

#feature 2 : chroma frequencies
def extract_chroma(file_path) :
    x, sr = librosa.load(file_path)
    hop_length = 512
    chroma = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
    return chroma

""" res = extract_chroma(file_path=r"data\genres_original\blues\blues.00000.wav")
print(res) """

#feature 3 : spectral centroids
def extract_spectral_centroids(file_path) : 
    x, sr = librosa.load(file_path)
    spectral_centroids = librosa.feature.spectral_centroid(x,sr=sr)[0]
    return spectral_centroids

#feature 4 : spectral rolloff
def extract_spectral_rolloff(file_path) : 
    x, sr = librosa.load(file_path)
    spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
    return spectral_rolloff

#feature 5 : mfccs
def extract_mfccs(file_path) : 
    x, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(x,sr)
    return mfccs 
   
#write resultat in csv : 
def write_csv(name,res,csvfile) :
    with open(csvfile,'a',newline='') as csvfile : 
        filewriter = csv.writer(csvfile)
        res = [str(e) for e in res]
        res = "|".join(res)
        filewriter.writerow([name,res])

# extract file name in folder:
def extract_filename (folder_path) :
    f = []
    for (dirpath, dirnames, filenames) in walk(folder_path):
        f.extend(filenames)
        break
    return f


# #for loop
# for file in lf : 
#     #res = extract_zero_crossing(file_path=r"C:\Users\trang\Desktop\Exemple_projet\music_classification\Data\genres_original\blues\%s" % (file))
#     write_csv(file, res , "./train_zero_crossing.csv")

