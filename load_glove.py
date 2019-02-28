import numpy as np
import pickle

print("[Info] Loading Glove Model")
with open("data/glove/glove.6B.50d.txt",'r', encoding="utf-8") as f:
	model = {}
	for line in f:
		splitLine = line.split()
		word = splitLine[0]
		embedding = np.array([float(val) for val in splitLine[1:]])
		model[word] = embedding

lookup = {}
dictionary = {}
lookup['dict'] = dictionary
lookup['dict']['src'] = model
lookup['dict']['tgt'] = model
with open('glove.pkl', 'wb') as f:
    pickle.dump(lookup, f)
