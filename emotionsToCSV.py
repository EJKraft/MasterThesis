import csv
from os import listdir
import numpy as np
import sys

input_folder = sys.argv[1]
#outputfolder = sys.argv[2]
#input_folder = '~/opensmile-2.3.0/Results/'


txt_files = listdir(input_folder)
txt_files.sort()
print('FILES' + str(txt_files))

#We want to iterate over the folder containing the txt data and write to csv
for i in range(0,len(txt_files)):
    #load txt file
    with open(input_folder + txt_files[i], 'r') as data:
        lines = data.readlines()
        #first we want to read arousal 
        ar_line = lines[1]
        arousal = ar_line[93:]
        #then we want to read valence
        val_line = lines[3]
        valence = val_line[93:]
        #For emodbEmotion and abcAffect it's more complicated, since we have to save multiple values, not only one
        emo_line = lines[4]
        emo_temp = emo_line.split(";",1)
        emo = emo_temp[1].split("::")
        emotion = [e[e.index(":") + 1:] for e in emo]
        emotion[len(emotion)-1] = emotion[len(emotion)-1][:8]

        #Now do the same for affect
        aff_line = lines[5]
        aff_temp = aff_line.split(";",1)
        aff = aff_temp[1].split("::")
        affect = [a[a.index(":") + 1:] for a in aff]
        affect[len(affect)-1] = affect[len(affect)-1][:8]
