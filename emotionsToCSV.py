import csv
from os import listdir
import numpy as np
import pandas as pd
import sys

input_folder = sys.argv[1]
#outputfolder = sys.argv[2]
#input_folder = '~/opensmile-2.3.0/Results/'


txt_files = listdir(input_folder)
txt_files.sort()
print('Processing ' + str(txt_files))

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
        df_ar = pd.DataFrame([arousal])
        df_ar.columns = ['arousal']
        df_val = pd.DataFrame([valence])
        df_val.columns = ['valence']
        df_emo = pd.DataFrame([emotion])
        df_emo = df_emo.transpose()
        df_emo.columns = ['emodbEmotion']
        df_aff = pd.DataFrame([affect])
        df_aff = df_aff.transpose()
        df_aff.columns = ['abcAffect']
        data = pd.concat([df_emo, df_aff],ignore_index = True, axis = 1)
        data = pd.concat([data, df_ar], ignore_index = True, axis = 1)
        data = pd.concat([data, df_val], ignore_index = True, axis = 1)
        data.columns = ['emodbEmotion', 'abcAffect', 'arousal', 'valence']

        #d = {'arousal': arousal, 'valence': valence, 'abcAffect': affect, 'emodbEmotion': emotion}
        #df_data = pd.DataFrame(data=d)
        print(data)
        data.to_csv(txt_files[i][:-4] + '.csv',index = False)
