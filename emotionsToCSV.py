import csv
from os import listdir
import numpy as np
import pandas as pd
import sys

input_folder = sys.argv[1]
#outputfolder = sys.argv[2]
#input_folder = '~/opensmile-2.3.0/Results/'
head = ['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Emo_Neutral', 'Sadness', 'Aggressiv', 'Cheerful', 'Intoxicated', 'Nervous', 'Aff_Neutral', 'Tired','Disinterest', 'Normal', 'High Interest', 'Arousal', 'Valence', 'Filename']
res_table = []

txt_files = listdir(input_folder)
txt_files.sort()

#We want to iterate over the folder containing the txt data and write to csv
for i in range(0,len(txt_files)):
    #load txt file
    with open(input_folder + txt_files[i], 'r') as data:
        print('Processing file ' + txt_files[i] + ' ...')
        test = []
        lines = data.readlines()
        #first we want to read arousal 
        ar_line = lines[1]
        arousal = ar_line[93:]
        arousal = arousal.rstrip()
        #test.append(arousal)
        #then we want to read valence
        val_line = lines[3]
        valence = val_line[93:]
        valence = valence.rstrip()
        #test.append(valence)
        #For emodbEmotion and abcAffect it's more complicated, since we have to save multiple values, not only one
        emo_line = lines[4]
        emo_temp = emo_line.split(";",1)
        emo = emo_temp[1].split("::")
        emotion = [e[e.index(":") + 1:] for e in emo]
        emotion[len(emotion)-1] = emotion[len(emotion)-1][:8]
        test.append(emotion)
        #Now do the same for affect
        aff_line = lines[5]
        aff_temp = aff_line.split(";",1)
        aff = aff_temp[1].split("::")
        affect = [a[a.index(":") + 1:] for a in aff]
        affect[len(affect)-1] = affect[len(affect)-1][:8]
        test.append(affect)
        #Now look at Level of Interest
        loi_line = lines[6]
        loi_temp = loi_line.split(";",1)
        loi = loi_temp[1].split("::")
        loi = [l[l.index(":") + 1:] for l in loi]
        loi[len(loi)-1] = loi[len(loi)-1][:8]
        test.append(loi)
        test = [item for sublist in test for item in sublist]
        test.append(arousal)
        test.append(valence)
        test.append(txt_files[i][:-4])
        res_table.append(test)

data_frame = pd.DataFrame.from_records(res_table)
data_frame.columns = head
print(data_frame)
    
data_frame.to_csv('OpenSMILEAnalysis.csv',index = False)
