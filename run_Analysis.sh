#!/bin/bash
for file in "$1"/* ; do
	f="$(basename -- $file)"
	echo "Processing $file ..."
	#Run OpenSMILE Analysis for every .wav file in folder
	./SMILExtract -C config/emobase_live4_batch_single.conf -I $file > ~/MasterThesis/MasterThesis/testResults/txtFiles/$f.txt
done
