import numpy as np
import pandas as pd
import itertools as it
from statsmodels.sandbox.stats.multicomp import multipletests
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

def constructDataFrames(filenames):
	arousal = []
	valence = []
	emotion = []
	affect = []
	loi = []
	characterIDs = []
	
	fnames = []
	filenames.sort()
	#filenames contains the hole path, not only the filenames
	for x in filenames:
		fnames.append([x])
		
	emotion_label = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
	affect_label = ['aggressiv', 'cheerful', 'intoxicated', 'nervous', 'neutral', 'tired']
	loi_label = ['disinterest', 'normal', 'high interest']
		
	# Now iterate over our filenames, load the data and save it to a list for further usage
	for i in range(len(fnames)):
		 # data contains all information (arousal, valence, emotion, affect) and we want to save the values of all files in a list
		data = pd.read_csv("OpenSMILE_Data/" + filenames[i])    

		# For arousal, valence and affect we have to drop nans, since they have less values than emotion
		temp_arousal = data['arousal']
		temp_arousal = temp_arousal.dropna()
		temp_valence = data['valence'] 
		temp_valence = temp_valence.dropna()
		temp_affect = data['abcAffect']
		temp_affect = temp_affect.dropna()
		temp_emotion = data['emodbEmotion']
		temp_loi = data['avicLoI']
		temp_loi = temp_loi.dropna()
		characterIDs.append(fnames[i][0][0])
		
		#Append the temp values to 'global lists'
		emotion.append(temp_emotion.values.tolist())
		affect.append(temp_affect.values.tolist())
		valence.append(temp_valence.values)
		arousal.append(temp_arousal.values)
		loi.append(temp_loi.values)
	
	#We want to have the labels as column seperators and the filenames as ID 
	#We want to do this, so that if we add more files (at the moment only 6 .csv files are loaded) we want to add rows and not columns
	#If we plot the data, emotion_label can be used as label
	df_emotion = pd.DataFrame.from_records(emotion)
	df_emotion.columns = emotion_label
	df_emotion['CharacterID'] = characterIDs
	df_emotion['file'] = filenames

	#Now do the same for affect
	df_affect = pd.DataFrame.from_records(affect)
	df_affect.columns = affect_label
	df_affect['CharacterID'] = characterIDs
	df_affect['file'] = filenames

	#Now for loi
	df_loi = pd.DataFrame.from_records(loi)
	df_loi.columns = loi_label
	df_loi['CharacterID'] = characterIDs
	df_loi['file'] = filenames

	#For Arousal and Valence, we want to combine these two features so that we can draw a scatter plot in the arousal valence space
	np_ar = np.array(arousal).ravel()
	np_val = np.array(valence).ravel()
	np_ar_val = np.array([np_ar, np_val])

	#Transpose Matrix so that it is in the same form as affect and emotion (columns = arousal, valence, ID = Filename)
	df_ar_val = pd.DataFrame.from_records(np_ar_val.T)
	#df_ar_val.index = filenames
	df_ar_val.columns = ['valence', 'arousal']
	df_ar_val['CharacterID'] = characterIDs
	df_ar_val['file'] = filenames
	
	#We want to use the dataframes and labels so we construct us a multidimensional list which we'll return 
	# First start with the labels
	labels = [emotion_label, affect_label, loi_label, characterIDs]
	#The with the data frames
	data_frames = [df_emotion, df_affect, df_loi, df_ar_val]
	return [data_frames, labels]	
	

def cohen_d(data1, data2):
	n1, n2 = len(data1), len(data2)
	dof = n1 + n2 - 2
	s1, s2 = np.var(data1, ddof = 1), np.var(data2,ddof=1)
	pool_std = np.sqrt(((n1-1) * s1 + (n2-1) * s2)/dof)
	u1,u2 = np.mean(data1), np.mean(data2)
	res = (u1-u2)/pool_std
	print('Cohen d: ' + str(res))
	return res
	
def correlations(dataset, data, label):
	cors = []
	coh_d = []
	for x in label:
		c = correlation(dataset, data[x])
		cors.append(c)
		d = cohen_d(dataset, data[x])
	return [cors, coh_d]
	
def correlation(data1, data2):
	res = data1.corr(data2)
	print('Correlation between ' + str(data1.name) + ' and ' + str(data2.name) + ': ' + str(res))
	return res

def cleanData(data, column_name, column_value):
	res = data.loc[data[column_name]==column_value]
	res = res.drop(['CharacterID','file', 'Age', 'Sex', 'Academic Status'], axis = 1)
	return res

	#type = emotion, level of interest, affect
	#kind = kind for sns.catplot
	#char_feature = Sex/ Academic Status/ Age
def catPlot(data, type, char_feature, kind):
	data_melt = meltData(data, char_feature, type)
	g = sns.catplot(x = type, y = 'Probability of ' + type, hue = char_feature, kind = kind, data = data_melt)
	plt.subplots_adjust(top = 0.9)
	g.fig.suptitle(kind + ' plot of OpenEAR: ' + type)
	return
	
def boxPlots(data, char_feature, types):
	for i in range(0,3):
		boxPlot(data[i], char_feature, types[i])
		plt.figure()
	return
	
def boxPlot(data, char_feature,type):
	data_melt = meltData(data, char_feature,type)	
	ax = sns.boxplot(x = type, y = 'Probability of ' + type, hue = char_feature, data = data_melt)
	ax.set_title('Box Plot of OpenEAR: ' + type)
	return	

def meltData(data, char_feature, type):
	if(char_feature == 'Sex'):
		data_melt = data.Sex.replace({0.0: "male", 1.0: "female"}, inplace = True)	
	elif(char_feature == 'Academic Status'):
		data_melt = data[char_feature].replace({0.0: "Bachelor", 1.0: "Master"}, inplace = True)
	elif(char_feature == 'Age'):
		data_melt = data[char_feature].replace({23: "Young", 24: "Middle", 25: "Old"}, inplace = True)
	data_melt = data.melt(var_name = type, value_name = 'Probability of ' + type, id_vars = char_feature)
	return data_melt
	
def distPlots(data, features, type):

	if(type == 0):
		feat = 'Emotion'
	elif(type == 1):
		feat = 'Affect'
	elif(type == 2):
		feat = 'Level of Interest'
	else:
		print('Enter 0 for emotion, 1 for affect or 2 for loi!')
	for f in features:
		plt.figure()
		plt.title('Distribution of OpenEAR: ' + feat + ' ' + f)
		sns.kdeplot(data[f], shade = True)
	return
	
def chi2(data, char_feature, type, shouldPrint = False):
	if(type == 0):
		feature = 'Emotion'
	elif(type == 1):
		feature = 'Affect'
	elif(type == 2):
		feature = 'Arousal Valence'
	elif(type == 3):
		feature = 'Level of Interest'
	if(char_feature == 'Sex' or char_feature == 'Academic Status'):
		df_data_group0 = cleanData(data, char_feature, 0)
		df_data_group1 = cleanData(data, char_feature, 1)
		if(char_feature == 'Sex'):
			# Last zero stands for char_feature = Sex
			fre_table = calcFrequencyTable([df_data_group0,df_data_group1],type, 0 )
			fre_table.index = ['Male', 'Female']
		elif(char_feature == 'Academic Status'):
			fre_table = calcFrequencyTable([df_data_group0,df_data_group1],type, 1 )
			fre_table.index = ['Bachelor', 'Master']
	elif(char_feature == 'Age'):
		df_data_group0 = cleanData(data, char_feature, 23)
		df_data_group1 = cleanData(data, char_feature, 24)
		df_data_group2 = cleanData(data, char_feature, 25)
		fre_table = calcFrequencyTable([df_data_group0,df_data_group1, df_data_group2], type, 2)
		fre_table.index = ['Young', 'Middle', 'Old']
		
	# ################
	# This line will be deleted once real data is inserted!!!!
	# ########
	fre_table += 5
	chi2 = st.chi2_contingency(fre_table)
	if(shouldPrint == True):
		print('Chi square of ' + feature + ' : ' + str(chi2[0]) + ' with p-value of: ' + str(chi2[1]))
	table = sm.stats.Table(fre_table)
	residuals = table.standardized_resids
	return [chi2, fre_table, residuals]

def chi2_post_hoc(fre_table, method, shouldPrint= False, calculateResiduals = False):
	all_combis = list(it.combinations(fre_table.index,2))
	p_vals = []
	res = []
	for comb in all_combis:
		#Create new data frame from combinations to conduct chi2 independence test
		new_df = fre_table[(fre_table.index == comb[0]) | (fre_table.index == comb[1])]
		#Calculate residuals if needed
		if(calculateResiduals == True):
			new_table = sm.stats.Table(new_df)
			new_res = new_table.standardized_resids
		chi2_ph = st.chi2_contingency(new_df, correction = True)
		p_vals.append(chi2_ph[1])
		if(calculateResiduals == True):
			res.append(new_res)
	reject_list, corrected_p_vals = multipletests(p_vals, method = method)[:2]
	if(shouldPrint == True):
		print('Combinations: ' + str(all_combis))
		print('Reject List: ' + str(reject_list))
		print('Corrected p-values: ' + str(corrected_p_vals))
	if(calculateResiduals == False):
		return [reject_list, corrected_p_vals, all_combis]
	
	else:
		return [reject_list, corrected_p_vals, all_combis, res ]

# Converts a panda data frame containing decimal values to frequency tables so that the largest number for a row is counted as a frequency of e.g. the emotion anger
def calcFrequencyTable(data, voice_feature, char_feature):
	if(voice_feature == 0):
		fre_table = pd.DataFrame(columns = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness'])
	elif(voice_feature == 1):
		fre_table = pd.DataFrame(columns = ['aggressive', 'cheerful', 'intoxicated', 'nervous', 'neutral', 'tired'])
	elif(voice_feature == 2):
		fre_table = pd.DataFrame(columns = ['arousal', 'valence'])
	elif(voice_feature == 3):
		fre_table = pd.DataFrame(columns = ['disinterest', 'normal', 'high interest'])
	else:
		print('Did not provide correct number for variable voice_feature! Use 0 for emotion, 1 for affect, 2 for arousal/valence and 3 for interest!')
	
	#0 stands for sex
	if(char_feature == 0):
		maxValueIndex_male = data[0].idxmax(axis = 1)
		max_ValueIndex_female = data[1].idxmax(axis = 1)			
		temp_male = maxValueIndex_male.value_counts()
		temp_female = max_ValueIndex_female.value_counts()
		
		fre_table = fre_table.append(temp_male, ignore_index = True)
		fre_table = fre_table.append(temp_female, ignore_index = True)
		
	# 1 stands for academic status. For now it is two dimensional as sex, but that may change depending on the final data set. That's why it's seperate..
	elif(char_feature == 1):
		maxValueIndex_bach = data[0].idxmax(axis = 1)
		max_ValueIndex_mast = data[1].idxmax(axis = 1)			
		temp_bach = maxValueIndex_bach.value_counts()
		temp_mast = max_ValueIndex_mast.value_counts()
		
		fre_table = fre_table.append(temp_bach, ignore_index = True)
		fre_table = fre_table.append(temp_mast, ignore_index = True)
		
	# 2 stands for age
	elif(char_feature == 2):
		maxValueIndex_young = data[0].idxmax(axis = 1)
		maxValueIndex_middle = data[1].idxmax(axis = 1)
		maxValueIndex_old = data[2].idxmax(axis = 1)
		temp_young = maxValueIndex_young.value_counts()
		temp_middle = maxValueIndex_middle.value_counts()
		temp_old = maxValueIndex_old.value_counts()
		
		fre_table = fre_table.append(temp_young, ignore_index = True)
		fre_table = fre_table.append(temp_middle, ignore_index = True)
		fre_table = fre_table.append(temp_old, ignore_index = True)
		
	fre_table = fre_table.fillna(0)
	return fre_table
	
def logReg(data, voice_feature, char_feature, specificVoiceF):
	d = data[[char_feature, specificVoiceF]]
	f = char_feature + ' ~ ' + specificVoiceF
	model = smf.logit(formula = f, data = d)
	res = model.fit()
	return res

	
def multiLogReg(data, voice_feature, char_feature, prohibitWarning = False):
	if(char_feature == 'Sex'):
		d = data.drop(['CharacterID', 'file', 'Age', 'Academic Status'], axis = 1)
	elif(char_feature == 'Academic Status'):
		d = data.drop(['CharacterID', 'file', 'Age', 'Sex'], axis = 1)
	elif(char_feature == 'Age'):
		d = data.drop(['CharacterID', 'file', 'Sex', 'Academic Status'], axis = 1)
	f = char_feature
	if(voice_feature == 'Emotion'):
		f += ' ~ anger + boredom + disgust + fear + happiness + neutral + sadness'
	elif(voice_feature == 'Affect'):
		f += ' ~ aggressiv + cheerful + intoxicated + nervous + neutral + tired'
	elif(voice_feature == 'LOI'):
		f += ' ~ disinterest + normal + high interest'
	elif(voice_feature == 'Arousal-Valence'):
		f += ' ~ arousal + valence'
	else:
		print('Enter valid voice feature: Emotion, Affect, LOI, Arousal-Valence!')

		
	model = smf.logit(formula = f, data = d)
	if(prohibitWarning == True):
		model.raise_on_perfect_prediction = False
	res = model.fit()
	return res
