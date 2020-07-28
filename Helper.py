import numpy as np
import pandas as pd
import itertools as it
from statsmodels.sandbox.stats.multicomp import multipletests
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt

def cohen_d(data1, data2):
	n1, n2 = len(data1), len(data2)
	dof = n1 + n2 - 2
	s1, s2 = np.var(data1, ddof = 1), np.var(data2,ddof=1)
	pool_std = np.sqrt(((n1-1) * s1 + (n2-1) * s2)/dof)
	u1,u2 = np.mean(data1), np.mean(data2)
	res = (u1-u2)/pool_std
	print('Cohen d: ' + str(res))
	return res
	
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
	if(char_feature == 'Sex'):
		data_melt = data.Sex.replace({0.0: "male", 1.0: "female"}, inplace = True)	
	elif(char_feature == 'Academic Status'):
		data_melt = data[char_feature].replace({0.0: "Bachelor", 1.0: "Master"}, inplace = True)
	elif(char_feature == 'Age'):
		data_melt = data[char_feature].replace({23: "Young", 24: "Middle", 25: "Old"}, inplace = True)
	data_melt = data.melt(var_name = type, value_name = 'Probability of ' + type, id_vars = char_feature)
	sns.catplot(x = type, y = 'Probability of ' + type, hue = char_feature, kind = kind, data = data_melt)
	return

def distPlots(data, features):
	for f in features:
		plt.figure(f)
		sns.kdeplot(data[f], shade = True)
	return
	
def chi2_post_hoc(fre_table, method, shouldPrint= False):
	all_combis = list(it.combinations(fre_table.index,2))
	p_vals = []
	for comb in all_combis:
		#Create new data frame from combinations to conduct chi2 independence test
		new_df = fre_table[(fre_table.index == comb[0]) | (fre_table.index == comb[1])]
		chi2_ph = st.chi2_contingency(new_df, correction = True)
		p_vals.append(chi2_ph[1])
	reject_list, corrected_p_vals = multipletests(p_vals, method = method)[:2]
	if(shouldPrint == True):
		print('Combinations: ' + str(all_combis))
		print('Reject List: ' + str(reject_list))
		print('Corrected p-values: ' + str(corrected_p_vals))
	return [reject_list, corrected_p_vals, all_combis]
		
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
			
