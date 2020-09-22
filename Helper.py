import numpy as np
import pandas as pd
import itertools as it
from statsmodels.sandbox.stats.multicomp import multipletests
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

def createMeanDataFrame(data, type, label):
	r = []
	for d in data:
		d = dropCharacterCols(d)
		d = d.mean()
		r.append(d)
	if(type == 'Academic'):
		 res = pd.DataFrame({'Master Student': r[0], 'PhD Student': r[1], 'Researcher': r[2], 'Professor': r[3], 'Assistant Professor': r[4], 'Post Doc': r[5]}, index = label)
	elif(type == 'Age'):
		res = pd.DataFrame({'Young': r[0], 'Intermediate': r[1], 'Old': r[2]}, index = label)
	elif(type == 'Sex'):
		res = pd.DataFrame({'Male': r[0], 'Female': r[1]})
	else:
		res = 0
	return res		

def dropCharacterCols(data):
	res = data.drop(['Char_ID','ID','Filename', 'Age', 'Sex', 'Academic Status', 'VideoTitle', 'Name'], axis = 1)
	return res
	
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
	res = res.drop(['ID','Char_ID','Filename', 'Age', 'Sex', 'Academic Status'], axis = 1)
	return res

	#type = emotion, level of interest, affect
	#kind = kind for sns.catplot
	#char_feature = Sex/ Academic Status/ Age
def catPlot(data, type, char_feature, kind):
	data_melt = meltData(data, char_feature, type)
	g = sns.catplot(x = type, y = 'Probability of ' + type, hue = char_feature, kind = kind, data = data_melt)	
	plt.subplots_adjust(top = 0.9)
	g.fig.suptitle(kind + ' plot of OpenEAR: ' + type)
	plt.xticks(rotation = 20)
	return
	
def boxPlots(data, char_feature, types):
	for i in range(0,3):
		plt.xticks(rotation=20)
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
		d = data.drop(['Char_ID','ID','Name','VideoTitle', 'Filename', 'Age', 'Academic'], axis = 1)
		d.Sex.replace({'Male': 0.0, 'Female':1.0}, inplace = True)
	elif(char_feature == 'Academic'):
		d = data.drop(['Char_ID', 'ID', 'Filename', 'Name', 'VideoTitle', 'Age', 'Sex'], axis = 1)
	elif(char_feature == 'Age'):
		d = data.drop(['Char_ID', 'ID', 'Filename', 'Name', 'VideoTitle','Sex', 'Academic'], axis = 1)
	else:
		print('Either use Sex, Academic or Age as input for character feature')
	
	f = char_feature
	
	if(voice_feature == 'Emotion'):
		f += ' ~ Anger + Boredom + Disgust + Fear + Happiness + Emo_Neutral + Sadness'
	elif(voice_feature == 'Affect'):
		f += ' ~ Aggressiv + Cheerful + Intoxicated + Nervous + Aff_Neutral + Tired'
	elif(voice_feature == 'LOI'):
		f += ' ~ Disinterest + Normal + High_Interest'
	elif(voice_feature == 'Arousal-Valence'):
		f += ' ~ Arousal + Valence'
	else:
		print('Enter valid voice feature: Emotion, Affect, LOI, Arousal-Valence!')

		
	model = smf.logit(formula = f, data = d)
	if(prohibitWarning == True):
		model.raise_on_perfect_prediction = False
	res = model.fit()
	return res
	
def multiNomiLogReg(data, voice_feature, char_feature, prohibitWarning = False):
	
	if(char_feature == 'Academic'):
		d = data.drop(['Char_ID', 'ID', 'Filename', 'Name', 'VideoTitle', 'Age', 'Sex'], axis = 1)
	elif(char_feature == 'Age'):
		d = data.drop(['Char_ID', 'ID', 'Filename', 'Name', 'VideoTitle','Sex', 'Academic'], axis = 1)
	else:
		print('Either use Sex, Academic or Age as input for character feature')
	f = char_feature
	if(voice_feature == 'Emotion'):
		f += ' ~ Anger + Boredom + Disgust + Fear + Happiness + Emo_Neutral + Sadness'
	elif(voice_feature == 'Affect'):
		f += ' ~ Aggressiv + Cheerful + Intoxicated + Nervous + Aff_Neutral + Tired'
	elif(voice_feature == 'LOI'):
		f += ' ~ Disinterest + Normal + High_Interest'
	elif(voice_feature == 'Arousal-Valence'):
		f += ' ~ Arousal + Valence'
	else:
		print('Enter valid voice feature: Emotion, Affect, LOI, Arousal-Valence!')
	mdl = smf.MNLogit.from_formula(f,d)
	mdl_fit = mdl.fit()
	return mdl_fit
	
#data = panda dataframe
#labels = labels/ table column names from dataframe
# returns list of F and p values for each attribute/ column in dataframe calculated by f_oneway from scipy.stats
def f_anova(data, labels, char_feature):
	res_F = []
	res_p = []
	for feat in labels:
		if(char_feature == 'Sex'):
			data.Sex.replace({0.0: "Male", 1.0: "Female"}, inplace = True)
			group1 = data.loc[data['Sex'] == 'Male']
			group2 = data.loc[data['Sex'] == 'Female']
			F, p = st.f_oneway(group1[feat], group2[feat])
		#elif(char_feature == 'Academical'):
			#data.Academic_Status.replace()
		elif(char_feature == 'Age'):
			data.Age.replace({23: "Young", 24: "Intermediate", 25: "Old"}, inplace = True)
			group1 = data.loc[data['Age'] == 'Young']
			group2 = data.loc[data['Age'] == 'Intermediate']
			group3 = data.loc[data['Age'] == 'Old']			
			F, p = st.f_oneway(group1[feat], group2[feat], group3[feat])
		else:
			print('Invalid Argument: Please enter either Sex, Academical or Age as char_feature!')
		res_F.append(F)
		res_p.append(p)
	return [res_F, res_p]