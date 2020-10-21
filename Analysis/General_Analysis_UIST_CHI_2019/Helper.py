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
		 res = pd.DataFrame({'Grad Student': r[0], 'PhD': r[1]}, index = label)
	elif(type == 'Age'):
		res = pd.DataFrame({'Young': r[0], 'Intermediate': r[1], 'Old': r[2]}, index = label)
	elif(type == 'Sex'):
		res = pd.DataFrame({'Male': r[0], 'Female': r[1]})
	elif(type == 'IsNativeSpeaker'):
		res = pd.DataFrame({'Asian Non-Native': r[0], 'Europ. Non-Native': r[1], 'Native Speaker': r[2]}, index = label)
	else:
		res = 0
	return res		

def dropCharacterCols(data):
	res = data.drop(['Char_ID','ID','Filename', 'Sex', 'Academic Status', 'VideoTitle', 'Name', 'IsNativeSpeaker'], axis = 1)
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
	res = res.drop(['ID','Char_ID','Filename',  'Sex', 'Academic Status'], axis = 1)
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
	
def chi2(data, labels, char_feature, shouldPrint = False):
	tables = calcFrequencyTable(data, labels, char_feature)
	chi2s = []
	residuals = []
	for i in range(0,len(tables)):
		# table += 5 has to be removed for final analysis, however, leaving this line of code out would result in an error bc there are 0 elements in the table
		#tables[i] += 5
		chi2 = st.chi2_contingency(tables[i])
		if(shouldPrint == True):
			print('Chi square of ' + labels[i] + ' : ' + str(chi2[0]) + ' with p-value of: ' + str(chi2[1]))
		table = sm.stats.Table(tables[i])
		residual = table.standardized_resids
		chi2s.append(chi2)
		residuals.append(residual)
	return [chi2s, residuals]

def chi2_post_hoc(data, labels, char_feature, method, shouldPrint= False, calculateResiduals = False):
	fre_tables = calcFrequencyTable(data, labels, char_feature)
	#Since the index (=Character_feature) will be the same for every voice_feature, we get all combinations before iterating over all voice_features
	all_combis = list(it.combinations(fre_tables[0].index,2))
	for i in range(0,len(fre_tables)):
		p_vals = []
		res = []
		for comb in all_combis:
			#Create new data frame from combinations to conduct chi2 independence test
			new_df = fre_tables[i][(fre_tables[i].index == comb[0]) | (fre_tables[i].index == comb[1])]
			#Calculate residuals if needed
			if(calculateResiduals == True):
				new_table = sm.stats.Table(new_df)
				new_res = new_table.standardized_resids
				
			
			# table += 5 has to be removed for final analysis, however, leaving this line of code out would result in an error bc there are 0 elements in the table
			new_df += 5
			chi2_ph = st.chi2_contingency(new_df, correction = True)
			p_vals.append(chi2_ph[1])
			if(calculateResiduals == True):
				res.append(new_res)
		reject_list, corrected_p_vals = multipletests(p_vals, method = method)[:2]
		if(shouldPrint == True):
			print(labels[i])
			print('Combinations: ' + str(all_combis))
			print('Reject List: ' + str(reject_list))
			print('Corrected p-values: ' + str(corrected_p_vals))
		if(calculateResiduals == False):
			return [reject_list, corrected_p_vals, all_combis]
	
	else:
		return [reject_list, corrected_p_vals, all_combis, res ]
		
def displayANOVA(anova_res, label, type, char_type):
	print('ANOVA test for ' + char_type + ' and ' + type + ': ')
	print(label)
	print(anova_res)
	print('\n')
	return
	
def binData(data):
	count = np.zeros(4)
	for i in data:
		if(i <= 0.25):
			count[0] +=1
		elif(i <= 0.5 and i > 0.25):
			count[1]+=1
		elif(i <= 0.75 and i > 0.5):
			count[2]+=1
		elif(i <= 1.0 and i > 0.75):
			count[3]+=1			
	return count

# Converts a panda data frame containing decimal values to frequency tables 
def calcFrequencyTable(data, labels, char_feature):
	tables = []
	
	for l in labels:
		if(char_feature == 'Sex'):
			df_tab = pd.DataFrame(columns = ['1st Quartile', '2nd Quartile', '3rd Quartile', '4th Quartile'], index = ['Male', 'Female'] )
			group1 = data.loc[data['Sex'] == 'Male'][l]
			group2 = data.loc[data['Sex'] == 'Female'][l]
			row1 = binData(group1)
			row2 = binData(group2)
			df_tab.loc['Male'] = row1
			df_tab.loc['Female'] = row2

		elif(char_feature == 'Academic'):
			df_tab = pd.DataFrame(columns = ['1st Quartile', '2nd Quartile', '3rd Quartile', '4th Quartile'], index = ['Grad Student', 'PhD'] )
			group1 = data.loc[data['Academic Status'] == 'Grad Student'][l]
			group2 = data.loc[data['Academic Status'] == 'PhD'][l]
			row1 = binData(group1)
			row2 = binData(group2)
			df_tab.loc['Grad Student'] = row1
			df_tab.loc['PhD'] = row2
		elif(char_feature == 'IsNativeSpeaker'):
			df_tab = pd.DataFrame(columns = ['1st Quartile', '2nd Quartile', '3rd Quartile', '4th Quartile'], index = ['Asian Non-Native', 'Europ. Non-Native','Native Speaker'] )
			group1 = data.loc[data['IsNativeSpeaker'] == 'Asian Non-Native'][l]
			group2 = data.loc[data['IsNativeSpeaker'] == 'Europ. Non-Native'][l]
			group3 = data.loc[data['IsNativeSpeaker'] == 'Native Speaker'][l]
			row1 = binData(group1)
			row2 = binData(group2)
			row3 = binData(group3)
			df_tab.loc['Asian Non-Native'] = row1
			df_tab.loc['Europ. Non-Native'] = row2
			df_tab.loc['Native Speaker'] = row3

		else:
			print('Either use Sex, Academic or Age as KeyWord Arguments for char_feature!')
			return
		tables.append(df_tab)
	return tables
	
def logReg(data, voice_feature, char_feature, specificVoiceF):
	d = data[[char_feature, specificVoiceF]]
	f = char_feature + ' ~ ' + specificVoiceF
	model = smf.logit(formula = f, data = d)
	res = model.fit()
	return res

	
def multiLogReg(data, voice_feature, char_feature, prohibitWarning = False):
	if(char_feature == 'Sex'):
		d = data.drop(['Char_ID','ID','Name','VideoTitle', 'VideoID','Filename',  'Academic'], axis = 1)
		d.Sex.replace({'Male': 0.0, 'Female':1.0}, inplace = True)
	elif(char_feature == 'Academic'):
		d = data.drop(['Char_ID', 'ID', 'Filename', 'Name', 'VideoID','VideoTitle', 'Sex'], axis = 1)
		d.Academic.replace({'Grad Student': 0.0, 'PhD': 1.0}, inplace = True)
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
		d = data.drop(['Char_ID', 'ID', 'Filename', 'VideoID','Name', 'VideoTitle', 'IsNativeSpeaker', 'Sex'], axis = 1)
	elif(char_feature == 'IsNativeSpeaker'):
		d = data.drop(['Char_ID', 'ID', 'Filename','VideoID', 'Name', 'VideoTitle','Sex', 'Academic'], axis = 1)
	else:
		print('Either use Sex, Academic or Age as input for character feature')
		d = 0
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
			group1 = data.loc[data['Sex'] == 'Male']
			group2 = data.loc[data['Sex'] == 'Female']
			F, p = st.f_oneway(group1[feat], group2[feat])
		elif(char_feature == 'Academical'):
			data.Academic_Status.replace()
		else:
			print('Invalid Argument: Please enter either Sex, Academical or Age as char_feature!')
			return
		res_F.append(F)
		res_p.append(p)
	return [res_F, res_p]