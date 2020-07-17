import numpy as np

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