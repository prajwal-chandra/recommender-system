import json
import re
from pyspark import SparkContext
import numpy as np
from scipy import stats	
import sys
from pprint import pprint			##Using this only for printing the output neatly

def extract_info(review):
	flag = 0.0
	if ('verified' in review.keys() and review['verified']):
		flag = 1.0
	if 'reviewText' in review.keys():
		all_strings = review['reviewText'].split()
		words = []
		for string in all_strings:
			s = string.lower()
			if re.match(r'((?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))', s):
				words.append(s)
		return (review['overall'],flag,words)

def relative_freq(review):
	total_no_words = len(review)
	common_words = cmwords.value
	result = []
	if total_no_words!=0:
		for each in common_words:
			result.append((each,review.count(each)/total_no_words))
	return result

def correlation(data,v_flag):
	rf,r,v = zip(*data)
	rel_freq = np.array(rf)
	rating = np.array(r)
	verified = np.array(v)

	rel_freq -= np.mean(rel_freq)
	rating -= np.mean(rating)
	verified -= np.mean(verified)
	
	rel_freq /= np.std(rel_freq)
	verified /= np.std(verified)
	Y_T = rating / np.std(rating)

	X_T = np.vstack((np.ones(len(rel_freq)),rel_freq))
	if v_flag:
		X_T = np.vstack((X_T,verified))

	X = np.transpose(X_T)
	Y = np.transpose(Y_T)
	
	temp1 = np.matmul(X_T,X)
	temp2 = np.matmul(X_T,Y)
	betas = np.matmul(np.linalg.inv(temp1), temp2)
	
	Y_pred = np.matmul(X,betas)
	RSS = np.sum((Y - Y_pred)**2)
	deg_free = len(rel_freq) - len(betas)
	denom1 = RSS/deg_free
	denom2 = np.sum(X[:,1]**2)
	denom = np.sqrt((denom1/denom2))
	t_stat = betas[1]/denom
	
	plt_beta = stats.t.cdf(t_stat,df = deg_free)
	if plt_beta<0.5:
		pvalue = plt_beta*1000		## Didn't multiply by 2 here because once it is decided that whether the correlation is positive or
	else:						## negative, it becomes a one tailed test instead of two tailed test.
		pvalue = (1-plt_beta)*1000

	return (betas[1],pvalue)

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Incorrect number of arguments", file=sys.stderr)
		sys.exit(-1)
	
	input_file = sys.argv[1]
	sc = SparkContext("local","Hypothesis_Testing")
	
	rdd1 = sc.textFile(input_file)
	rdd2 = rdd1.map(lambda x: json.loads(x))
	rdd3 = rdd2.map(lambda x: extract_info(x))		# Fetchig rating,verfied,reviewtext fields from json
	rdd4 = rdd3.filter(lambda x: x!=None)			# Removing entries which don't have a user review
	rdd5 = rdd4.flatMap(lambda x: [(each,1) for each in x[2]])
	rdd6 = rdd5.reduceByKey(lambda a,b: a+b)
	rdd7 = rdd6.sortBy(lambda x: x[1],False)		# Sorting to get 1000 most common words
	
	common_words = rdd7.map(lambda x: x[0]).take(1000)
	cmwords = sc.broadcast(common_words)
	
	rdd8 = rdd4.map(lambda x: (x[0],x[1],relative_freq(x[2])))			# Finding relative frequency for top 1000 common words in all the reviews
	rdd9 = rdd8.filter(lambda x: (x[2]))
	rdd10 = rdd9.flatMap(lambda x: [(each[0],(each[1],x[0],x[1])) for each in x[2]])
	rdd11 = rdd10.groupByKey().map(lambda x: (x[0],list(x[1])))			# Grouping with word as key and RF,Rating,verified as value
	
	betas_without_control = rdd11.map(lambda x: (x[0],correlation(x[1],False)))	
	betas_with_control = rdd11.map(lambda x: (x[0],correlation(x[1],True)))

	top20_pos_no_control = betas_without_control.takeOrdered(20,key = lambda x: -x[1][0])
	top20_neg_no_control = betas_without_control.takeOrdered(20, key = lambda x: x[1][0])
	
	top20_pos_control = betas_with_control.takeOrdered(20,key = lambda x: -x[1][0])
	top20_neg_control = betas_with_control.takeOrdered(20, key = lambda x: x[1][0])

	print("\nTop 20 positively correlated words with beta value and corrected p-values")
	pprint(top20_pos_no_control)
	print("\nTop 20 negatively correlated words with beta value and corrected p-values")
	pprint(top20_neg_no_control)
	print("\nTop 20 positively correlated words with beta value and corrected p-values after controlling on verified")
	pprint(top20_pos_control)
	print("\nTop 20 negatively correlated words with beta value and corrected p-values after controlling on verified")
	pprint(top20_neg_control)
	
