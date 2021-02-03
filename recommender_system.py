import json
import sys
from pyspark import SparkContext
import numpy as np
from scipy import stats
from pprint import pprint			##Using this only for printing the output neatly


def not_rated(users):
	all_users_list = distinct_users.value
	target_users = [each[0] for each in users]
	rest_users = set(all_users_list) - set(target_users)
	return list(rest_users)


def find_neighbours(item_profile):
	result = []
		
	item_user,item_rating = zip(*item_profile[1])
	targets = target_bc.value
	for each_t in targets:
		target = each_t[0]
		t_profile = each_t[1]
		t_user,t_rating = zip(*t_profile)
		
		common_users = list(set(item_user) & set(t_user))
		if(len(common_users)>=2 and target!=item_profile[0]):
			meancentered_ir = item_rating - np.mean(item_rating)
			meancentered_tr = t_rating - np.mean(t_rating)
			item_dict = dict(zip(item_user,meancentered_ir))
			t_dict = dict(zip(t_user,meancentered_tr))
			val = 0
			d1 = sum(meancentered_ir*meancentered_ir)
			d2 = sum(meancentered_tr*meancentered_tr)
			for user in common_users:
				val+= (item_dict[user]*t_dict[user])
			if(d1!=0 and d2!=0):
				denom1 = np.sqrt(d1)
				denom2 = np.sqrt(d2)
				similarity = val/(denom1*denom2)
				if(similarity>0):
					result.append((target,(item_profile[0],similarity)))
	return result
			

def prediction(target,profile):
	user_items,rating = zip(*profile)
	neigh_items,sim = zip(*target)
	common_items = list(set(user_items) & set(neigh_items))
	if(len(common_items)>=2):
		diction1 = dict(profile)
		diction2 = dict(target)
		rating = 0
		sum_w = 0
		for each in common_items:
			a = diction1[each]
			b = diction2[each]
			rating += a*b
			sum_w+= b
		return rating/sum_w
	else:
		return -100

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Incorrect number of arguments", file=sys.stderr)
		sys.exit(-1)
	
	input_file = sys.argv[1]
	target_list = eval(sys.argv[2])
	
	sc = SparkContext("local","Collaborative_Filtering")
	
	rdd1 = sc.textFile(input_file)
	rdd2 = rdd1.map(lambda x: json.loads(x))

	rdd3 = rdd2.map(lambda x: ( (x['reviewerID'], x['asin']), x['overall']) )		##Taking required data
	
	filter_a = rdd3.groupByKey().map(lambda x: (x[0][1],(x[0][0],list(x[1])[-1])))		## Selecting the last review in file in case of multiple entries
																		## for an item by the user
	
	filter_b = filter_a.groupByKey().map(lambda x: (x[0],list(x[1]))).filter(lambda x: len(x[1])>=25)		#Applying filter B
	
	rdd4 = filter_b.flatMap(lambda x: [(each[0],(x[0],each[1])) for each in x[1]] )				#Regrouping so that reviewerID becomes a key
	users = rdd4.groupByKey().map(lambda x: (x[0],list(x[1]))).filter(lambda x: len(x[1])>=5)		# Applying filter C
	
	distinct_users = sc.broadcast(users.keys().collect())					
	
	rdd5 = users.flatMap(lambda x: [(each[0],(x[0],each[1])) for each in x[1]])		
	all_items = rdd5.groupByKey().map(lambda x: (x[0],list(x[1])))				# Dense representation of Utility matrix
	
	target_items = all_items.filter(lambda x: x[0] in target_list)				# Getting item profile for items in target (input) list
	target_bc = sc.broadcast(target_items.collect())

	# Finding neighbours for all the target items
	
	neighbours = all_items.flatMap(lambda item: find_neighbours(item) ).groupByKey().map(lambda x: (x[0],list(x[1])))
	 
	no_rating = target_items.flatMap(lambda x: [(each,x[0]) for each in not_rated(x[1])])	# Finding users who haven't rated the target items
	
	no_rating_profile = no_rating.join(users).map(lambda x: (x[1][0],(x[0],x[1][1])))		# Getting profiles of the above users
	
	all_data = neighbours.join(no_rating_profile)							
	
	predicted_values = all_data.map(lambda x: (x[0],x[1][1][0], prediction(x[1][0],x[1][1][1])))		# Predicting unknown ratings
	final = predicted_values.filter(lambda x: x[2]!=-100)
	#pprint(final.collect())
	sc.stop()
