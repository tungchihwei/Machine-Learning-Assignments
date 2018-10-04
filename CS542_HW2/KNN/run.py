#!/usr/bin/env python
import sys
import csv
from math import sqrt
from operator import itemgetter

def distance_calculate(test_point, train_point, col):
	d = 0
	for i in range(len(test_point)-1):
		if i in col:
			if test_point[i] != train_point[i]:
				d += 1
		else:
			test_tmp = str(test_point[i])
			test = float(test_tmp)

			train_tmp = str(train_point[i])
			train = float(train_tmp)

			d += (test-train)**2

	distance = sqrt(d)

	return distance

k = int(sys.argv[1])
train_file = sys.argv[2]
test_file = sys.argv[3]

train_data = []
with open(train_file, 'r') as file:
	r_file = csv.reader(file, delimiter=',', quotechar='|')
	for i in r_file:
		train_data.append(i)

col_num = []
if "lenses" in train_file:
	col_num = [0, 1, 2, 3, 4]
elif "crx" in train_file:
	col_num = [0, 3, 4, 5, 6, 8, 9, 11, 12, 15]

test_data = []
with open(test_file, 'r') as file:
	r_file = csv.reader(file, delimiter=',', quotechar='|')
	for i in r_file:
		test_data.append(i)

result = []
for test in test_data:
	test_dist = []
	label = []

	for train in train_data:
		dist = distance_calculate(test, train, col_num)
		#print(dist)

		if len(test_dist) != k:
			test_dist.append(dist)
			label.append(train[len(train)-1])
		elif dist < max(test_dist):
			test_dist[test_dist.index(max(test_dist))] = dist
			label[test_dist.index(max(test_dist))] = train[len(train)-1]

	dic = {}
	for l in label:
	    dic[l] = dic.get(l, 0)+1
	# print(dic)
	#print(max(dic.items(), key=itemgetter(1)))
	com = max(dic.items(), key=itemgetter(1))
	#com[0] -> label  com[1] -> num
	test_tmp = test
	test_tmp.append(com[0])
	# print(com[0])
	result.append(test_tmp)

# print(result)

result_file = test_file + '.knnresult'
with open(result_file, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(result)

# http://www.math.le.ac.uk/people/ag153/homepage/KNN/OliverKNN_Talk.pdf




















