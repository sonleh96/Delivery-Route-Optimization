import json
import requests
import csv
import numpy

coords = numpy.empty([52, 2])

with open('coords.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for idx, row in enumerate(csv_reader):
		coords[idx,:] = row

indexes = numpy.empty([6, 2])
indexes[0,:] = [0,11]
indexes[1,:] = [12,23]
indexes[2,:] = [24,35]
indexes[3,:] = [36,47]
indexes[4,:] = [48,51]
indexes[5,:] = [52,55]

dd = numpy.empty([52,52])
dt = numpy.empty([52,52])

for s in range(5):
	for d in range(5):
		min_s = indexes[s,0]
		min_d = indexes[d,0]
		if s < d:
			min_d = indexes[d,0]-12
		elif s > d:
			min_s = indexes[s,0]-12
		s_l = list(range(int(indexes[s,0]-min_s), int(indexes[s+1,0]-min_s)))
		d_l = list(range(int(indexes[d,0]-min_d), int(indexes[d+1,0]-min_d)))
		s_s = ';'.join([str(i) for i in s_l])
		d_s = ';'.join([str(i) for i in d_l])
		c_s = None
		c_l = None
		if s==d:
			c_l = coords[int(indexes[s,0]):int(indexes[s+1,0]),:]
		else:
			c_l = coords[int(indexes[s,0]):int(indexes[s+1,0]),:]
			c_l = numpy.append(c_l, coords[int(indexes[d,0]):int(indexes[d+1,0]),:], axis=0)
	c_s = ';'.join(','.join(str(x) for x in y) for y in c_l)

	url = "https://api.mapbox.com/directions-matrix/v1/mapbox/driving/" + c_s + "?destinations=" + d_s + "&sources=" + s_s + "&annotations=duration,distance&access_token=API_KEY"
	print("Calling url: " + url)
	response = requests.get(url)
	json_data = json.loads(response.text)
	if json_data['code'] != "Ok":
		print("Error calling API: " + str(s) + " d: " + str(d))
	dd[int(indexes[s,0]):int(indexes[s+1,0]), int(indexes[d,0]):int(indexes[d+1,0])] = json_data['distances']
	dt[int(indexes[s,0]):int(indexes[s+1,0]), int(indexes[d,0]):int(indexes[d+1,0])] = json_data['durations']


numpy.savetxt("d_dist.csv", dd, delimiter=",")
numpy.savetxt("d_time.csv", dt, delimiter=",")