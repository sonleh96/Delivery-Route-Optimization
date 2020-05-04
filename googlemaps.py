import json
import requests
import csv
import numpy

coords = numpy.empty([52, 2])

with open('coords.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for idx, row in enumerate(csv_reader):
		coords[idx,:] = row

indexes = numpy.empty([7, 2])
indexes[0,:] = [0,9]
indexes[1,:] = [10,19]
indexes[2,:] = [20,29]
indexes[3,:] = [30,39]
indexes[4,:] = [40,49]
indexes[5,:] = [50,51]
indexes[6,:] = [52,55]

dd = numpy.empty([52,52])
dt = numpy.empty([52,52])

for s in range(6):
	for d in range(6):
		s_s = None
		s_l = coords[int(indexes[s,0]):int(indexes[s+1,0]),:]
		d_s = None
		d_l = coords[int(indexes[d,0]):int(indexes[d+1,0]),:]

		s_s = '|'.join(','.join(str(x) for x in y) for y in s_l)
		d_s = '|'.join(','.join(str(x) for x in y) for y in d_l)

	url = "https://maps.googleapis.com/maps/api/distancematrix/json?origins=" + s_s + "&destinations=" + d_s + "&key=API_KEY"
	print("Calling Url: " + url)
	response = requests.get(url)
	json_data = json.loads(response.text)
	distances = numpy.empty([int(indexes[s+1,0]-indexes[s,0]), int(indexes[d+1,0]-indexes[d,0])])

	times = numpy.empty([int(indexes[s+1,0]-indexes[s,0]), int(indexes[d+1,0]-indexes[d,0])])
	rows = json_data['rows']
	for i in range(len(rows)):
		row = rows[i]
		elements = row['elements']
		for j in range(len(elements)):
			element = elements[j]
			if element['status'] != 'OK':
				print('Error: i: ' + str(i) + ' j: ' + str(j))
				distances[i,j] = element['distance']['value']
				times[i,j] = element['duration']['value']
	dd[int(indexes[s,0]):int(indexes[s+1,0]), int(indexes[d,0]):int(indexes[d+1,0])] = distances
	dt[int(indexes[s,0]):int(indexes[s+1,0]), int(indexes[d,0]):int(indexes[d+1,0])] = times
	
numpy.savetxt("d_dist.csv", dd, delimiter=",")
numpy.savetxt("d_time.csv", dt, delimiter=",")