import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import random
from matplotlib.widgets import Button
from matplotlib.text import Text
import pulp
import math
import time
import networkx
from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
import itertools
import csv

loc_x = []
loc_y = []
n = 50
c = 50
loc_size = 1000
marker_size = 5
trucks = 2
solver = "gurobi"
li = [0] * n
warehouses = 1
lines_x = []
lines_y = []
obj_value = ""
exec_value = ""
model = None
x = None

min_loc_x = None
min_loc_y = None
max_loc_x = None
max_loc_y = None

distances = [[0 for x in range(n+warehouses)] for y in range(n+warehouses)]
distances2 = {}

def generate_data(n,whs):
	#global loc_x,loc_y,
	global obj_value,exec_value,distances,distances2
	global min_loc_x,min_loc_y,max_loc_x,max_loc_y
	distances = [[0 for x in range(n+whs)] for y in range(n+whs)]
	distances2 = {}

	
	with open('distances_1wh.csv') as csv_file: 
		csv_reader = csv.reader(csv_file, delimiter=',')
		for idx,row in enumerate(csv_reader):
			for i in range(len(row)):
				distances2[idx,i] = float(row[i])
				distances[idx][i] = float(row[i])

	with open('coords_1wh.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			lng = float(row[0])
			lat = float(row[1])
			loc_x.append(lng)
			loc_y.append(lat)

	min_loc_x = min(loc_x)
	min_loc_y = min(loc_y)
	max_loc_x = max(loc_x)
	max_loc_y = max(loc_y)
	for i in range(len(loc_x)):
		loc_x[i] = (loc_x[i] - min_loc_x) / (max_loc_x - min_loc_x) * 1000

	for i in range(len(loc_y)):
		loc_y[i] = (loc_y[i] - min_loc_y) / (max_loc_y - min_loc_y) * 1000
		print('lng: ' + str(loc_x[i]) + ' lat: ' + str(loc_y[i]))
	
	obj_value = ""
	exec_value = ""

def plot_data(ax, loc_x, loc_y, warehouses):
	l, = ax.plot(loc_x[warehouses:], loc_y[warehouses:], linestyle='None', marker='o',
markersize=marker_size, antialiased=True, markeredgecolor='#F5A623',
markerfacecolor='#F8E71C',markeredgewidth=2)
	l2, = ax.plot(loc_x[0:warehouses], loc_y[0:warehouses], linestyle='None', marker='s',
markersize=marker_size+2, antialiased=True, markeredgecolor='#7ED321',
markerfacecolor='#B8E986',markeredgewidth=3)


fig, ax = plt.subplots()

generate_data(n,warehouses)
plot_data(ax, loc_x, loc_y, warehouses)

plt.axis([0, 1000, 0, 1000])
plt.subplots_adjust(left=0.3)

# Radio Buttons - Li
rax = plt.axes([0.05, 0.875, 0.15, 0.1])
radioli = RadioButtons(rax, ('Hide Li', 'Show Li'))

def lifunc(label):
	global li, loc_x, loc_y, n, warehouses,fig,ax, obj_value, exec_value

	if label=="Show Li":
		for idx, val in enumerate(li):
			x = loc_x[idx+warehouses]
			y = loc_y[idx+warehouses]

# 			ax.text(x+15, y-15, "Li=" + str(int(val)), family="sans-serif",
#				horizontalalignment='left', verticalalignment='top') 
			ax.text(x+15, y-15, "i=" + str(int(val)), family="sans-serif",
				horizontalalignment='left', verticalalignment='top')
		fig.canvas.draw()
	if label=="Hide Li":
		ax.clear()
		ax.set_xlim(0,1000)
		ax.set_ylim(0,1000)

	ax.plot(lines_x, lines_y, color='#B8E986', markeredgewidth=50, markersize=50,
		linewidth=5)
	ax.plot(lines_x, lines_y, color='#7ED321')


	ax.text(350, 1075, obj_value, family="serif", horizontalalignment='right',
		verticalalignment='top')
	ax.text(450, 1075, exec_value, family="serif", horizontalalignment='left',
		verticalalignment='top')

	plot_data(ax, loc_x, loc_y, warehouses)
	fig.canvas.draw()


radioli.on_clicked(lifunc)
# Radio Buttons - n
rax = plt.axes([0.05, 0.7, 0.15, 0.175])
radio = RadioButtons(rax, ('n=10', 'n=25', 'n=50', 'n=75', 'n=100', 'n=150'))

def nfunc(label):
	global loc_x, loc_y, n, warehouses, ax, li
	s = label.split('=')
	n = int(s[1])
	generate_data(n,warehouses)
	ax.clear()
	ax.set_xlim(0,1000)
	ax.set_ylim(0,1000)
	plot_data(ax, loc_x, loc_y, warehouses)
	li = []
	#l, = ax.plot(loc_x[1:], loc_y[1:], linestyle='None', marker='o', markersize=marker_size,antialiased=True, markeredgecolor='#F5A623', markerfacecolor='#F8E71C',markeredgewidth=2)
	#l2, = ax.plot(loc_x[0], loc_y[0], linestyle='None', marker='s', markersize=marker_size+2,antialiased=True, markeredgecolor='#7ED321', markerfacecolor='#B8E986',markeredgewidth=3)
	plt.draw()
radio.on_clicked(nfunc)


# Radio Buttons - warehouses
rax = plt.axes([0.05, 0.575, 0.15, 0.125])
radiowh = RadioButtons(rax, ('1 WHS', '2 WHS', '3 WHS'))

def whfunc(label):
	global warehouses, n, ax, loc_x, loc_y, fig
	s = label.split(" ")
	warehouses = int(s[0])
	ax.clear()
	ax.set_xlim(0,1000)
	ax.set_ylim(0,1000)
	generate_data(n,warehouses)
	plot_data(ax, loc_x, loc_y, warehouses)
	fig.canvas.draw()
radiowh.on_clicked(whfunc)

# Radio Buttons - trucks
rax = plt.axes([0.05, 0.4, 0.15, 0.175])
radio2 = RadioButtons(rax, ('1 Truck', '2 Trucks', '3 Trucks', '4 Trucks', '5 Trucks'))

def trucksfunc(label):
	global trucks
	s = label.split(" ")
	trucks = int(s[0])
	radio2.on_clicked(trucksfunc)

# Radio Buttons - truck capacity
rax = plt.axes([0.05, 0.15, 0.15, 0.25])
radio4 = RadioButtons(rax, ('c=5', 'c=10', 'c=15', 'c=20', 'c=25', 'c=50', 'c=75', 'c=100'))

def cfunc(label):
	global c
	s = label.split('=')
	c = int(s[1])
radio4.on_clicked(cfunc)

# Radio Buttons - solver
rax = plt.axes([0.05, 0.0, 0.15, 0.15])
radio3 = RadioButtons(rax, ('CBC', 'GLPK', 'Held-Karp', 'Cut Plane'))

def solverfunc(label):
	global solver
	solver = label.lower()
radio3.on_clicked(solverfunc)

# Plot connecting lines
def plot_data_lines(lines_x,lines_y):
	ax.plot(lines_x, lines_y, color='#B8E986', markeredgewidth=50, markersize=50, linewidth=5)
	ax.plot(lines_x, lines_y, color='#7ED321')

def addcut(cut_edges,model,x,warehouses):
	invalid = False #ADDED
	G = networkx.Graph()
	G.add_edges_from(cut_edges)
	Components = list(networkx.connected_components(G))
	if len(Components) == 1:
		return False

	for S in Components:
		invalid_path = True
		for warehouse in range(0,warehouses):
			if warehouse in S:
				invalid_path = False
				break		
		if invalid_path:
			invalid=True
			break
	if not invalid:
		return False

	model.freeTransform()

	for S in Components:
		invalid_path = True
		for warehouse in range(0,warehouses):
			if warehouse in S:
				invalid_path = False
				break
		if invalid_path:
			model.addCons(quicksum(x[i,j] for i in S for j in S if j!=i) <= len(S)-1)
			print("cut: len(%s) <= %s" % (S,len(S)-1))
	return True

def held_karp(dists):
	"""
	Implementation of Held-Karp, an algorithm that solves the Traveling
	Salesman Problem using dynamic programming with memoization.
	Parameters:
	dists: distance matrix
	Returns:
	A tuple, (cost, path).
	"""
	n = len(dists)

	C = {}

	for k in range(1, n):
		C[(1 << k, k)] = (dists[0][k], 0)

	for subset_size in range(2, n):
		for subset in itertools.combinations(range(1, n), subset_size):
		# Set bits for all nodes in this subset
		bits = 0
		for bit in subset:
			bits |= 1 << bit

		# Find the lowest cost to get to this subset
		for k in subset:
			prev = bits & ~(1 << k)

			res = []
			for m in subset:
				if m == 0 or m == k:
					continue
				res.append((C[(prev, m)][0] + dists[m][k], m))
			C[(bits, k)] = min(res)

	bits = (2**n - 1) - 1

	res = []
	for k in range(1, n):
		res.append((C[(bits, k)][0] + dists[k][0], k))
	opt, parent = min(res)

	path = []
	for i in range(n - 1):
		path.append(parent)
		new_bits = bits & ~(1 << parent)
		_, parent = C[(bits, parent)]
		bits = new_bits
	path.append(0)
	return opt, list(reversed(path))


def solve(event):
	global n, trucks, solver, li, ax, warehouses, loc_x, loc_y, lines_x, lines_y, obj_value, exec_value
	global distances, distances2
	print("solving...")

	ax.clear()
	ax.set_xlim(0,1000)
	ax.set_ylim(0,1000)
	plot_data(ax, loc_x, loc_y, warehouses)
	ax.text(400, 1075, "solving...", family="serif", horizontalalignment='left', verticalalignment='top')
	plt.draw()
	fig.canvas.draw()

	start_time = time.time()

	# Calculate constraint for Li
	total_l = 0
	total_c = 0.0
	paths = 0
	slack = False
	while total_c < n or paths < trucks:
		c1 = min(c,n)
		total_c += c1
		total_l += c1*(c1+1)/2
		paths += 1

	if total_c > n:
		slack = True

	if solver=="cbc" or solver=="glpk" or solver=="gurobi":
		# Objective function
		z = pulp.LpProblem('Test', pulp.LpMinimize)

		# Generate decision variables
		x = {}
		y = {}
		variables = []
		l = {}
		s = {}
		for i in range(n+warehouses):
			for j in range(n+warehouses):
				if i==j:
					continue
				x[i,j] = pulp.LpVariable('x_' + str(i) + '_' + str(j), 0, 1, pulp.LpInteger)
			if i >= warehouses:
				l[i] = pulp.LpVariable('l_' + str(i), 1, min(n,c), pulp.LpInteger)

		# Objective function
		z += pulp.lpSum([distances[i][j] * x[i,j] for i in range(n+warehouses) for j in 
			list(range(i)) + list(range(i+1,n+warehouses))])

		# Constraints
		constraintSeq = []
		constraintTrucks = []
		for i in range(n+warehouses):
			if i>=warehouses:
				constraintSeq.append(l[i])
			constraintFrom = []

			constraintTo = []
			for j in range(n+warehouses):
				if i==j:
					continue
				if i>=warehouses and j>=warehouses:
					z += pulp.lpSum([l[i], -1*l[j], n*x[i,j], -n+1]) <= 0

				if i>=warehouses:
					constraintFrom.append(x[i,j])
					constraintTo.append(x[j,i])
				if i<warehouses:
					constraintTrucks.append(x[j,i])
				if i>=warehouses:
					z += pulp.lpSum(constraintFrom) == 1 # paths from location
					z += pulp.lpSum(constraintTo) == 1 # paths to location
				if i==warehouses and (paths > 1 or warehouses>1):
					z += pulp.lpSum(constraintTrucks) == paths # paths to warehouse

		if not slack:
		z += pulp.lpSum(constraintSeq) == total_l
		else:
		z += pulp.lpSum(constraintSeq) <= total_l

		# Solve
		if solver=="cbc":
			status = z.solve()
		if solver=="glpk":
			status = z.solve(pulp.GLPK())		
		if solver=="gurobi":
			status = z.solve(pulp.GUROBI_CMD())

		# should be 'Optimal'
		if pulp.LpStatus[status]!="Optimal":
			print("RESULT: ".pulp.LpStatus[status])
		print("Objective function value: "+str(z.objective.value()))

		# Print variables & save path
		lines_x = []
		lines_y = []
		li = [0] * n
		for i in range(n+warehouses):
			if i>=warehouses:
				li[i-warehouses] = pulp.value(l[i])
			for j in range(n+warehouses):
				if i==j:
					continue
				if pulp.value(x[i,j]) == 1:
					lines_x.append(loc_x[i])
					lines_x.append(loc_x[j])
					lines_y.append(loc_y[i])
					lines_y.append(loc_y[j])
					lines_x.append(np.nan)
					lines_y.append(np.nan)

		obj_value = "c=" + str(round(z.objective.value(),2))

	elif solver=="cut plane":
		model = Model("tsp")
		model.hideOutput()
		x = {}
		l = {}
		for i in range(n+warehouses):
			for j in range(n+warehouses):
				if i != j:
					x[i,j] = model.addVar(ub=1, name="x(%s,%s)"%(i,j))
			if (paths > 1 or warehouses > 1) and i >= warehouses:
				l[i] = model.addVar(ub=min(c,n),lb=1, name="l(%s)"%(i))

		if paths == 1 and warehouses == 1:

			# SYMMETRIC DISTANCE MATRIX ONLY
			#for i in range(n+warehouses):
			#model.addCons(quicksum(x[j,i] for j in range(n+warehouses) if j != i) + \
			# quicksum(x[i,j] for j in range(n+warehouses) if j != i) == 2,"Degree(%s)"%i)

			# ASYMMETRIC DISTANCE MATRIX
			for i in range(n+warehouses):
				model.addCons(quicksum(x[j,i] for j in range(n+warehouses) if j != i) == 1,"In(%s)"%i)
				model.addCons(quicksum(x[i,j] for j in range(n+warehouses) if j != i) == 1,"Out(%s)"%i)
		
		else:
			for i in range(warehouses, n+warehouses):
				model.addCons(quicksum(x[j,i] for j in range(n+warehouses) if j != i) == 1,"In(%s)"%i)
				model.addCons(quicksum(x[i,j] for j in range(n+warehouses) if j != i) == 1,"Out(%s)"%i)

			for i in range(warehouses, n+warehouses):
				for j in range(warehouses, n+warehouses):
					if i!=j:
						model.addCons(l[i] -l[j] +n*x[i,j] <= n-1, "Li(%s,%s)"%(i,j))
			model.addCons(quicksum(x[j,i] for i in range(warehouses) for j in range(n+warehouses) if i!=j) == paths, "Paths(%s)"%paths)

			if not slack:
				model.addCons(quicksum(l[i] for i in range(warehouses,n+warehouses)) == total_l,"TotalL")
			
			else:
				model.addCons(quicksum(l[i] for i in range(warehouses,n+warehouses)) <= total_l, "TotalL")
		
		model.setObjective(quicksum(distances2[i,j]*x[i,j] for (i,j) in x), "minimize")

		EPS = 1.e-6
		isMIP = False
		model.setPresolve(SCIP_PARAMSETTING.OFF)

		while True:
			model.optimize()
			#edges = []
			lines_x = []
			lines_y = []
			edges = []
			li = [0] * n
			for (i,j) in x:
				# i=j already skipped
				if model.getVal(x[i,j]) > EPS:
					#edges.append( (i,j) )
					lines_x.append(loc_x[i])
					lines_x.append(loc_x[j])
					lines_y.append(loc_y[i])
					lines_y.append(loc_y[j])
					lines_x.append(np.nan)
					lines_y.append(np.nan)
					edges.append( (i,j) )
			if paths>1 or warehouses>1:
				for i in range(warehouses, n+warehouses):
					li[i-warehouses] = int(model.getVal(l[i]))

			obj_value = "c=" + str(round(model.getObjVal(),2))

			ax.clear()
			ax.set_xlim(0,1000)
			ax.set_ylim(0,1000)

			plot_data_lines(lines_x,lines_y)
			plot_data(ax, loc_x, loc_y, warehouses)
			ax.text(400, 1075, "solving...", family="serif", horizontalalignment='left', verticalalignment='top')

			fig.canvas.draw()

			if addcut(edges,model,x,warehouses) == False:
				if isMIP: # integer variables, components connected: solution found
					break
				model.freeTransform()
				for (i,j) in x: # all components connected, switch to integer model
					model.chgVarType(x[i,j], "B")
				if paths > 1 or warehouses > 1:
					for i in range(warehouses,n+warehouses):
						model.chgVarType(l[i], "I")
				isMIP = True

		sol_li = [0] * (n+warehouses)
		sol_xij = {}

		print('solved.')
	elif solver == 'held-karp':
		li = [0] * n
		opt, path = held_karp(distances)
		print(path)
		obj_value = "c=" + str(round(opt,2))
		x = [[0 for x in range(n+warehouses)] for y in range(n+warehouses)]
		for idx, val in enumerate(path):
			if idx < (len(path)-1):
				x[val][path[idx+1]] = 1;
			elif idx == (len(path)-1):
				x[val][path[0]] = 1;

		for i in range(n+warehouses):
			for j in range(n+warehouses):
				if x[i][j] == 1:
					#edges.append( (i,j) )
					lines_x.append(loc_x[i])
					lines_x.append(loc_x[j])
					lines_y.append(loc_y[i])
					lines_y.append(loc_y[j])
					lines_x.append(np.nan)
					lines_y.append(np.nan)
					#edges.append( (i,j) )

	# Print computation time
	time2 = time.time() - start_time
	exec_value = time2
	units = 'secs'
	if time2 > 60:
		time2 /= 60
		units = 'mins'
	if time2 > 60:
		time2 /= 60
		units = 'hours'

	time2 = round(time2,2)

	exec_value = "exec=" + str(time2) + " " + units

	print("--- " + str(time2) + " " + units + " ---")

	# Redraw points
	ax.clear()
	ax.set_xlim(0,1000)
	ax.set_ylim(0,1000)

	plot_data_lines(lines_x,lines_y)
	plot_data(ax, loc_x, loc_y, warehouses)

	ax.text(350, 1075, obj_value, family="serif", horizontalalignment='right', verticalalignment='top')
	ax.text(450, 1075, exec_value, family="serif", horizontalalignment='left', verticalalignment='top')
	fig.canvas.draw()
	
axsolve = plt.axes([0.87, 0.905, 0.1, 0.075])
bsolve = Button(axsolve, 'Solve')
bsolve.on_clicked(solve)
plt.show()