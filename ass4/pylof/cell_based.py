import numpy as np
import math

xlen = 100
ylen = 150

# Parameter Settings
M = 2
L = 2
D = L * 2 * math.sqrt(2)

# Load data from csv
data = np.loadtxt("click-stream event.csv", delimiter=',')

# Store the number of objects in a cell
# Step 1: init the number of objects in a cell to 0
cellCount = np.zeros((xlen, ylen), dtype=int)

# Store what object(s) is/are stored in a cell
cellObjects = np.empty([xlen, ylen], dtype=list)
# Init the cellObjects as an empty list of 2d list
for i in range(len(cellObjects)):
	for j in range(len(cellObjects[i])):
		cellObjects[i][j] = [[]]

# Grid size of 2,L = 2
# Step 2: Count the number of objects in a cell
# Consider “pause_video” and “play_video” 
for d in data:
	x = int(d[1]//L)
	y = int(d[2]//L)
	cellCount[x][y] += 1
	if (not cellObjects[x][y][0]):
		cellObjects[x][y][0] = [d[1],d[2]]
	else:
		cellObjects[x][y].append([d[1],d[2]])

# Store the color of a cell, init as white
cell = np.empty((xlen, ylen), dtype=str)
for i in range(len(cell)):
	for j in range(len(cell[i])):
		cell[i][j] = "w"

# Step 3: Mark the cell as red if cellCount > M
for i in range(len(cellCount)):
	for j in range(len(cellCount[i])):
		if cellCount[i][j] > M:
			cell[i][j] = "r"


# Checking to avoid index out of range
def checkvalid(x, y):
	if (cell[x][y] == "r"):
		return False
	if (x < 0 or y < 0 or x > xlen-1 or y > ylen-1):
		return False
	return True

# Add the count of all L1 neighbour, can be improved using a loop
def addL1neighbour(x, y):
	sum = 0
	if (checkvalid(x, y-1)):
		sum += cellCount[x][y-1]
	if (checkvalid(x, y+1)):
		sum += cellCount[x][y+1]
	if (checkvalid(x+1, y)):
		sum += cellCount[x+1][y]
	if (checkvalid(x+1, y+1)):
		sum += cellCount[x+1][y+1]
	if (checkvalid(x+1, y-1)):
		sum += cellCount[x+1][y-1]
	if (checkvalid(x-1,y)):
		sum += cellCount[x-1][y]
	if (checkvalid(x-1, y+1)):
		sum += cellCount[x-1][y+1]
	if (checkvalid(x-1, y-1)):
		sum += cellCount[x-1][y-1]
	return sum

# Add the count of self, L1, L2 neighbours
def addSelfL1L2neighbour(x, y):
	sum = 0
	for i in range(x+5):
		for j in range(y+5):
			if (checkvalid(i-4,j-4)):
				sum += cellCount[i][j]
	return sum

# Return all L2 neighbours as list
def L2Objects(x, y):
	l = list()
	for i in range(x+5):
		for j in range(y+5):
			if (checkvalid(i-4,j-4) and abs(x-i) >1 and abs(y-j) > 1):
				for _object in cellObjects[i][j]:
					if (_object != []):
						l.append(_object)
	return l

# Calculate the euclidean distance
def euclideanDist(l1, l2):
	dist = math.sqrt((l1[0] - l2[0]) **2 + (l1[1] - l2[1]) **2)
	return dist

# Step 4: Mark the surrounding cell pink if it is red and the 
# surround is not red
for i in range(len(cell)):
	for j in range(len(cell[i])):
		if cell[i][j] == "r":
			if (checkvalid(i, j-1)):
				cell[i][j-1] = "p"
			if (checkvalid(i, j+1)):
				cell[i][j+1] = "p"
			if (checkvalid(i+1,j)):
				cell[i+1][j] = "p"
			if (checkvalid(i+1,j+1)):
				cell[i+1][j+1] = "p"
			if (checkvalid(i+1,j-1)):
				cell[i+1][j-1] = "p"
			if (checkvalid(i-1,j)):
				cell[i-1][j] = "p"
			if (checkvalid(i-1, j+1)):
				cell[i-1][j+1] = "p"
			if (checkvalid(i-1,j-1)):
				cell[i-1][j-1] = "p"

outlier = list()

# Step 5
for i in range(len(cell)):
	for j in range(len(cell[i])):
		if (cell[i][j] == "w" and cellCount[i][j] != 0):
			# Step 5a
			countw2 = cellCount[i][j] + addL1neighbour(i ,j)
			# Step 5b
			if countw2 > M:
				cell[i][j] = "p"
			else:
				# Re-adding self, L1 and L2 rather than L2
				# for convience of writing a simpler for loop
				# Step 5c 1.
				countw3 = addSelfL1L2neighbour(i ,j)
				# Step 5c 2.
				if (countw3 <= M):
					for obj in cellObjects[i][j]:
						outlier.append(obj)
				else:
					# Step 5c 3.
					for objectp in cellObjects[i][j]:
						# Step 5c 3i.
						countp = countw2
						# Step 5c 3ii.
						for objectq in L2Objects(i, j):
							dist = euclideanDist(objectp, objectq)
							# print(dist)
							if dist <= D:
								countp += 1
								if countp > M:
									break
						if countp > M :
							break
						# Step 5c 3iii.							
						outlier.append(objectp)

# Print out the outliers
print("No. of outliers:", len(outlier))
print(outlier)
	