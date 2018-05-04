import lof
import csv
import numpy as np
    
if __name__ == "__main__":
	# Load the data into tuple
	data = np.loadtxt("click-stream event.csv", delimiter=',')
	dataTuple = tuple(map(tuple, data))

	# Print top 5 outliers which use Manhanttan distance
	print("top 5 outliers which use Manhanttan distance")
	print(lof.outliersM(2, dataTuple)[:5])
	
	# Print top 5 outliers which use Euclidean distance
	print("top 5 outliers which use Euclidean distance")
	print(lof.outliers(3, dataTuple)[:5])