import scipy.io as io
import numpy as np
import os
import glob
import imageio


# Importing the files
for filename in glob.glob('dataset/ground_truth/train/*.mat'):
	matfilePath = filename[27:-4]
	file = io.loadmat(filename)

	#Taking the edges
	edges = file['groundTruth'][0][0][0][0][1]
	edges_255 = edges * 255

	# save images
	path = 'Ground Truth/' + matfilePath + '_edges.jpg'
	imageio.imwrite(os.path.join(path), edges_255)


