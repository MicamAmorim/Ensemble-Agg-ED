from Chok import *
from confusionMatrix import *
from confusionMatrix import metrics_table


# Defining variables__________________________________________________________________________________________________

# Parameters
Tnormas = {"Min": minimo, "product": produto, "lukasiewicz": lukasiewicz, "drastic": drastic, "nilpotent": nilpotent, "hamatcher": hamatcher}
#nomes = ["Min", "product", "lukasiewicz", "drastic", "nilpotent", "hamatcher"]
nomes = ["Min"]
#method = ["canny", "entropy", "frangi", "meijering", "sato", "blur"]
method = ["canny"]
Disk_size = list(range(15,20))
num_segments = 30
r = 0
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
	

# File informations
## Import each image in the folder Images 

if __name__ == '__main__':

	for filename in glob.glob('Images/*.jpg'): #assuming jpg
		img=cv2.imread(filename)
		FileName = filename[7:-4]
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		#plt.imshow(gray)
		#plt.title(FileName)
		#plt.show(block = False)
		#plt.pause(0.1)
		#plt.close("all")

		gray = clahe.apply(gray)

		#plt.imshow(gray)
		#plt.title(FileName)
		#plt.show(block = False)
		#plt.pause(0.1)
		#plt.close("all")

		filters1 = []
		Edge_detection(gray, filters1, Disk_size,method, FileName)

		Agreggation1 = []
		Agreggation(filters1, Agreggation1, nomes, Tnormas, FileName, method)

		angle_canny = []
		Angles(Agreggation1, angle_canny, nomes)

		non_max_Canny = []
		Non_Max_supression(Agreggation1, angle_canny, non_max_Canny, nomes, FileName, method)

		final_Bin1 = []
		Histerisys(final_Bin1, non_max_Canny, nomes, FileName, method)

		cleanImage1 = []
		CleanSegments(final_Bin1, cleanImage1, nomes, num_segments, FileName, method)


	for filename in glob.glob('Ground Truth/*.jpg'): #assuming jpg
		Real = cv2.imread(filename)
		Real_name = filename[13:-10]
		print(Real_name)
		Real = convert2bin(Real)

		predito_list = []
		predito_nome = []
		for filename in glob.glob('Clean segments/'+Real_name+'*.png'): #assuming png
			Predito = cv2.imread(filename)
			predito_nome.append(filename[15:-4])
			Predito = convert2bin(Predito)
			predito_list.append(Predito)

		Filters_list = []
		Filters_nome = []
		for filename in glob.glob('Filters/'+Real_name+'*.png'): #assuming jpg
			Filters = cv2.imread(filename)
			Filters_nome.append(filename[8:-4])
			Filters = convert2bin(Filters)
			Filters_list.append(Filters)
	
		method_names = predito_nome + Filters_nome
		method_img_list = predito_list + Filters_list

		print(method_names, len(method_names))
		print(len(method_img_list))

		metrics_table(Real, r, method_names, method_img_list)













  
  
