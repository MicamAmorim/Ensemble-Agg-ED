
# Importing librarys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage import feature
from skimage.filters import frangi, sobel, gaussian, meijering, sato
from scipy import ndimage
import glob
import os


###################################################################################################################################
################################################ F U N C T I O N S ################################################################
###################################################################################################################################

# Funções de T-norma

def minimo(x,y):
  #print("Tnorma do minimo")
  return np.min([x, y])

def produto(x,y):
  #print("Tnorma do produto")
  return np.prod([x, y])

def lukasiewicz(x, y):
  #print("Tnorma de lukasiewicz")
  return np.max([0, x + y - 1])

def drastic(x,y):
  #print("Tnorma drastic")
  if(y == 1):
    return x
  elif(x == 1):
    return y
  else:
    return 0
def nilpotent(x,y):
  #print("Tnorma nilpotent")
  if((x+y) > 1):
    return np.min([x,y])
  else:
    return 0

def hamatcher(x,y):
  #print("Tnorma de hamatcher")
  if(x == 0 and y == 0):
    return 0
  else:
    #print(x,y)
    return (x*y)/(x+y - x*y)

#Choquet Integral

def chok(mu, Tnorma = None):

  c = 0
  X = []
  for mu_i in mu:
    X.append(mu_i)
  X.sort()
  X.insert(0, 0)

  if Tnorma != None:
    for i in range(len(X)- 1):
      M = (len(X) - i - 1)/(len(X) - 1)
      xx = X[i+1] - X[i]
      c = Tnorma(xx, M) + c
    return c

  else:
    print("Please, choose a value between 1 and 6:\n Min [1];\n  product [2];\n  lukasiewicz [3];\n  drastic [4];\n  nilpotent [5];\n  hamatcher [6]")

#Generalized Choquet

def choquet_Tconorm(matrix, Tnorma):

  depth = len(matrix)
  height = int(matrix[0][0].size)
  width =  int(matrix[0].size/matrix[0][0].size)
  print(height, width, depth)

  Agreggated = np.zeros([width, height])
  
  for i in range(0, width):
    for j in range(0, height):
      lista = []
      for k in range(len(matrix)):
        lista.append(matrix[k][i][j])

      Agreggated[i,j] = chok(lista, Tnorma)
  
  return Agreggated

def choquet_Tconorm_ContentAware(matrix, Tnorma):

  depth = len(matrix)
  height = int(matrix[0][0].size)
  width =  int(matrix[0].size/matrix[0][0].size)
  print(height, width, depth)

  Agreggated = np.zeros([width, height])
  
  for i in range(0, width):
    for j in range(0, height):
      lista = []
      for k in range(len(matrix)):
        lista.append(matrix[k][i][j])

        if i-1 < 0 or j-1 < 0 or i+1 >= width or j+1 >= height:
          continue
        else:
          lista.append(matrix[k][i+1][j])
          lista.append(matrix[k][i+1][j+1])
          lista.append(matrix[k][i+1][j-1])
          lista.append(matrix[k][i-1][j])
          lista.append(matrix[k][i-1][j+1])
          lista.append(matrix[k][i-1][j-1])
          lista.append(matrix[k][i][j-1])
          lista.append(matrix[k][i][j+1])


      Agreggated[i,j] = chok(lista, Tnorma)
  
  return Agreggated


def normaliza(data, ref):
  xmax, xmin = ref.max(), ref.min()
  return (data - xmin)/(xmax - xmin)



def Filters_calc(Z25, filters, Disk_size, method):

# Filtering calculation for every disk size
  for i in Disk_size:
    
    #INPUT NORMALIZATION
    Z25 = normaliza(Z25, Z25)

      
    
    #________________________________Canny______________________________________

    if "canny" in method:

      bor = np.float32(feature.canny(Z25, sigma = i*0.1))
      #OUTPUT NORMALIZATION
      bor = normaliza(bor, bor)

      #NaN VALUES VERIFICATION
      array_sum = np.sum(bor)
      array_has_nan = np.isnan(array_sum)
      if(array_has_nan == False):
        filters.append(bor)

    #______________________________Entropy______________________________________
    if "entropy" in method:
    
      ent = entropy(Z25, disk(i))

      #OUTPUT NORMALIZATION
      ent = normaliza(ent, ent)
      filters.append(ent) #coloca a entropia na lista  

    #_______________________________frangi______________________________________
    if "frangi" in method:
      fra = np.float32(frangi(Z25, [i*0.1]))

      #OUTPUT NORMALIZATION
      fra = normaliza(fra, fra)
      filters.append(fra)

    #______________________________meijering____________________________________
    if "meijering" in method:
    
      mei = np.float32(meijering(Z25,[i*0.1]))
      #OUTPUT NORMALIZATION
      mei = normaliza(mei, mei)

      #NaN VALUES VERIFICATION
      array_sum = np.sum(mei)
      array_has_nan = np.isnan(array_sum)
      if(array_has_nan == False):
        filters.append(mei)

    #_______________________________Sato________________________________________
    if "sato" in method:
    
      sat = np.float32(sato(Z25,[i*0.1]))
      #OUTPUT NORMALIZATION
      sat = normaliza(sat, sat)

      #NaN VALUES VERIFICATION
      array_sum = np.sum(sat)
      array_has_nan = np.isnan(array_sum)
      if(array_has_nan == False):
        filters.append(sat)

    #_________________________Gaussian Blur_____________________________________
    if "blur" in method:    
    
      gau = np.float32(gaussian(Z25, i))

      #OUTPUT NORMALIZATION
      gau = normaliza(gau, gau)

      #NaN VALUES VERIFICATION
      array_sum = np.sum(gau)
      array_has_nan = np.isnan(array_sum)
      if(array_has_nan == False):
        pass
        filters.append(gau)

      BAR = []
      BAR.append("#")
      return BAR



def sobel_alngles(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
   
    return theta 


def non_max_suppression(img, D):
    M, N = img.shape
    img = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180
    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z


def bwconncomp(orgImage, n):
  mag = orgImage/orgImage.max()*255
  mag = np.uint8(mag)
  totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(mag, n)

  return totalLabels, label_ids, values, centroid

def cleanLineSegments(orgImage, minLength):
  if(minLength<=1):
    minLength = minLength*np.sqrt(((orgImage.shape[0])**2 + (orgImage.shape[1])**2))
  
  connComps, label_ids, values, centroid = bwconncomp(orgImage,8)
  cleanImage = orgImage.copy()

  
  for idxComp in np.unique(label_ids):
    if(len(label_ids[label_ids == idxComp])<minLength):
      cleanImage[label_ids == idxComp] = 0
      
  return cleanImage

def threshold_otsu_impl(image, nbins=0.1):
    
    #validate grayscale
    if len(image.shape) == 1 or len(image.shape) > 2:
        print("must be a grayscale image.")
        return
    
    #validate multicolored
    if np.min(image) == np.max(image):
        print("the image must have multiple colors")
        return
    
    all_colors = image.flatten()
    total_weight = len(all_colors)
    least_variance = -1
    least_variance_threshold = -1
    
    # create an array of all possible threshold values which we want to loop through
    color_thresholds = np.arange(np.min(image)+nbins, np.max(image)-nbins, nbins)
    
    # loop through the thresholds to find the one with the least within class variance
    for color_threshold in color_thresholds:
        bg_pixels = all_colors[all_colors < color_threshold]
        weight_bg = len(bg_pixels) / total_weight
        variance_bg = np.var(bg_pixels)

        fg_pixels = all_colors[all_colors >= color_threshold]
        weight_fg = len(fg_pixels) / total_weight
        variance_fg = np.var(fg_pixels)

        within_class_variance = weight_fg*variance_fg + weight_bg*variance_bg
        if least_variance == -1 or least_variance > within_class_variance:
            least_variance = within_class_variance
            least_variance_threshold = color_threshold
        # print("trace:", within_class_variance, color_threshold)
            
    return least_variance_threshold

def threshold_otsu_RangedConstrain(image,th ,nbins=0.1):
    
    #validate grayscale
    if len(image.shape) == 1 or len(image.shape) > 2:
        print("must be a grayscale image.")
        return
    
    #validate multicolored
    if np.min(image) == np.max(image):
        print("the image must have multiple colors")
        return
    
    all_colors = image.flatten()
    total_weight = len(all_colors)
    least_variance = -1
    least_variance_threshold = -1
    
    # create an array of all possible threshold values which we want to loop through
    color_thresholds = np.arange(np.min(image)+nbins, th-nbins, nbins)
    
    # loop through the thresholds to find the one with the least within class variance
    for color_threshold in color_thresholds:
        bg_pixels = all_colors[all_colors < color_threshold]
        weight_bg = len(bg_pixels) / total_weight
        variance_bg = np.var(bg_pixels)

        fg_pixels = all_colors[all_colors >= color_threshold]
        weight_fg = len(fg_pixels) / total_weight
        variance_fg = np.var(fg_pixels)

        within_class_variance = weight_fg*variance_fg + weight_bg*variance_bg
        if least_variance == -1 or least_variance > within_class_variance:
            least_variance = within_class_variance
            least_variance_threshold = color_threshold
        # print("trace:", within_class_variance, color_threshold)
            
    return least_variance_threshold


def rc_otsu(final):
  th = threshold_otsu_impl(final, nbins=0.1)
  final[final < th] = 0
  
  th2 = threshold_otsu_RangedConstrain(final,th,nbins=0.1)
  final[final < th2] = 0
  final[final > th] = 1

  return final


#####################################################################################################################################
################################################## MAIN FUNCTIONS ###################################################################
#####################################################################################################################################

# Edge detection with X and Gaussian Blur________________________________________________________________________________________
def Edge_detection(gray,filters1, Disk_size, method, FileName):

  folder = "Filters"
  if not os.path.exists(folder):
    os.mkdir(folder)

  path = []

  for i in Disk_size:
    gau = np.float32(gaussian(gray, i*0.1))
    Filters_calc(gau, filters1, Disk_size, method)
    for j in Disk_size:
      for w in method:
        path.append(folder + "/" + FileName + "_" + w + "_Param_" + str(j*0.1) + "_Gauss_"+ "sigma_"+ str(i*0.1) +".png")
    
  for k, imagem in enumerate(filters1):
    cv2.imwrite(path[k],cv2.normalize(imagem, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F))

    #plt.imshow(imagem)
    #plt.show(block = False)
    #plt.pause(0.1)
    #plt.close("all")

# Agregation with various TNorms____________________________________________________________________________________________________
def Agreggation(filters1, Agreggation1, nomes, Tnormas, FileName, method):

  for i, nome in enumerate(nomes):
    print(nome)
    Agreggation1.append(choquet_Tconorm(filters1, Tnormas[nome]))
    #plt.imshow(Agreggation1[i], cmap='Greys')

    folder = "Agregation"

    if not os.path.exists(folder):
      os.mkdir(folder)

    Method = "_"
    for j in range(len(method)):
      Method += method[j] + "_"
      

    path = folder + "/" + FileName + "_" + "agregation" + Method +"_" + nome + ".png"
    cv2.imwrite(path,cv2.normalize(Agreggation1[i], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F))
    
    #plt.title(nome)
    #plt.show(block = False)
    #plt.pause(0.1)
    #plt.close("all")

#Taking detections angles with sobel_______________________________________________________________________________________________
def Angles(Agreggation1, angle_canny, nomes):
 
  for i, nome in enumerate(nomes):
    angle_canny.append(sobel_alngles(Agreggation1[i]))
    #plt.imshow(angle_canny[i])
    #plt.title(nome)
    #plt.show(block = False)
    #plt.pause(0.1)
    #plt.close("all")

# Applying Non-max Supression_____________________________________________________________________________________________________
def Non_Max_supression(Agreggation1, angle_canny, non_max_Canny, nomes, FileName, method):

  for i, nome in enumerate(nomes):
    non_max_Canny.append(non_max_suppression(Agreggation1[i], angle_canny[i]))
    #plt.imshow(non_max_Canny[i],cmap='Greys')
    
    folder = "Supression"

    if not os.path.exists(folder):
      os.mkdir(folder)

    Method = "_"
    for j in range(len(method)):
      Method += method[j] + "_"

    path = folder + "/" +  FileName + "_" + "NonMaxSup" + Method +"_" + nome + ".png"
    cv2.imwrite(path,cv2.normalize(non_max_Canny[i], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F))

    #plt.show(block = False)
    #plt.pause(0.1)
    #plt.close("all")

#Histerisys with Otsu____________________________________________________________________________________________________________

def Histerisys(final_Bin1, non_max_Canny, nomes, FileName, method):

  for i, nome in enumerate(nomes):
    final_Bin1.append(rc_otsu(non_max_Canny[i]))
    #plt.imshow(final_Bin1[i],cmap='Greys')
    
    folder = "Histerisys"

    if not os.path.exists(folder):
      os.mkdir(folder)

    Method = "_"
    for j in range(len(method)):
      Method += method[j] + "_"

    path = folder + "/" + FileName + "_" + "RCO" + Method + "_" + nome + ".png"
    cv2.imwrite(path,cv2.normalize(final_Bin1[i], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F))
    
    #plt.title(nome)
    #plt.show(block = False)
    #plt.pause(0.1)
    #plt.close("all")

# Clean segments with a thresholding of x______________________________________________________________________________________
def CleanSegments(final_Bin1, cleanImage1, nomes, num_segments, FileName, method):

  for i, nome in enumerate(nomes):
    cleanImage1.append(cleanLineSegments(final_Bin1[i], num_segments))
    #plt.imshow(cleanImage1[i],cmap='Greys')
    
    folder = "Clean segments"

    if not os.path.exists(folder):
      os.mkdir(folder)

    Method = "_"
    for j in range(len(method)):
      Method += method[j] + "_"

    path = folder + "/" + FileName + "_" + "CleanSeg" + Method +"_" + nome + ".png"
    cv2.imwrite(path,cv2.normalize(cleanImage1[i], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F))

    #plt.title(nome)
    #plt.show(block = False)
    #plt.pause(0.1)
    #plt.close("all")

