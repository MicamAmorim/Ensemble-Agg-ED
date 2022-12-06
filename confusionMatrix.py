import seaborn as sns
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import *
from skimage.metrics import (adapted_rand_error,
                              variation_of_information)

import multiprocessing
from multiprocessing import Pool
import glob
from tabulate import tabulate

#Predito = "Clean segments/urso_CleanSeg_canny_Min.png"
#Real = "Ground Truth/urso_edges.jpg"
#canny = "Filters/urso_canny.png"

#Y_pred = cv2.imread(Predito)
#Y_val = cv2.imread(Real)
#Y_outro = cv2.imread(canny)

#Y_pred = cv2.cvtColor(Y_pred, cv2.COLOR_BGR2GRAY)
#Y_val = cv2.cvtColor(Y_val, cv2.COLOR_BGR2GRAY)
#Y_outro = cv2.cvtColor(Y_outro, cv2.COLOR_BGR2GRAY)

#_, Y_pred = cv2.threshold(Y_pred, 0, 255, cv2.THRESH_OTSU)
#_, Y_val = cv2.threshold(Y_val, 0, 255, cv2.THRESH_OTSU)
#_, Y_outro = cv2.threshold(Y_outro, 0, 255, cv2.THRESH_OTSU)

def convert2bin(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    return img


def viz(arr,x,y, m, n, r):
    zero = 0
    um = 0

    if r==0:
        if arr[x,y] == 0:
            zero+=1
        else:
            um+=1

        return [zero, um]
    else:
        for i in range(x-r, x+r+1):
            for j in range(y-r, y+r+1):
                if i >= 0 and i < m and j >= 0 and j < n:
                    
                    if arr[x,y] == 0:
                        zero+=1
                    else:
                        um+=1

    return [zero, um]



def confusão_raio(Y_pred, Y_val, r):
    m,n = Y_pred.shape


    FP = 0
    FN = 0
    TP = 0
    TN = 0
    for x in range(m):
        for y in range(n):

            VizPred = viz(Y_pred,x,y, m, n, r)
            VizVal = viz(Y_val,x,y, m, n, r)

            #print(VizPred, VizVal)


            if (( VizPred[1] - VizVal[1]) >= 1):
                FP += 1
            if ((VizPred[0] - VizVal[0]) >= 1):
                FN += 1
            if (VizPred[1] == VizVal[1]):
                TP += 1
            if ((VizPred[0] == VizVal[0])):    
                TN += 1
              
    cmat = [[TP, FN], 
            [FP, TN]]
    return cmat

def print_confusion(cmat):
    plt.figure(figsize = (6,6))
    sns.heatmap(cmat/np.sum(cmat), cmap="Reds", annot=True, fmt = '.2%', square=1,   linewidth=2.)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.show()

#__________________________________________ M E T R I C S ___________________________________________


def metrics_table(Y_val, r, method_names, method_img_list):
    mydata = []
    for name, im_test in zip(method_names, method_img_list):

        #pool = Pool(multiprocessing.cpu_count() - 1)

        #res = pool.apply_async(confusão_raio, (im_test, Y_val, r))
        #cmat = res.get()
        cmat = confusão_raio(im_test, Y_val, r)

        print(cmat)

        Accuracy = (cmat[0][0] + cmat[1][1])/(cmat[0][0] + cmat[0][1] + cmat[1][0] + cmat[1][1])
        Precision = cmat[0][0]/(cmat[0][0] + cmat[1][0])
        Sensitivity  = cmat[0][0]/(cmat[0][0] + cmat[0][1])
        Specificity  = cmat[1][1]/(cmat[1][1] + cmat[1][0])
        Fscore = 2*((Precision * Sensitivity) / (Precision + Sensitivity))

        print(f'\n## Results for: {name}')
        print(f'Accuracy: {Accuracy}')
        print(f'Precision: {Precision}')
        print(f'Sensitivity: {Sensitivity}') 
        print(f'Specificity: {Specificity}')
        print(f'Fscore: {Fscore}')

        mydata.append([name, Accuracy, Precision, Sensitivity, Specificity, Fscore])

    head = ["Method", "Accuracy", "Precision", "Sensitivity", "Specificity", "Fscore"]
    table = tabulate(mydata, headers=head, tablefmt="tsv")
    #grid
    folder = "Metrics"

    if not os.path.exists(folder):
      os.mkdir(folder)

    fp = open(folder + '/Metrics_table_'+ name +'.csv', 'w')
    fp.write(table)
    print(table)
    fp.close()


    
    

    





