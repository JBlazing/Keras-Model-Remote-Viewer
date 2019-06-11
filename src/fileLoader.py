import cv2
import os
import re
import numpy as np
import sklearn as sk
def getDataPaths(file , filePaths=['','',''] , batches=127):


    f = open(file , 'r')
    
    b = f.readlines()
    
    f.close()
    
    b = [ l.split(',') for l in b ]
    
    for l in b:
        l[-1] = l[-1].rstrip()
    
    for l in b:
        for i in range(len(filePaths)):
            l[i] = filePaths[i] + l[i]
            
    b = np.array(b)
    
    splits = int(b.shape[0] / batches) - 1
    
    return np.array_split(b , splits)

    

def getFiles(path):
    
    files = os.fsencode(path)

    
    prods = {}
    for nam in os.listdir(files):
        name = nam.decode()
        parts = name.split("_")
        id = re.findall('\d+' , parts[0])
        id = int(id[0])
        #image = cv2.imread(path + os.fsdecode(nam))
        f_path = path + os.fsdecode(nam)
        
        if int(id) in prods:
            prods[id].append(f_path)
        else:
            prods[id] = [f_path]
        
    for key , itm in prods.items():
        itm.sort()
    return [ itm for key, itm in prods.items()]
    


def resizeImages(path , save_path , new_size):

    files = os.fsencode(path)
    
    for file in os.listdir(files):
        image = cv2.imread(path + os.fsdecode(file))
        image = cv2.resize(image , new_size)
        cv2.imwrite(save_path + os.fsdecode(file) , image)

def processImages(X):

    for i , x in enumerate(X):
        X[i] = cv2.normalize(x,X[i], -1,  1 , cv2.NORM_MINMAX , cv2.CV_32FC3)
    '''
    return  (   np.stack([ cv2.normalize(x , x ,-1 , 1 , norm_type=cv2.NORM_MINMAX , dtype=cv2.CV_32F) for x in X]) , \
                np.stack([ cv2.normalize(x , x ,-1 , 1 , norm_type=cv2.NORM_MINMAX , dtype=cv2.CV_32F) for x in Y]) )
    '''
    
def deNormalize(X):
    for i , x in enumerate(X):
        X[i] = cv2.normalize(x , X[i] , 0 , 255 , cv2.NORM_MINMAX , cv2.CV_32FC3)
        

def parse_Filenames(filesList):

    X = []
    Y = []
    Y_Idxs = []
    y_index = 0
    for item in filesList:
        models = item[ :-1]
        target = item[-1]
        t = [ y_index for x in models]
        X += models
        Y.append(target)
        Y_Idxs += t
        y_index += 1

    return X,Y,Y_Idxs

def get_disassociated(Y_Idxs , N):

    dis = []
    for Y in Y_Idxs:
        while True:
            r = np.random.randint(0,N)
            if r != Y:
                break
        
        dis.append(r)

    return dis


def loadFiles(X):

    return np.stack([ cv2.imread(x).astype(np.float32) for x in X ])

