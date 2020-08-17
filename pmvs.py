import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from copy import copy,deepcopy
from optical_center import getOpticalCenter
from scipy.optimize import minimize
from time import time

def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def get_nth_maximum(arr,n):
    uniqueValues = list(arr.flatten())
    uniqueValues.sort()
    if len(uniqueValues) < n:
        return uniqueValues[0]
    return uniqueValues[len(uniqueValues)-n]

def get_optical_axis(projectionMat):
    return np.array([projectionMat[2][0],projectionMat[2][1],projectionMat[2][2],0])

def outside_image_boundry(yCoord,xCoord,height,width):
    return (xCoord < 0 or yCoord < 0 or xCoord >= width or yCoord >= height)

def get_qt_counts():
    qts = np.array([])
    for idx,immod in enumerate(imagesModels):
        numberoffree = 0
        for i in range(len(immod['grid'])):
            for j in range(len(immod['grid'][0])):
                if len(immod['grid'][i][j]['Qt']) == 0:
                    numberoffree +=1
        qts = np.append(qts,numberoffree)
    return qts

def checkKey(dict,key):
    if key in dict.keys():
        return True
    else:
        return False

def constants(dataPath):
    datasetPath = dataPath
    ß1 = 2
    ß2 = 32
    µ = 5       # the projection of one of its edges into R(p) is parallel to the image rows, and the smallest axis-aligned square containingits image covers a µ × µ pixel^2 area
    # We associate with p a reference image R(p),the images S(p) where p should be visible and the images T(p) where it is truly found 
    gamma = 3

    cosMinAngle = np.math.cos(np.math.radians(20))
    cosMaxAngle = np.math.cos(np.math.radians(60))
    patchGridSize = 5
    cell = {
        "Qt":list(),
        "Qf":list()
    }

# Initialize image model from a given path
def init_imgs(datasetPath):
    # Read imgs
    filesNames = glob(datasetPath+'images/*.jpg')
    filesNames = sorted(filesNames)
    imgs = [cv.imread(file) for file in filesNames]
    
    # Construct corresponding image grid
    grids = list()
    for img in imgs:
        grid = np.array([np.array([cell for x in range(0,img.shape[1]//ß1)]) for y in range(0,img.shape[0]//ß1)])
        for i in range(len(grid)):
            for j in range(len(grid[0])):      
                cell1={
                    "Qt":list(),
                    "Qf":list()
                }
                grid[i][j] = cell1
        grids.append(grid)
        
    return imgs,grids
    
# Read camera parameters and return the projection matrices for all pictures
def read_parameters_file(datasetPath):
    
    filesNames = glob(datasetPath+'projections/*.txt')
    filesNames = sorted(filesNames)
    files = [open(file) for file in filesNames]
    projections = []
    optAxes = []
    
    for file in files:
        lines = file.readlines()
        p = np.zeros((3,4))
        for idx,line in enumerate(lines):
            # CONTOUR
            if idx == 0:
                continue
                
            row = line.split()
            for cIdx,col in enumerate(row):
                p[idx-1][cIdx] = np.float32(col)
                
        projections.append(p)
        optAxis = get_optical_axis(p)
        optAxis *= np.linalg.det(p[:,:-1])
        norm = np.linalg.norm(optAxis)
        optAxis /= norm
        optAxes.append(optAxis)
        
    return projections,optAxes

# Get Harris and DoG operators for a given image
def get_dog_harris(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
            
    # Get DoG
    g1 = cv.GaussianBlur(gray,(0,0),sigmaX=1)
    g2 = cv.GaussianBlur(gray,(0,0),sigmaX=1*np.sqrt(2))
    diff = cv.absdiff(g1,g2)
    dog = diff

    # Get Harris
    bSize = 3
    kSize = 1
    corners = cv.cornerHarris(src=gray,blockSize=bSize,ksize=kSize,k=0.06)
    
    return dog , corners

def sparse_dog_harris(dog,harris):
    n = 4
    sparseDog = copy(dog)
    sparseHarris = copy(harris)
    sparseDogPositions = []
    sparseHarrisPositions = []
    for yIdx in range(0,len(dog),ß2):
        for xIdx in range(0,len(dog[0]),ß2):
            nThMaximumDog = get_nth_maximum(dog[yIdx:yIdx+ß2,xIdx:xIdx+ß2],n)
            if nThMaximumDog != -1:
                found = False
                for rowIdx,row in enumerate(dog[yIdx:yIdx+ß2]):
                    for columnIdx,column in enumerate(row[xIdx:xIdx+ß2]):
                        if not found and column == nThMaximumDog:
                            found = True
                            if column != 0:
                                sparseDogPositions.append((xIdx+columnIdx,yIdx+rowIdx))
                        else:
                            sparseDog[yIdx+rowIdx,xIdx+columnIdx] = 0
                # sparseDog[yIdx:yIdx+ß2,xIdx:xIdx+ß2] = sparseDog[yIdx:yIdx+ß2,xIdx:xIdx+ß2]*(sparseDog[yIdx:yIdx+ß2,xIdx:xIdx+ß2] == nThMaximumDog)
            nThMaximumHarris = get_nth_maximum(harris[yIdx:yIdx+ß2,xIdx:xIdx+ß2],n)
            if nThMaximumHarris != -1:
                found = False
                for rowIdx,row in enumerate(harris[yIdx:yIdx+ß2]):
                    for columnIdx,column in enumerate(row[xIdx:xIdx+ß2]):
                        if not found and column == nThMaximumHarris:
                            found = True
                            if column != 0:
                                sparseHarrisPositions.append((xIdx+columnIdx,yIdx+rowIdx))
                        else:
                            sparseHarris[yIdx+rowIdx,xIdx+columnIdx] = 0
                # sparseHarris[yIdx:yIdx+ß2,xIdx:xIdx+ß2] = sparseHarris[yIdx:yIdx+ß2,xIdx:xIdx+ß2]*(sparseHarris[yIdx:yIdx+ß2,xIdx:xIdx+ß2] == nThMaximumHarris)
            # show_images([dog[yIdx:yIdx+ß2,xIdx:xIdx+ß2],sparseDog[yIdx:yIdx+ß2,xIdx:xIdx+ß2],harris[yIdx:yIdx+ß2,xIdx:xIdx+ß2],sparseHarris[yIdx:yIdx+ß2,xIdx:xIdx+ß2]],['before dog','after dog','before harris','after harris'])

    # sparseDog = cv.dilate(sparseDog,None)
    # sparseDog = cv.dilate(sparseDog,None)
    # sparseHarris = cv.dilate(sparseHarris,None)
    # sparseHarris = cv.dilate(sparseHarris,None)
    return sparseDog,sparseHarris,sparseDogPositions,sparseHarrisPositions

def compute_fundamental(x1,x2):
    """ Computes the fundamental matrix from corresponding points
    (x1,x2 3*n arrays) using the normalized 8 point algorithm.
    each row is constructed as
    [x’*x, x’*y, x’, y’*x, y’*y, y’, x, y, 1] """
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don’t match.")
    # build matrix for equations
    A = np.zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
        x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
        x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
    
    # compute linear least square solution
    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)
    # constrain F
    # make rank 2 by zeroing out last singular value
    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U,np.dot(np.diag(S),V))
    return F

def compute_epipole(F):
    """ Computes the (right) epipole from a
    fundamental matrix F.
    (Use with F.T for left epipole.) """
    # return null space of F (Fx=0)
    U,S,V = np.linalg.svd(F)
    e = V[-1]
    return e/e[2]

def plot_epipolar_line(im,F,x,epipole=None,show_epipole=True):
    """ Plot the epipole and epipolar line F*x=0
    in an image. F is the fundamental matrix
    and x a point in the other image."""
    m,n = im.shape[:2]
    line = np.dot(F,x)
    # epipolar line parameter and values
    t = np.linspace(0,n,100)
    lt = np.array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])
    # take only line points inside the image
    ndx = (lt>=0) & (lt<m)
    plt.plot(t[ndx],lt[ndx],linewidth=2)
    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)
        plt.plot(epipole[0]/epipole[2],epipole[1]/epipole[2],'r*')

# Get the fundmental matrix between 2 pictures
def get_fundmental_matrix_book(idx1,idx2):
    sift = cv.xfeatures2d.SIFT_create()
    # find keypoints and descriptors with SIFT
    kp1,des1 = sift.detectAndCompute(images[idx1],None)
    kp2,des2 = sift.detectAndCompute(images[idx2],None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    pts1 = []
    pts2 = []

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    print("pts1.shape:%s\tpts2.shape:%s"%(pts1.shape,pts2.shape))
    x1 = np.vstack( (pts1,np.ones(pts1.shape[1])) )
    x2 = np.vstack( (pts2,np.ones(pts2.shape[1])) )

    fundmentalMat = compute_fundamental(x1,x2)
    # compute the epipole
    e = compute_epipole(fundmentalMat)
    
    # plotting
    plt.figure()
    plt.imshow(images[0])
    for i in range(5):
        plot_epipolar_line(images[0],fundmentalMat,x2[:,i],e,False)
    plt.axis('off')

    plt.figure()
    plt.imshow(im2)
    # plot each point individually, this gives same colors as the lines
    for i in range(5):
        plt.plot(x2[0,i],x2[1,i],'o')
    plt.axis('off')
    
    print(("Fundmental Matrix between image[%d] and image[%d]:\n%a") % (idx1,idx2,fundmentalMat))
    return fundmentalMat



# Get the fundmental matrix between 2 pictures
def get_fundmental_matrix_sift(idx1,idx2):
    sift = cv.xfeatures2d.SIFT_create()
    # find keypoints and descriptors with SIFT
    kp1,des1 = sift.detectAndCompute(images[idx1],None)
    kp2,des2 = sift.detectAndCompute(images[idx2],None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    pts1 = []
    pts2 = []

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    print("pts1.shape:%s\tpts2.shape:%s"%(pts1.shape,pts2.shape))
    fundmentalMat, _ = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
    print(("Fundmental Matrix between image[%d] and image[%d]:\n%a") % (idx1,idx2,fundmentalMat))
    return fundmentalMat



# skewForm : skewForm(v).dot(u) = v cross u
def skewForm(vec):
    sk = np.zeros((3,3))
    sk[0][0] = 0
    sk[0][1] = -vec[2]
    sk[0][2] = vec[1]
    sk[1][0] = vec[2]
    sk[1][1] = 0
    sk[1][2] = -vec[0]
    sk[2][0] = -vec[1]
    sk[2][1] = vec[0]
    sk[2][2] = 0
    # sk = np.array(
    #     [0,-vec[2],vec[1]],
    #     [vec[2],0,-vec[0]],
    #     [-vec[1],vec[0],0]
    #     )

    return sk

def get_fundmental_matrix(img1,img2):
    p00 = img1["projMat"][0].reshape(1,4)
    p01 = img1["projMat"][1].reshape(1,4)
    p02 = img1["projMat"][2].reshape(1,4)

    p10 = img2["projMat"][0].reshape(1,4)
    p11 = img2["projMat"][1].reshape(1,4)
    p12 = img2["projMat"][2].reshape(1,4)

    F = np.zeros((3,3))
    
    ppinv = np.zeros((3,3))

    ppinv = np.matmul(img2["projMat"], np.linalg.pinv(img1["projMat"]))

    epipole = np.zeros((3,1))

    epipole = np.matmul(img2["projMat"],img1["optCenter"])
    
    funMat = np.zeros((3,3))

    funMat = np.matmul(skewForm(epipole),ppinv)

    return funMat

    # F[0][0] = np.linalg.det(np.concatenate((p01, p02, p11, p12),axis=0))
    # F[0][1] = np.linalg.det(np.concatenate((p01, p02, p12, p10),axis=0))
    # F[0][2] = np.linalg.det(np.concatenate((p01, p02, p10, p11),axis=0))

    # F[1][0] = np.linalg.det(np.concatenate((p02, p00, p11, p12),axis=0))
    # F[1][1] = np.linalg.det(np.concatenate((p02, p00, p12, p10),axis=0))
    # F[1][2] = np.linalg.det(np.concatenate((p02, p00, p10, p11),axis=0))

    # F[2][0] = np.linalg.det(np.concatenate((p00, p01, p11, p12),axis=0))
    # F[2][1] = np.linalg.det(np.concatenate((p00, p01, p12, p10),axis=0))
    # F[2][2] = np.linalg.det(np.concatenate((p00, p01, p10, p11),axis=0))
    
    # return F

# Draw the epilines corresponding to a point in the first image
# Draw also the points satisfying epipolar consistancy 
def drawlines(img1,lines,pts1):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''

    reducedFeaturesImage = copy(img1["image"])
    fullFeaturesImage = copy(img1["image"])
    r,c,_ = reducedFeaturesImage.shape
    
    for r,pt1 in zip(lines,pts1):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        cv.line(reducedFeaturesImage, (x0,y0), (x1,y1), color,1)
        cv.line(fullFeaturesImage, (x0,y0), (x1,y1), color,1)
    
    
    a = lines[0][0]
    b = lines[0][1]
    c = lines[0][2]
    
    maxDistance = 2 * np.sqrt((a**2)+(b**2)) 
    legalFeatures = []
    for pt in pts1:
        # color = tuple(np.random.randint(0,255,3).tolist())
        ptx = pt[0]
        pty = pt[1]
        if abs(a*ptx+b*pty+c) <= maxDistance :
            cv.circle(reducedFeaturesImage,tuple(pt),5,(0,255,0),-1)
            cv.circle(fullFeaturesImage,tuple(pt),5,(0,255,0),-1)
            legalFeatures.append(np.float32([ptx,pty]))
        else:
            cv.circle(fullFeaturesImage,tuple(pt),5,(255,0,0),-1)

    return fullFeaturesImage,reducedFeaturesImage,legalFeatures



#global variables needed by the objective function
referenceImgIdx = 0
depthVec = 0    
optimPhotos = 0



def encode(center, normal, photo, opticalCenter):
    depthVector = center - opticalCenter.reshape(4,1)
    depth = np.linalg.norm(np.array(depthVector))
    theta = np.math.acos(normal[2])#pitch
    phi = np.math.atan2(normal[1], normal[0])#yaw
    depthVector /= depth
    return depth, theta, phi, depthVector



def decode(imageModel, unitDepthVec, depth, theta, phi):
    opticalCenter = imageModel["optCenter"]
    depthVector = depth * unitDepthVec
    center = opticalCenter.reshape(4,1) + depthVector
    normal = np.zeros((4,1))
    normal[0] = np.math.sin(theta)*np.math.cos(phi)
    normal[1] = np.math.sin(theta)*np.math.sin(phi)
    normal[2] = np.math.cos(theta)
    return center, normal



def ncc_objective(center, rightVector, upVector, refPhotoModel, targetPhotosIDs):

    cell1 = project_patch(center, refPhotoModel, rightVector, upVector)#overload to get the center  #TODO
    SumNcc = 0
    for i in range(len(targetPhotosIDs)):
        photo = imagesModels[targetPhotosIDs[i]['idx']]
        cell2 = project_patch(center, photo, rightVector, upVector)
        SumNcc += ncc_score(cell1, cell2)
    
    return SumNcc / len(targetPhotosIDs)



def objective(x):
    depth, theta, phi = x[0], x[1], x[2]
    center, normal = decode(imagesModels[referenceImgIdx], depthVec, depth, theta, phi)
    #TODO#some conditions
    if np.dot(imagesModels[referenceImgIdx]["optAxis"], depthVec) < 0:
        return 1.0
    patch = {}
    patch["center"] = center
    patch["normal"] = normal
    patch["referenceImgIdx"] = referenceImgIdx
    right, up = get_patch_vectors(patch) 
    return -ncc_objective(center, right, up, imagesModels[referenceImgIdx], optimPhotos)



def normalize(depth, unitDepthVector, patchTrueSet):
    sum = 0
    for i in range (len(patchTrueSet)):
        photo = imagesModels[patchTrueSet[i]['idx']]
        depthVectorProj = np.matmul(photo['projMat'], unitDepthVector)
        depthVectorProj /= depthVectorProj[2]
        sum += np.linalg.norm(np.array(depthVectorProj[-1])) #remove t
        
    sum /= len(patchTrueSet)
    unitDepthVector /= sum
    depth *= sum
    return depth, unitDepthVector



def optimize_patch(patch):
    global referenceImgIdx
    global depthVec
    global optimPhotos
    refPhoto = imagesModels[patch["referenceImgIdx"]]["image"]
    opticalCenter = imagesModels[patch["referenceImgIdx"]]["optCenter"]

    depth, theta, phi, unitDepthVec = encode(patch["center"], patch["normal"], refPhoto, opticalCenter)
    depth, unitDepthVec = normalize(depth, unitDepthVec, patch["trueSet"]) #TODO add trueset to patch
    targetPhotos = patch["trueSet"]
    referenceImgIdx, depthVec, optimPhotos = patch["referenceImgIdx"], unitDepthVec, targetPhotos

    option = {
        'disp': False, #Set to True to print convergence messages.
        'maxiter': 1000,
        'xatol': 0.0005,
        'adaptive' : False #adaptivebool, optional #Adapt algorithm parameters to dimensionality of problem. Useful for high-dimensional minimization
    }
    initialGuess = np.array([depth, theta, phi])
    solution  = minimize(objective, initialGuess, method='Nelder-Mead', options = option)
    center, normal = decode(imagesModels[patch["referenceImgIdx"]], unitDepthVec, solution.x[0], solution.x[1], solution.x[2])
    patch["center"], patch["normal"] = center, normal



def get_relevent_images(imgModels,idx):
    releventImages = []
    myOptAxis = imgModels[idx]["optAxis"]

    for i in range(len(imgModels)):
        if i == idx:
            continue
        otherOptAxis = imgModels[i]["optAxis"]
        cosAngle = np.dot(myOptAxis,otherOptAxis)

        if cosAngle > np.math.cos(np.math.pi/3):
            releventImages.append(i)

    return releventImages



def ncc_score(cell1,cell2):
    mean1 = np.mean(cell1)
    mean2 = np.mean(cell2)
    
    std1 = std2 = product = 0
	
    for i in range(len(cell1)):
        diff1 = cell1[i] - mean1
        diff2 = cell2[i] - mean2
        product += diff1 * diff2
        std1 += diff1 * diff1
        std2 += diff2 * diff2
	
    stds = std1 * std2
    if stds == 0:
        return 0

    return product / np.math.sqrt(stds)

def project_patch(patchCenter,imgModel,rightVector,upVector):
    cell = np.zeros(patchGridSize*patchGridSize*3)
    
    projMat = imgModel["projMat"]
    projCenter = np.matmul(projMat,patchCenter)
    projRight = np.matmul(projMat,rightVector).reshape(3,1)
    projUp = np.matmul(projMat,upVector).reshape(3,1)

    scale = 1/projCenter[2]
    projCenter = scale * projCenter
    projRight = scale * projRight
    projUp = scale * projUp

    step = (patchGridSize-1)/2
    diagVector = projUp + projRight
    diagVector = step * diagVector
    topLeftVector = projCenter - diagVector

    cellIdx = 0
    for i in range(patchGridSize):
        for j in range(patchGridSize):
            xCoord = topLeftVector[0] + i*projUp[0] + j*projRight[0]
            yCoord = topLeftVector[1] + i*projUp[1] + j*projRight[1]
            yCoord = int(yCoord+0.5)
            xCoord = int(xCoord+0.5)

            # pixel is outside the image
            if outside_image_boundry(yCoord,xCoord,len(imgModel['image']),len(imgModel['image'][0])):
                cell[cellIdx], cell[cellIdx+1], cell[cellIdx+2] = 0,0,0
            else:
                cell[cellIdx], cell[cellIdx+1], cell[cellIdx+2] = imgModel["image"][yCoord][xCoord]

            cellIdx +=3

    return cell

def get_ncc_score(patch,releventImgModel,rightVector,upVector):
    referenceImgModel = imagesModels[patch["referenceImgIdx"]]

    cell1 = project_patch(patch["center"],referenceImgModel,rightVector,upVector)
    cell2 = project_patch(patch["center"],releventImgModel,rightVector,upVector)
    return ncc_score(cell1,cell2)

def get_patch_vectors(patch):
    referenceImageModel = imagesModels[patch["referenceImgIdx"]]
    projMat = referenceImageModel["projMat"]

    ppinv = np.linalg.pinv(projMat)

    scale = np.dot(ppinv[:,0],patch["normal"])
    rightVector = ppinv[:,0].reshape(4,1) - scale*patch["normal"]

    scale = np.dot(ppinv[:,1],patch["normal"])
    upVector = ppinv[:,1].reshape(4,1) - scale*patch["normal"]


    scale = np.dot(projMat[2],patch["center"])
    rightVector = (scale/(np.dot(projMat[0],rightVector)))*rightVector
    upVector = (scale/(np.dot(projMat[1],upVector)))*upVector

    return rightVector, upVector

def get_t_images(patch,alfa,visibleImages):
    tImages = []

    rightVector,upVector = get_patch_vectors(patch)
    for visibleImage in visibleImages:
        visibleImageIdx = visibleImage['idx']
        visibleImageModel = imagesModels[visibleImageIdx]
        
        depthVector = np.float32([
            visibleImageModel["optCenter"][0] - patch["center"][0],
            visibleImageModel["optCenter"][1] - patch["center"][1],
            visibleImageModel["optCenter"][2] - patch["center"][2],
            visibleImageModel["optCenter"][3] - patch["center"][3]
        ])

        if np.dot(np.squeeze(depthVector), np.squeeze(patch["normal"])) <= 0:
            continue
        
        nccScore = get_ncc_score(patch, visibleImageModel, rightVector, upVector)
        if (1- nccScore) <= alfa:
            #imgCoord = np.matmul(visibleImageModel['projMat'], patch['center'])
            #imgCoord = imgCoord/imgCoord[2][0] #divide by t
            
            #x = int(imgCoord[0][0]) // ß1
            #y = int(imgCoord[1][0]) // ß1
            visibleImage['true'] = True
            tImages.append({
                "idx":visibleImageIdx,
                "gStarScore": 1-nccScore
                #"cell":{
                #    'ptx':x,
                #    'pty':y
                #},
            }) #TODO remove nccscore if not used
    
    return tImages

def get_visible_images(patch,releventImgsIdxs):
    visibleSet = []
    pNormal3 = np.array([patch['normal'][0][0],patch['normal'][1][0],patch['normal'][2][0]]).reshape(3,1)
    for releventIdx in releventImgsIdxs:
        if releventIdx == patch['referenceImgIdx']:
            continue
        
        viewVector = imagesModels[releventIdx]['optCenter'].reshape(4,1) - patch['center']
        viewVector3 = np.array([viewVector[0][0], viewVector[1][0], viewVector[2][0]]).reshape(3,1)
        viewVector3 = viewVector3 / np.linalg.norm(viewVector3)
        #print("get_visible_images: \npNormal3", pNormal3, " \nviewVector3: ", viewVector3)
        if  np.dot(np.squeeze(pNormal3),np.squeeze(viewVector3)) > np.math.cos(np.math.pi/3):
            #print("Normal:",pNormal3,"\nviewVector",viewVector3)
            #imgCoord = np.matmul(imagesModels[releventIdx]['projMat'], patch['center'])
            #print('Bdfore:',patch['center'],'\n',imgCoord)
            #imgCoord = imgCoord/imgCoord[2][0] #divide by t
            #print('After:',patch['center'],'\n',imgCoord)
            
            #x = int(imgCoord[0][0]) // ß1
            #y = int(imgCoord[1][0]) // ß1
            visibleSet.append({
                'idx':releventIdx,
                'true':False
                #"cell":{
                #   'ptx':-1,
                 #  'pty':-1
                #}
            })
    
    #imgCoord = np.matmul(imagesModels[releventIdx]['projMat'], patch['center'])
    #imgCoord = imgCoord/imgCoord[2][0] #divide by t

    #x = int(imgCoord[0][0]) // ß1
    #y = int(imgCoord[1][0]) // ß1
    visibleSet.append({
        'idx':patch['referenceImgIdx'],
        'true':False
     #   "cell":{
     #       'ptx':x,
      #      'pty':y
     #   }
    })
    return visibleSet



def register_patch(patch):
    add = False
    for sImg in patch["visibleSet"]:
        imgModel = imagesModels[sImg['idx']]
        imgCoord = np.matmul(imgModel['projMat'], patch['center'])
        imgCoord = imgCoord/imgCoord[2][0] #divide by t
        #print("register_patch: imgCoord", imgCoord)
        x = int(imgCoord[0][0]) // ß1
        y = int(imgCoord[1][0]) // ß1
        #print("register_patch: x", x, " y:",y)
        if outside_image_boundry(y,x,len(imagesModels[0]['image'])//ß1,len(imagesModels[0]['image'][0])//ß1):
            sImg['cell'] = {
            'ptx':-1,
            'pty':-1,
            }
            continue
        add = True
        cell1 = imgModel['grid'][y][x]
        patch['isOutlier'] = False
        if sImg['true']:
            cell1['Qt'].append(patch)
        else:
            cell1['Qf'].append(patch)
            
#         if not any([imgIdx == sImg['idx'] for imgIdx in patch['trueSet']]):
#             cell1['Qf'].append(patch)
#         else:
#             cell1['Qt'].append(patch)
        
        sImg['cell'] = {
            'ptx':x,
            'pty':y,
        }
    if add:
        patches.append(patch)



def empty_cell(imageID, y, x):
    cell_y = y // ß1
    cell_x = x // ß1
    if len(imagesModels[imageID]['grid'][cell_y][cell_x]['Qt']) == 0 and len(imagesModels[imageID]['grid'][cell_y][cell_x]['Qf']) == 0 :
        return True
    return False



def get_features_statsify_epipoler_consistency(baseImageIdx, featurePt,featureType):
    triangulations = list()
    for i in range(len(imagesModels[baseImageIdx]["releventImgsIdxs"])):
        releventImageIdx = imagesModels[baseImageIdx]["releventImgsIdxs"][i]
        fundmentalMat = get_fundmental_matrix(imagesModels[baseImageIdx],imagesModels[releventImageIdx])

        if fundmentalMat is None:
            continue
        pt1 = featurePt
        pts1 = np.int32([pt1])
        pts2 = np.int32(imagesModels[releventImageIdx][featureType])

        originalWithFeaturePt = imagesModels[baseImageIdx]["image"].copy()
        cv.circle(originalWithFeaturePt,tuple(pt1),5,(0,0,255),-1)

        # Get the epilines of features in left image on the right image
        # parameter1: points required to get its epilines in the other image
        # parameter2: which image that points are belong, 1-left 2-right
        # parameter3: fundmental matrix between the 2 images
        # returns list of epilines that lie on the other image and corresponding to the points
        lines = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,fundmentalMat)
        lines = lines.reshape(-1,3)
        #sara_im = plot_epipolar_line(imagesModels[i]["image"],fundmentalMat,[pt1[0],pt1[1],1])

        # draw the epiline on the other image
        # parameter1: the second image
        # parameter2: the epilines that lie on the second image
        # parameter3: the features lie on the second image
        fullFeaturesImage,reducedFeaturesImage,legalFeatures = drawlines(imagesModels[releventImageIdx],lines,pts2)

        #Triangulation
        for j in range(len(legalFeatures)):
            if not(empty_cell(releventImageIdx, int(legalFeatures[j][1]), int(legalFeatures[j][0]))): #TODO check t = 1
                continue
                
            triangulatedPointsHomogeneous = cv.triangulatePoints(imagesModels[baseImageIdx]["projMat"],imagesModels[releventImageIdx]["projMat"],pt1,legalFeatures[j])
            triangulatedPoint = triangulatedPointsHomogeneous[:4, :] / triangulatedPointsHomogeneous[3, :]

            #triangulatedPoint = triangulate_point(np.array([pt1[0], pt1[1],1]),legalFeatures[j],imagesModels[baseImageIdx]["projMat"],imagesModels[i]["projMat"])

            distFromcenter = abs(abs(np.linalg.norm(np.array(imagesModels[baseImageIdx]["optCenter"][:-1]) - np.array([triangulatedPoint[0][0], triangulatedPoint[1][0], triangulatedPoint[2][0]]))) - abs(np.linalg.norm(np.array(imagesModels[releventImageIdx]["optCenter"][:-1]) - np.array([triangulatedPoint[0][0], triangulatedPoint[1][0], triangulatedPoint[2][0]]))))

            triangulation = {
                "originalImg": releventImageIdx,
                "position": triangulatedPoint,
                "distFromCenter": distFromcenter,
                "ptx": legalFeatures[j][0],
                "pty": legalFeatures[j][1]
            }

            triangulations.append(triangulation)

        #show_images([imagesModels[baseImageIdx]["image"],imagesModels[releventImageIdx]["image"],fullFeaturesImage,reducedFeaturesImage, originalWithFeaturePt],["image"+str(baseImageIdx),"image"+str(releventImageIdx),"fullfeatures in image"+str(releventImageIdx),"reducedfeatures in image"+str(releventImageIdx), "originalWithFeaturePt"+str(releventImageIdx)])

    triangulations = sorted(triangulations, key=lambda k: k["distFromCenter"]) 
    #for i in range(len(triangulations)):
        #print("triangulations: ", triangulations[i]["originalImg"], "ptx", triangulations[i]["ptx"], "pty", triangulations[i]["pty"], triangulations[i]["distFromCenter"])
    
    return triangulations



def construct_patches(baseImageIdx, triangulations):
    #print("construct_patches ...")
    baseOptCenter = imagesModels[baseImageIdx]["optCenter"]
    for candidate in triangulations:

        patch = {}
        patch["referenceImgIdx"] = baseImageIdx
        patch["center"] = candidate["position"]
        patch["normal"] = np.float32([
                baseOptCenter[0] - candidate["position"][0],
                baseOptCenter[1] - candidate["position"][1],
                baseOptCenter[2] - candidate["position"][2],
                baseOptCenter[3] - candidate["position"][3],
            ])
        patch["normal"] = patch["normal"] / np.linalg.norm(patch["normal"])

        patch['visibleSet'] = get_visible_images(patch,imagesModels[baseImageIdx]["releventImgsIdxs"])
        #print(patch['visibleSet'][:])
        #break
        patch["trueSet"] = get_t_images(patch,0.6,patch['visibleSet']) 

        #print("len(patch[trueSet]): ", len(patch["trueSet"]))
        if len(patch["trueSet"]) <= 1 : 
            continue

        optimize_patch(patch)
        patch["visibleSet"] = get_visible_images(patch,imagesModels[baseImageIdx]["releventImgsIdxs"])
        patch["trueSet"] = get_t_images(patch,0.3,patch['visibleSet'])
        #print("len(patch[trueSet]): ", len(patch["trueSet"]), " gamma: ", gamma)
        if len(patch["trueSet"]) >= gamma:
            patch['gStarScore'] = sum([tImg['gStarScore'] if tImg['idx'] != patch['referenceImgIdx'] else 0 for tImg in patch['trueSet']])
            patch['gStarScore'] /= (len(patch['trueSet'])-1)
            register_patch(patch)
            break

# 


from pickle import dump,load
def save_data(phase,patches=None):
    for i,imageModel in enumerate(imagesModels):
        print("save file",i)
        a_file = open(datasetPath+phase+"/ImageModel"+str(i)+".pkl", "wb")
        dump(imageModel,a_file)
        a_file.close()
        
    if patches:
        print("save file patches")
        a_file = open(datasetPath+phase+"/patches.pkl", "wb")
        dump(patches,a_file)
        a_file.close()
                      
    print(phase,"Saving---->DONE")

def load_data(phase):
    loadedImagesModels = []
    patches = []
    for i in range(16):
        print("load file",i)
        a_file = open(datasetPath+phase+"/ImageModel"+str(i)+".pkl","rb")
        loadedImagesModels.append(load(a_file))
        a_file.close()
    try:
        a_file = open(datasetPath+phase+"/patches.pkl","rb")
        patches = load(a_file)
        a_file.close()
    except:
        print("no patches")
        
    print(phase,"Loading---->DONE")
    return loadedImagesModels,patches



def write_header(file):
    file.write("ply\n")
    file.write("format ascii 1.0\n")
    file.write("element vertex "+ str(len(patches)) + "\n")
    file.write("property float x\n")
    file.write("property float y\n")
    file.write("property float z\n")
    file.write("property float nx\n")
    file.write("property float ny\n")
    file.write("property float nz\n")
    file.write("property uchar diffuse_red\n")
    file.write("property uchar diffuse_green\n")
    file.write("property uchar diffuse_blue\n")
    file.write("end_header\n")



def write_ply(): 
    file = open("matching_pointcloud.txt.ply", "w")
    write_header(file)
    for patch in patches:
        file.write(str(patch["center"][0][0]) + " " +  str(patch["center"][1][0]) + " " + str(patch["center"][2][0]) + " ")
        file.write(str(patch["normal"][0][0]) + " " +  str(patch["normal"][1][0]) + " " + str(patch["normal"][2][0]) + " ")
        file.write("255"+ " " + "0" + " "+"0")
        file.write("\n")
    file.close()



def write_expanded_header(file):
    file.write("ply\n")
    file.write("format ascii 1.0\n")
    file.write("element vertex "+ str(len(patches)) + "\n")
    file.write("property float x\n")
    file.write("property float y\n")
    file.write("property float z\n")
    file.write("property float nx\n")
    file.write("property float ny\n")
    file.write("property float nz\n")
    file.write("property uchar diffuse_red\n")
    file.write("property uchar diffuse_green\n")
    file.write("property uchar diffuse_blue\n")
    file.write("end_header\n")



def write_expanded_ply(): 
    file = open("expansion_pointcloud.txt.ply", "w")
    write_expanded_header(file)
    for idx,patch in enumerate(patches):
        file.write(str(patch["center"][0][0]) + " " +  str(patch["center"][1][0]) + " " + str(patch["center"][2][0]) + " ")
        file.write(str(patch["normal"][0][0]) + " " +  str(patch["normal"][1][0]) + " " + str(patch["normal"][2][0]) + " ")
        file.write("255"+ " " + "0" + " "+"0")
        file.write("\n")

    file.close()



def get_boundaries():
    minimumX = np.math.inf
    minimumY = np.math.inf
    minimumZ = np.math.inf
    maximumX = - np.math.inf
    maximumY = - np.math.inf
    maximumZ = - np.math.inf
    for patch in patches:
        # Get the minimum center
        if patch["center"][0][0] < minimumX:
            minimumX = patch["center"][0][0]
        if patch["center"][1][0] < minimumY:
            minimumY = patch["center"][1][0]
        if patch["center"][2][0] < minimumZ:
            minimumZ = patch["center"][2][0]

        # get the maximum center
        if patch["center"][0][0] > maximumX:
            maximumX = patch["center"][0][0]
        if patch["center"][1][0] > maximumY:
            maximumY = patch["center"][1][0]
        if patch["center"][2][0] > maximumZ:
            maximumZ = patch["center"][2][0]
    print("minimum X: ",minimumX," maximum X: ",maximumX)
    print("minimum Y: ",minimumY," maximum Y: ",maximumY)
    print("minimum Z: ",minimumZ," maximum Z: ",maximumZ)

    return minimumX,minimumY,minimumZ,maximumX,maximumY,maximumZ



def isNeighbor(originalPatch,visibleImage, nPatch , horizontal):
    # x of the original patch
    # y of the original patch
    x = visibleImage['cell']['ptx'] * ß1
    y = visibleImage['cell']['pty'] * ß1

    difference = originalPatch['center'] - nPatch ['center'] 
    rightSide = abs (np.dot(np.squeeze(difference),np.squeeze(originalPatch['normal'])) )+ abs(np.dot(np.squeeze(difference),np.squeeze(nPatch['normal'])))
    
    cellStart = np.array([x,y,1]).reshape(3,1)
    projMat = imagesModels[visibleImage['idx']]["projMat"]
    ppinv = np.linalg.pinv(projMat)

    m = projMat[:,:3]
    p4 = projMat[:,3].reshape(3,1)
    minv = np.linalg.inv(m)

    b = cellStart - p4
    cellStart3D = np.matmul(minv,b) 
    cellStart3D = np.append(cellStart3D,1).reshape(4,1)

    optCenter = imagesModels[originalPatch['referenceImgIdx']]["optCenter"].reshape(4,1) 
    startRay = cellStart3D - optCenter
    startRay = startRay / np.linalg.norm(startRay)
    # end of the cell 
    if (horizontal):
        cellEnd= np.array([x+ß1, y ,1]).reshape(3,1)
    else:
        cellEnd= np.array([x, y+ß1 ,1]).reshape(3,1)
    
    b = cellEnd - p4
    cellEnd3D = np.matmul(minv,b) 
    cellEnd3D = np.append(cellEnd3D,1).reshape(4,1)

    endRay = cellEnd3D - optCenter
    endRay = endRay / np.linalg.norm(endRay)
    # Get the intersection of the start of the cell and the palne of the patch
    tStart = (- np.dot(np.squeeze(originalPatch['normal']),np.squeeze(cellStart3D- originalPatch['center'])))/(np.dot(np.squeeze(originalPatch['normal']),np.squeeze(startRay)))
    intersectionStart = tStart*startRay + cellStart3D

    scaleT = 1/intersectionStart[3]
    intersectionStart = intersectionStart * scaleT
    # Get the intersection of the end of the cell and the palne of the patch
    tEnd = (- np.dot(np.squeeze(originalPatch['normal']),np.squeeze(cellEnd3D- originalPatch['center'])))/(np.dot(np.squeeze(originalPatch['normal']),np.squeeze(endRay)))
    intersectionEnd = tEnd*endRay + cellEnd3D

    scaleT = 1/intersectionEnd[3]
    intersectionEnd = intersectionEnd * scaleT
    diff = abs(intersectionStart-intersectionStart)
    ρ = np.linalg.norm(diff)

    if( rightSide < 2*ρ ):
        return True
    else:
        return False



def get_neighbor_cells(originalPatch):
    neighborCells = []
    for visibleImage in originalPatch['visibleSet']:
        x = visibleImage['cell']['ptx']
        y = visibleImage['cell']['pty']
        if x == -1: #not included in the patch visible set 
            continue
        
        # Get neighbor cells
        for neighborY in range(y-1,y+2,1):
            for neighborX in range(x-1,x+2,1):
                # diagonal cells
                if (abs(neighborY-y) + abs(neighborX-x)) != 1:
                    continue

                if not outside_image_boundry(neighborY,neighborX,len(imagesModels[0]['image'])//ß1,len(imagesModels[0]['image'][0])//ß1):
                    neighborCell = imagesModels[visibleImage['idx']]['grid'][neighborY][neighborX]
                    # non empty cell Qt
                    
                    if len(neighborCell['Qt']) != 0:
                        continue

                    neighborExist = False
                    horizontal = False
                    if ( y == neighborY ):
                        horizontal = True
                    for nPatch in neighborCell['Qf']:

                        if ( isNeighbor (originalPatch,visibleImage, nPatch , horizontal) ):
                            neighborExist = True
                            break
                        # print("not neighbors")
                    if( neighborExist == False):
                        neighborCells.append({
                        'idx':visibleImage['idx'],
                        "x":neighborX*ß1, #cell center in img
                        "y":neighborY*ß1,
                        "neighborCell":neighborCell
                        })
    return neighborCells

def construct_expanded_patch(originalPatch,neighborCell):    
    newPatch = {}
    newPatch["referenceImgIdx"] = originalPatch["referenceImgIdx"]
    newPatch["normal"] = originalPatch["normal"]
    #newPatch["trueSet"] = originalPatch["trueSet"]

    # Get the ray
    #Back projection
    cellCenter = np.array([neighborCell['x'],neighborCell['y'],1]).reshape(3,1)
    projMat = imagesModels[neighborCell['idx']]["projMat"]
    ppinv = np.linalg.pinv(projMat)

    m = projMat[:,:3]
    p4 = projMat[:,3].reshape(3,1)
    minv = np.linalg.inv(m)

    b = cellCenter - p4
    cellCenter3D = np.matmul(minv,b) 
    cellCenter3D = np.append(cellCenter3D,1).reshape(4,1)

    optCenter = imagesModels[originalPatch['referenceImgIdx']]["optCenter"].reshape(4,1) 
    ray = cellCenter3D - optCenter
    ray = ray / np.linalg.norm(ray)

    # cellCenter3D = np.matmul(ppinv,cellCenter)
    # scaleCell = 1/cellCenter3D[3]
    # cellCenter3D = scaleCell * cellCenter3D
    # ray = cellCenter3D + imagesModels[originalPatch['referenceImgIdx']]["optCenter"].reshape(4,1)
    # normalize the ray
    
    if abs(np.dot(np.squeeze(ray),np.squeeze(newPatch['normal']))) < 10**-6:
        print("ray parallel to originalPatch")
        return None
    
    # Get the intersection
    t = (- np.dot(np.squeeze(originalPatch['normal']),np.squeeze(cellCenter3D- originalPatch['center'])))/(np.dot(np.squeeze(originalPatch['normal']),np.squeeze(ray)))
    intersection = t*ray + cellCenter3D

    scaleT = 1/intersection[3]
    intersection = intersection * scaleT

    # Check if the intersection in the bounding volume 
    if (intersection[0] > maximumX or intersection[0] < minimumX or intersection[1] > maximumY or intersection[1] < minimumY or intersection[2] > maximumZ or intersection[2] < minimumZ) :
        #print("baraaa")
        return None
    

    newPatch['center'] = intersection
    newPatch["visibleSet"] = originalPatch['visibleSet']
    newPatch['trueSet'] = get_t_images(newPatch,0.6,newPatch["visibleSet"])
    #-------
    if len(newPatch["trueSet"]) <= 1:
        return None
    
    newPatch['gStarScore'] = sum([tImg['gStarScore'] if tImg['idx'] != newPatch['referenceImgIdx'] else 0 for tImg in newPatch['trueSet']])
    newPatch['gStarScore'] /= (len(newPatch['trueSet'])-1)
    #optimize_patch(newPatch)
    #newPatch['trueSet'] = get_t_images(newPatch,0.3,newPatch["visibleSet"])

    #for newVImg in newVImgs:
     #   found = False
      #  for vImg in newPatch["visibleSet"]:
       #     if newVImg['idx'] == vImg['idx']:
        #        found = True
         #       break

        #if not found:
         #   newPatch["visibleSet"].append(newVImg)
    
    #visibleIdxs = [vImg['idx'] for vImg in newPatch["visibleSet"]]
    #newPatch["trueSet"] = get_t_images(newPatch,0.7,visibleIdxs)
    #------
    return newPatch



def checkEqualPatches(patch,nPatch):
    if (patch['referenceImgIdx'] == nPatch['referenceImgIdx'] 
    and (patch['normal'] == nPatch['normal']).all() 
    and (patch['center'] == nPatch['center']).all() 
    and patch['visibleSet'] == nPatch['visibleSet'] 
    and patch['trueSet'] == nPatch ['trueSet'] 
    and patch['gStarScore'] == nPatch['gStarScore']):
        return True
    else:
        return False



def init():
    constants(dataPath)
    images,grids = init_imgs(datasetPath)
    projections,optAxes = read_parameters_file(datasetPath)
    print("Read Input---->DONE")
    imagesModels = list()

    for idx,image in enumerate(images):
        dog,harris = get_dog_harris(image)
        sparseDog,sparseHarris,dogPositions,harrisPositions = sparse_dog_harris(dog,harris)
        opticalCenter = getOpticalCenter(projections[idx])
        imgModel={
            "image": images[idx],
            "projMat": projections[idx],
            "optCenter": opticalCenter,
            "optAxis": optAxes[idx],
            "grid": grids[idx],
            "dog": dog,
            "harris": harris,
            "sparseDog": sparseDog,
            "sparseHarris": sparseHarris,
            "dogPositions": dogPositions,
            "harrisPositions": harrisPositions
        }
        print("ImageID:", str(idx),"\tharris:",str(len(harrisPositions)),"\tDoG:", str(len(dogPositions)))
        imagesModels.append(imgModel)

    print("Feature Detection---->DONE")

    for i in range(len(imagesModels)):
        imagesModels[i]["releventImgsIdxs"] = get_relevent_images(imagesModels,i)

    print("Get Relevent Images---->DONE")
    # show_images([imagesModels[0]["dog"],imagesModels[0]["sparseDog"],imagesModels[0]["harris"],imagesModels[0]["sparseHarris"]],['dog','sparse dog','harris','sparse harris'])



def matching():
    print("Start Matching....")
    patches = list()
    numberOfPatches = 0
    print("Total number of patches: ", len(patches))
    for i in range(len(imagesModels)):
        baseImageIdx = i
        completeCell = np.zeros((len(imagesModels[baseImageIdx]['image'])//ß1,len(imagesModels[baseImageIdx]['image'][0])//ß1))
        featureTypes = ["harrisPositions",'dogPositions']
        for featureType in featureTypes:
            #if featureType == 'harrisPositions':
               # continue
            for featurePt in imagesModels[baseImageIdx][featureType]:
                if completeCell[featurePt[1]//ß1][featurePt[0]//ß1]:
                    continue
                if not(empty_cell(baseImageIdx, featurePt[1], featurePt[0])):
                    continue

                features  = get_features_statsify_epipoler_consistency(baseImageIdx, featurePt,featureType)
                construct_patches(baseImageIdx, features)
                completeCell[featurePt[1]//ß1][featurePt[0]//ß1] = 1
            print("ImageID:", str(baseImageIdx),featureType,"Number of Features:", str(len(imagesModels[baseImageIdx][featureType])),"Done-->Number of constructed patches:", str(len(patches) - numberOfPatches))
            numberOfPatches = len(patches)
        print()
    print("Total number of patches:", len(patches))
    originalImageModels = deepcopy(imagesModels)
    originalPatches = deepcopy(patches)



def expansion():
    print("Start Expansion....")
    patches = deepcopy(originalPatches)
    totalPatches = deepcopy(originalPatches)
    patchesStack = deepcopy(originalPatches)
    expandedPatches = []
    imagesModels = deepcopy(originalImageModels)
    minimumX,minimumY,minimumZ,maximumX,maximumY,maximumZ = get_boundaries()
    print("Total number of patches: ", len(patches))
    while len(patchesStack) != 0:
        print("The total number of patches now:",len(totalPatches),"\tremaining patches:",len(patchesStack))
        patch = patchesStack.pop(0)
        neighborCells = get_neighbor_cells(patch)

        for neighborCell in neighborCells:
            newPatch = construct_expanded_patch(patch,neighborCell)
            if newPatch is None:
                continue

            if len(newPatch["trueSet"]) >= gamma:
                register_patch(newPatch)
                expandedPatches.append(newPatch)
                totalPatches.append(newPatch)



def recursive_expansion():
    idx = 1
    totTim = time()
    while(len(expandedPatches)!=0):
        print("Pass"+str(idx)+":")
        patchesStack = deepcopy(expandedPatches)
        expandedPatches = []
        print("Total number of patches: ", len(patches),"\tremaining patches:",len(patchesStack))
        tim = time()
        while len(patchesStack) != 0:
            print("The total number of patches now:",len(patches),"\tremaining patches:",len(patchesStack))
            patch = patchesStack.pop(0)
            neighborCells = get_neighbor_cells(patch)
            for neighborCell in neighborCells:    
                newPatch = construct_expanded_patch(patch,neighborCell)
                if newPatch is None:
                    continue

                if len(newPatch["trueSet"]) >= gamma:
                    register_patch(newPatch)
                    expandedPatches.append(newPatch)
        print("Pass time:",time()-tim)
        idx += 1
    print("Total time:",time()-totTim)



def filter1():
    print("Visibility Consistency Filter......")
    for patch in patches:
        for trueImg in patch['trueSet']:
            x = trueImg['cell']['ptx']
            y = trueImg['cell']['pty']
            if x == -1: #not included in the patch visible set 
                continue
            Qt = imagesModels[trueImg['idx']]['grid'][y][x]['Qt']
            if len(Qt) <= 1:
                continue

            otherGStarScore = 0
            for otherPatch in Qt:
                if otherPatch == patch:
                    continue
                if not isNeighbor(patch,otherPatch):
                    otherGStarScore += otherPatch['gStarScore']

            print("OutlierPatch",patch['gStarScore']*patch['trueSet'],otherGStarScore)
            if patch['gStarScore']*patch['trueSet'] < otherGStarScore:
                # outlier patch
                patch['isOutlier'] = True
                
def filter2():
    print("Regularization Filter......")
    for index,patch in enumerate(patches):
        adjacentPatches = []
        neighborPatches = []

        for visibleImage in patch['visibleSet']:
            x = visibleImage['cell']['ptx']
            y = visibleImage['cell']['pty']
            if x == -1: #not included in the patch visible set 
                continue

            for neighborY in range(y-1,y+2,1):
                for neighborX in range(x-1,x+2,1):
                    # diagonal cells
                    if (abs(neighborY-y) + abs(neighborX-x)) == 2:
                        continue
                    if not outside_image_boundry(neighborY,neighborX,len(imagesModels[0]['image'])//ß1,len(imagesModels[0]['image'][0])//ß1):
                        neighborCell = imagesModels[visibleImage['idx']]['grid'][neighborY][neighborX]
                        horizontal = False
                        if ( y == neighborY ):
                            horizontal = True
                        count = 0
                        for nPatch in neighborCell['Qt']:
                            if checkKey(nPatch,'isOutlier') and nPatch['isOutlier'] == True:
                                continue

                            if not checkEqualPatches(patch,nPatch):
                                count += 1
                                if ( isNeighbor (patch,visibleImage, nPatch , horizontal) ):
                                    neighborPatches.append(nPatch)
                                adjacentPatches.append(nPatch)
                        count = 0 
                        for nPatch in neighborCell['Qf']:
                            if checkKey(nPatch,'isOutlier') and nPatch['isOutlier'] == True:
                                continue

                            if not checkEqualPatches(patch,nPatch):
                                count += 1
                                if ( isNeighbor (patch,visibleImage, nPatch , horizontal) ):
                                    neighborPatches.append(nPatch)
                                adjacentPatches.append(nPatch)

        if len(adjacentPatches) == 0:
            print("adjacent cell is zero")
            print("length of neighbor: ",len(neighborPatches))
            print("x: ",x)
            print("y: ",y) 
        if len(adjacentPatches) > 0 and len(neighborPatches)/len(adjacentPatches) < 0.25:
            patch['isOutlier'] = True
        if index % 1000 == 0:
            print("index: ",index)
            
def filtering():
    print("Start Filtering....")
    filter1()
    filter2()

def main(dataPath):
    init()
    matching()
    expansion()