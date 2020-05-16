from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from functions import *
from skimage import morphology, measure, filters
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
import matplotlib.pyplot as pyplot
from skimage.feature import greycomatrix, greycoprops
from sklearn.neural_network import MLPClassifier
from sklearn import cross_validation
import pickle

path = "featuresdicom.xlsx"
INPUT_SCAN_FOLDER = "G:\\final\\dicom final\\database\\malignant\\LIDC-IDRI-0072\\"

fileDICOMFeatureList = pd.read_excel(path, header=None)

matrixFeatures = np.array((fileDICOMFeatureList.as_matrix())[1:, :])
yMatrixFeatures = matrixFeatures[:, 7]
xMatrifFeatures = matrixFeatures[:, 0:7]
X_train, X_test, y_train, y_test = train_test_split(xMatrifFeatures, yMatrixFeatures, test_size=0.2, random_state=10)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# clf = MLPClassifier(hidden_layer_sizes=1000,solver='lbfgs')
#
# clf.fit(X_train, y_train)
modelFileMLP = 'mlpmodel.sav'
# pickle.dump(clf,open(filename,'wb'))

modelMLP = pickle.load(open(modelFileMLP, 'rb'))
# print(clf)
MLPscore = modelMLP.score(X_train, y_train)
MLPtest = modelMLP.predict(X_test)

print('MLP training=', MLPscore * 100)
print("MLP testing accuracy=", np.mean(MLPtest == y_test) * 100)
KNNmodel = KNeighborsClassifier()
kfold = cross_validation.KFold(n=len(X_train), n_folds=10, random_state=10)
cv_results = cross_validation.cross_val_score(KNNmodel, X_train, y_train, cv=kfold, scoring='accuracy')
message = "%s: %f " % ("KNN cross validation accuracy", cv_results.mean())
print(message)
K_value = 3
neigh = KNeighborsClassifier(n_neighbors=K_value, weights='uniform', algorithm='auto')
neigh.fit(X_train, y_train)
KNNpredictValue = neigh.predict(X_test)
print("KNN testing accuracy=", np.mean(KNNpredictValue == y_test) * 100)

listProperties = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy']
listFeatures = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'mean', 'stddev', 'label']
properties = np.zeros(6)

# glcmMatrix = []
final = []
arrayOriginalImages = dicomRead(INPUT_SCAN_FOLDER)
test3D(ConstPixelDims = arrayOriginalImages)
tumorArea = []
arrayTumorContour = []
for z in range(125, 180):

    tempImageSlice = arrayOriginalImages[z][:][:]
    # img=img.pixel_array
    #imgg = tempImageSlice
    tempImageMask = segment(tempImageSlice)
    tempImageMask = np.where(tempImageMask == 255, 1, 0)
    # pyplot.imshow(tempImageMask, cmap='gray')
    # pyplot.show()
    tempImageConvMask = tempImageMask * tempImageSlice
    tempImageConvMask = (tempImageConvMask / 256).astype('uint8')
    ImageConvMask = tempImageConvMask
    tempImageSliceMean = arrayOriginalImages[z][:][:].mean()
    tempImageSliceStdDev = arrayOriginalImages[z][:][:].std()
    glcmMatrix = (greycomatrix(tempImageConvMask, [1], [0], levels=2 ** 8))

    for j in range(0, len(listProperties)):
        properties[j] = (greycoprops(glcmMatrix, prop=listProperties[j]))

    arrayFeatureValues = np.array([[properties[0], properties[1], properties[2], properties[3], properties[4], tempImageSliceMean, tempImageSliceStdDev]])

    # pyplot.imshow(imgg,cmap='gray')
    # pyplot.show()
    # df = pd.DataFrame(final, columns=listFeatures)

    y_pred = neigh.predict(arrayFeatureValues)
    tempSegmentedImage = tempImageConvMask
    print(y_pred)
    if (y_pred == 2 or y_pred == 1):
        segmented1 = tempSegmentedImage

        tempSegmentedImageMean = np.mean(tempSegmentedImage)
        tempSegmentedImageStdDev = np.std(tempSegmentedImage)
        segmentedImage = tempSegmentedImage - tempSegmentedImageMean
        segmentedImage = tempSegmentedImage / (tempSegmentedImageStdDev + 0.00001)
        # pyplot.imshow(imgg,cmap='gray')
        # pyplot.show()
        # hist = pyplot.hist(segmented.flatten(), bins=200)

        ROI = segmentedImage[100:400, 100:400]
        ROImean = np.mean(ROI)
        ROImaxv = np.max(tempSegmentedImage)
        ROIminv = np.min(tempSegmentedImage)
        tempSegmentedImage[tempSegmentedImage == ROImaxv] = tempSegmentedImageMean
        tempSegmentedImage[tempSegmentedImage == ROIminv] = tempSegmentedImageMean
        ROIkmeans = KMeans(n_clusters=3).fit(np.reshape(ROI, [np.prod(ROI.shape), 1]))
        ROIkmeanscenters = sorted(ROIkmeans.cluster_centers_.flatten())
        ROIkmeansthreshold = np.mean(ROIkmeanscenters)
        threshROIImg = np.where(segmentedImage >= ROIkmeansthreshold, 1.0, 0.0)
        threshROIImg = morphology.erosion(threshROIImg, np.ones([9, 9]))
        threshROIImg = morphology.dilation(threshROIImg, np.ones([9, 9]))
        # pyplot.imshow(threshROIImg, cmap='gray')
        # pyplot.show()
        tumorContours = measure.find_contours(threshROIImg, 0.8)

        # Display the image and plot all contours found
        tempTumorArea = []
        if (tumorContours):
            contourLabels = label(threshROIImg)
            contourRegions = regionprops(contourLabels, threshROIImg)
            arrayTumorContour.append(tumorContours)
            tempTumorArea = (tempTumorArea.append(contourRegions[i].area) for i in range(len(contourRegions)))
            tempTumorArea = (contourRegions[0].area)
            tumorArea.append(tempTumorArea)
            fig, ax = pyplot.subplots()
            ax.imshow(tempImageSlice, interpolation='nearest', cmap=pyplot.cm.gray)

            for n, singleContour in enumerate(tumorContours):
                ax.plot(singleContour[:, 1], singleContour[:, 0], linewidth=2)

            ax.axis('image')
            ax.set_xticks([])
            ax.set_yticks([])
            # threshROIImg = threshROIImg * imgg
            pyplot.imshow(tempImageSlice, cmap='gray')
            pyplot.show()


            if (y_pred == 1):
                print(str(z) + ' Image is tumorous')
                print(tempTumorArea.max())
                # if(tempTumorArea<Put area here):
                # elif(areaa<Put area here):
                # elif(areaa < Put area here):
            elif (y_pred == 2):
                print(str(z) + ' Image is tumorous')
                print(tempTumorArea.max())
                # if(areaa<Put area here):
                # elif(areaa<Put area here):
                # elif(areaa < Put area here):
        else:
            print(str(z) + ' Image is non tumorous')
    else:
        print(str(z)+' Image is non tumorous')

if (len(tumorArea)):
    volume = 0;
    for i in range(0, len(tumorArea) - 1):
        if (i == 0):
            volume = volume + (((tumorArea[i] + 0) * 1.25) / 2)
        else:
            volume = volume + (((tumorArea[i] + tumorArea[i - 1]) * 1.25) / 2)
    print(volume)


