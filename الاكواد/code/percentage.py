import pandas as pd
import numpy as np
import cv2
import warnings
import os
import pickle
import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from os import listdir
from os.path import isfile, join

warnings.filterwarnings('ignore')

# get the path of the images
imagePath = '../input/leukemia-dataset/ALL_IDB1/ALL_IDB1/im'
imagesfiles = [f for f in listdir(imagePath) if isfile(join(imagePath, f))]
path_list = list()
class_list = list()
for image in imagesfiles:
    # print(image)
    image_class = image.split('_')[-1].split('.')[0]
    path_list.append(image)
    # print(path_list)
    class_list.append(int(image_class))
    # print(image_class)
# store the path and class of each one into pandas dataframe
data = pd.DataFrame({'image path': path_list, 'class': class_list})
data.head()

# get count of the values
data.count()

# get count of the values each class
fig = plt.figure()
label = ['class 1', 'class 0']
data_count = [len(data[data['class'] == 1].index), len(data[data['class'] == 0].index)]
plt.bar(label, data_count)
plt.show()


# USE LBP for feature extraction
def get_pixel(img, center, x, y):
    new_value = 0
    try:
        # If local neighbourhood pixel
        # value is greater than or equal
        # to center pixel values then
        # set it to 1
        if img[x][y] >= center:
            new_value = 1
    except:
        pass

    return new_value

# Function for calculating LBP
def lbp_calculated_pixel(img, x, y):
    center = img[x][y]

    val_ar = []

    # top_left
    val_ar.append(get_pixel(img, center, x - 1, y - 1))

    # top
    val_ar.append(get_pixel(img, center, x - 1, y))

    # top_right
    val_ar.append(get_pixel(img, center, x - 1, y + 1))

    # right
    val_ar.append(get_pixel(img, center, x, y + 1))

    # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))

    # bottom
    val_ar.append(get_pixel(img, center, x + 1, y))

    # bottom_left
    val_ar.append(get_pixel(img, center, x + 1, y - 1))

    # left
    val_ar.append(get_pixel(img, center, x, y - 1))

    # Now, we need to convert binary
    # values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val

# LBP
def LBP(img_bgr):
    height, width, _ = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_lbp = np.zeros((height, width), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    return img_lbp

# apply the LBP for each image
def create_dataset(data):
    path = "../input/leukemia-dataset/ALL_IDB1/ALL_IDB1/im"
    img_data_array = []
    class_name = []
    text_data = []
    dim = (160, 160)
    for index, file in data[['image path']].iterrows():
        img_path = (file.values[0])
        image_path = os.path.join(path, img_path)
        image = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        image = np.array(image)
        image = image.astype('float32')
        image /= 255
        image = LBP(image)
        img_data_array.append(image.flatten())
    return np.array(img_data_array)


# get the LBP of each image
img_data_array = create_dataset(data)

# split the data into train and test
split_number = int((len(data) * 90) / 100)
data_Y = data.loc[:, ['class']]
train_X = img_data_array[:split_number]
train_Y = data_Y[:split_number]
test_X = img_data_array[split_number:]
test_Y = data_Y[split_number:]

#----------------------------------------------------------SVM----------------------------------------------------------
param_grid = {'C': [0.1, 1, 100, 1000],
              'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
              'degree': [1, 2, 3, 4, 5, 6],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid)
grid.fit(train_X, train_Y)
y_pred = grid.predict(test_X)
print("------------------------------------SVM Result------------------------------------")
print("SVM accuracy_score:", accuracy_score(test_Y, y_pred))
print("SVM precision_score:", precision_score(test_Y, y_pred))
print("SVM recall_score:", recall_score(test_Y, y_pred))
print("SVM f1_score:", f1_score(test_Y, y_pred))
SVN_accuracy = accuracy_score(test_Y, y_pred)
SVN_preceision = precision_score(test_Y, y_pred)
SVN_recall = recall_score(test_Y, y_pred)
SVN_f1 = f1_score(test_Y, y_pred)
with open('leukemia_SVM.pickle', 'wb') as f1:
    pickle.dump(grid, f1)


#----------------------------------------------------------KNN----------------------------------------------------------
knn_model = KNeighborsClassifier()
knn_model.fit(train_X, train_Y)
y_pred = knn_model.predict(test_X)

print("------------------------------------KNN Result------------------------------------")
print("KNN accuracy_score:", accuracy_score(test_Y, y_pred))
print("KNN precision_score:", precision_score(test_Y, y_pred))
print("KNN recall_score:", recall_score(test_Y, y_pred))
print("KNN f1_score:", f1_score(test_Y, y_pred))
KNN_accuracy = accuracy_score(test_Y, y_pred)
KNN_preceision = precision_score(test_Y, y_pred)
KNN_recall = recall_score(test_Y, y_pred)
KNN_f1 = f1_score(test_Y, y_pred)
with open('leukemia_KNN.pickle', 'wb') as f1:
    pickle.dump(knn_model, f1)


#----------------------------------------------------------NB-----------------------------------------------------------
NB_model = GaussianNB()
NB_model.fit(train_X, train_Y)
y_pred = NB_model.predict(test_X)

print("------------------------------------NB Result-------------------------------------")
print("NB accuracy_score:", accuracy_score(test_Y, y_pred))
print("NB precision_score:", precision_score(test_Y, y_pred))
print("NB recall_score:", recall_score(test_Y, y_pred))
print("NB f1_score:", f1_score(test_Y, y_pred))
NB_accuracy = accuracy_score(test_Y, y_pred)
NB_preceision = precision_score(test_Y, y_pred)
NB_recall = recall_score(test_Y, y_pred)
NB_f1 = f1_score(test_Y, y_pred)
with open('leukemia_NB.pickle', 'wb') as f1:
    pickle.dump(NB_model, f1)

#-------------------------------------------------Drawing the bar chart-------------------------------------------------
# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize=(12, 8))
# set height of bar
SVN = [SVN_accuracy, SVN_preceision, KNN_preceision, SVN_f1]
NV = [NB_accuracy, NB_preceision, NB_recall, NB_f1]
KNN = [KNN_accuracy, KNN_preceision, KNN_recall, KNN_f1]
# Set position of bar on X axis
br1 = np.arange(len(SVN))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
metrics = ['Accuracy', 'Preceision', 'Recall', 'F1-SCORE']
ALGORITHMS = ['SVM', 'KNN', 'NAIVE BAYES']
# Make the plot
plt.bar(br1, SVN, color='yellow', width=barWidth, edgecolor='grey', label=ALGORITHMS[0])
plt.bar(br2, KNN, color='orange', width=barWidth, edgecolor='grey', label='KNN')
plt.bar(br3, NV, color='b', width=barWidth, edgecolor='grey', label='NAIVE BAYES')
# Adding Xticks
plt.xlabel('Metrics', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(metrics))], metrics)

plt.legend()
plt.show()
