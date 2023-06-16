from ast import Break
import os
from unicodedata import category 
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import pickle 
import random
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
# from sklearn.model selection 
# import train_test
# dir = 'C:\\Users\\Md. Iqbal Hossain\\Desktop\\machine\\15\\kagglecatsanddogs_3367a\\PetImages'
dir = 'D:\\3rd Year Mini Project\\Implementation\\dataset\\dataset\\resized'
# categories = ['Cat', 'Dog']
# categories=['alfalfa',
# 'allium',
# 'borage',
# 'burdock',
# 'calendula',
# 'cattail',
# 'chickweed',
# 'chicory',
# 'chive_blossom',
# 'coltsfoot',
# 'common_mallow',
# 'common_milkweed',
# 'common_vetch',
# 'common_yarrow',
# 'coneflower',
# 'cow_parsley',
# 'cowslip',
# 'crimson_clover',
# 'crithmum_maritimum',
# 'daisy',
# 'dandelion',
# 'fennel',
# 'fireweed',
# 'gardenia',
# 'garlic_mustard',
# 'geranium',
# 'ground_ivy',
# 'harebell',
# 'henbit',
# 'knapweed',
# 'meadowsweet',
# 'mullein',
# 'pickerelweed',
# 'ramsons',
# 'red_clover']
data = []
for category in categories: 
    path = os.path.join(dir, category) 
    label= categories.index(category)

    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        pet_img=cv2.imread(imgpath,0)
        # cv2.imshow('image',pet_img)
        try:
            pet_img=cv2.resize(pet_img,(50,50))
            image = np.array(pet_img).flatten()

            data.append([image,label])
        except Exception as e:
            pass

# print(len(data))
        # break
    # break

# cv2.waitKey(0)
# cv2.destroyAllWindows()


pick_in = open('data1.pickle', 'wb') 
pickle.dump(data,pick_in)
pick_in.close()


pick_in = open('data1.pickle', 'rb') 
data=pickle.load(pick_in)
pick_in.close()


random.shuffle(data) 
features = [] 
labels = []
for feature , label in data:
    features.append(feature) 
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size= 0.25)

model = SVC(C=1, kernel='poly', gamma='auto')
model.fit(xtrain, ytrain)

prediction=model.predict(xtest)
accuracy = model.score(xtest, ytest)

# categories = ['Cat', 'Dog']
categories=['alfalfa',
'allium',
'borage',
'burdock',
'calendula',
'cattail',
'chickweed',
'chicory',
'chive_blossom',
'coltsfoot',
'common_mallow',
'common_milkweed',
'common_vetch',
'common_yarrow',
'coneflower',
'cow_parsley',
'cowslip',
'crimson_clover',
'crithmum_maritimum',
'daisy',
'dandelion',
'fennel',
'fireweed',
'gardenia',
'garlic_mustard',
'geranium',
'ground_ivy',
'harebell',
'henbit',
'knapweed',
'meadowsweet',
'mullein',
'pickerelweed',
'ramsons',
'red_clover']


print('accuracy: ', accuracy)

print('Prediction is : ', categories[prediction[0]])

mypet=xtest[0].reshape(50,50)
plt.imshow(mypet, cmap='grey')
plt.show()

