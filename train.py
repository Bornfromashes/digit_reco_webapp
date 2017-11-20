
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.externals import joblib
from PIL import Image
import sys


# In[7]:


digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)


# In[22]:


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
classifier = svm.SVC(gamma=0.001)
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


# In[15]:


images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)



# In[24]:


if len(sys.argv) != 2:
    print("Incorrect number of arguments, add a BMP file as cmd line argument.n")
    sys.exit()
custom_IM = Image.open(sys.argv[1])
custom_pixels = list(custom_IM.getdata())
corr_pixels = []
for row in custom_pixels:
    new_row = 255 - row[0]
    corr_pixels.append(new_row)
if len(corr_pixels) != 784:
    print("Incorrect Image Dimensions (needs to be 784)n")
    sys.exit()
test_set = np.array(corr_pixels)

classifier = joblib.load("../classifier/kaggle_digit_recognizer.pkl")
predicted = classifier.predict(test_set)
print(predicted)


