# -*- coding: utf-8 -*-

from google.colab import files
uploaded = files.upload()

from google.colab import files
uploaded = files.upload()

!pip install python-resize-image
from glob import glob
from PIL import Image
from resizeimage import resizeimage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

pokemons = glob('images/*')

for i in pokemons: 
    if i[-3:] == 'jpg': 
        im = Image.open(i)
        im.save(f'{i[:-3]}png')

def resize(image_list): 
    image_list = sorted(image_list)
    flattened = []
    for path in image_list:
        if path[-3:] != 'jpg': 
            # open it as a read file in binary mode
            with open(path, 'r+b') as f:
                # open it as an image
                with Image.open(f) as image:
                    # convert to same color scale
                    image = image.convert("RGB") 
                    # flatten the matrix to an array and append it to all flattened images
                    flattened.append(np.array(image).flatten())
        
    # Use np.stack to put the data into the right dimension
    X = np.stack(i for i in flattened)

    return X

pokemonspca = resize(pokemons)

!unzip -o images.zip

import pandas as pd
import matplotlib

pca = PCA(3)
transf = pca.fit_transform(pokemonspca)
df = pd.DataFrame(transf)

cols = [f"PC{i+1}" for i in range(len(df.columns))]
df.columns = cols 

types = pd.read_csv("pokemon.csv")
types = types.sort_values("Name", ignore_index=True)

# type = grass vs others

grass_type = []
for i in types['Type1']:
  if i == 'Grass':
    grass_type.append(1)
  else:
    grass_type.append(0)

df["type"] = grass_type
print(df.head())

# use this line to only display datapoints for type X
#df = df.loc[df["type"]=="Flying"]

levels, categories = pd.factorize(df['type'])
colors = [plt.cm.tab20(i) for i in levels] 
handles = [matplotlib.patches.Patch(color=plt.cm.tab20(i), label=c) for i, c in enumerate(categories)]

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
ax.scatter(df['PC1'], df['PC2'], df['PC3'], c=colors)
plt.gca().set(xlabel='PC1', ylabel='PC2',zlabel='PC3', title='Pokemon types PCA')
plt.legend(handles=handles, title='Types')

# this is after PCA with 3 dimensions!!!

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix,classification_report

X = df.drop("type", axis=1)
y = df["type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# train multi-class SVM
# default gamma: 0.2857
# changing C doesn't change the performance on test data; changing gamma can bring training data score to 1.0 from 0.9
svm_model = SVC()
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)


print('Score for training data:', svm_model.score(X_train, y_train)) 
print('Score for test data:', svm_model.score(X_test, y_test)) 
print(confusion_matrix(y_test,y_pred))

# this is the FULL data

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import plot_confusion_matrix,classification_report


X_train, X_test, y_train, y_test = train_test_split(pokemonspca, y, test_size=0.20, random_state=42)

# train multi-class SVM
# default C was 1, changed to 5 -- more overfitting
svm_full = SVC(C=5)
svm_full.fit(X_train, y_train)

print('Score for training data:', svm_full.score(X_train, y_train)) 
print('Score for test data:', svm_full.score(X_test, y_test)) 
print(plot_confusion_matrix(svm_full, X_test, y_test))

