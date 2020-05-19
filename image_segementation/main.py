import numpy as np
from matplotlib import image
import os
from sklearn.model_selection import train_test_split
import pickle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
train_ratio = 0.8

base_dir = 'data'
model_path = 'model_1.pkl'
# load images
# scene = image.imread(os.path.join(base_dir, 'complete.jpg'))
# cow_class = image.imread(os.path.join(base_dir, 'cow.jpg'))
# sky_class = image.imread(os.path.join(base_dir, 'sky.jpg'))
# grass_class = image.imread(os.path.join(base_dir, 'grass.jpg'))


# cow_train, cow_test  = train_test_split(cow_class, test_ratio)

# sky_train, sky_test  = train_test_split(sky_class, test_ratio)

# grass_train, grass_test  = train_test_split(grass_class, test_ratio)


def load_data(base_path):
    lables = []
    data = []
    for lable in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, lable)):
            for img in os.listdir(os.path.join(base_path, lable)):
                img = image.imread(os.path.join(base_path, lable, img))
                for row in img:
                    for col in row:
                        data.append([col])
                        lables.append(lable)

    unique_lables = ['cow', 'grass', 'sky']
    categorical_lables = [i for i in unique_lables]
    lables = [categorical_lables.index(lable) for lable in lables]
    lables = np.array(lables)
    # lables = lables.reshape((lables.size, 1))
    data = np.array(data)
    # data = np.append(data, lables, axis=1)
    return data, lables


data, lables = load_data(base_dir)

train_data, test_data, train_lables, test_lables = train_test_split(data, lables, train_size=train_ratio)

# kernel possiblities = linear, poly, rbf, sigmoid, precomputed
model = SVC(C=0.8, kernel='rbf')
model.fit(train_data, train_lables)

# save model
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# load_model
# with open(model_path, 'rb') as f:
#     model = pickle.load(f)

predictions = model.predict(test_data)

# build confusion matrix
possible_labels = set(test_lables)
possible_labels = list(possible_labels)

conf_matrix = confusion_matrix(
    test_lables, predictions, labels=possible_labels)

df_cm = pd.DataFrame(conf_matrix, index=list(possible_labels), columns=list(possible_labels))
sn.heatmap(df_cm, annot=True)

plt.show()




print('done')
