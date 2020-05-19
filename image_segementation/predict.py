import pickle
from matplotlib import image
import numpy as np

model_path = 'model_1.pkl'
img_path = 'total.jpg'

# load model
with open(model_path, 'rb') as f:
    model = pickle.load(f)


img = image.imread(img_path)

data = []
for row in img:
    for col in row:
        data.append(col)
data = np.array(data)

predictions = model.predict(data)
