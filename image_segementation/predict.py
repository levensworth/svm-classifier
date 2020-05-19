import pickle
from matplotlib import image
import numpy as np

model_path = 'model_poly.pkl'
img_path = 'data/complete.jpg'

class_color = [
    [255, 0,0],
    [0, 255, 0],
    [0, 0, 255]
]

# load model
with open(model_path, 'rb') as f:
    model = pickle.load(f)


img = image.imread(img_path)

predictions = []
for row in img:
    row_pred = []
    for col in row:
        pred = model.predict(np.array([col]))
        row_pred.append(class_color[int(pred)])
    predictions.append(row_pred)

predictions = np.array(predictions)
predictions = predictions.astype('uint8')
image.imsave('output.jpg', predictions)
