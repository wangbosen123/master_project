import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# image = cv2.imread("face.png",cv2.IMREAD_GRAYSCALE)
# image = np.expand_dims(image,axis=-1)
# image[50:110, 15:110,:] = np.random.uniform(0, 255, size=(110 - 50, 110 - 15, 1))
# cv2.imwrite("occ_face.jpg",image)

image = cv2.imread("mydog.jpg")
print(image.shape)
image[0:1108, 0:1447,:] = np.random.uniform(0,255,size=(1108,1447,1))
cv2.imwrite("occ2_mydog.jpg",image)

x = [1,2,3,4,5,6]
b = [7,8,9,0,1,4]
c = [4,5,6,7,8,9]

data = {"k" : [i for i in range(len(x))],
        "psnr" : x,
        "mse" : b,
        "cr" : c}

df = pd.DataFrame(data)
print(df)
df.to_csv("data",index=False, sep="\t")