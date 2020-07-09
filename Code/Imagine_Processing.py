# Imagine processing
# @ Michael

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


# creat a 6 by 6 images
# while - (255, 255, 255)
# black - (1, 1, 1)
# create a pure while picture
img = np.ones((6, 6, 3), dtype=int)*255
plt.imshow(img)  # it is just a white canvars
plt.axis('off')

# add vertical edges
img[:, 3:4, :] = np.ones((6, 1, 3), dtype=int)

# check the image
plt.imshow(img)


img[:, :, 0]

# array([[  1, 255,   1, 255,   1, 255],
#        [  1, 255,   1, 255,   1, 255],
#        [  1, 255,   1, 255,   1, 255],
#        [  1, 255,   1, 255,   1, 255],
#        [  1, 255,   1, 255,   1, 255],
#        [  1, 255,   1, 255,   1, 255]])


def matrixConv(kenl, dtm):
    # Assume size of kenel is less than datamatrix
    # 假设参数核矩阵小于数据矩阵
    m = np.shape(kenl)[0]
    n = np.shape(kenl)[1]
    k = np.shape(dtm)[0]
    h = np.shape(dtm)[1]
    resltmtx = np.zeros([k-m+1, h-n+1])
    for i in range(resltmtx.shape[0]):
        for j in range(resltmtx.shape[1]):
            temp = 0
            for u in range(m):
                for p in range(n):
                    temp += kenl[u, p] * dtm[u+i, p+j]
                    resltmtx[i, j] = temp
    return(resltmtx)


Kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, 1]], dtype=int)
Ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)


img2 = np.ones((4, 4, 3), dtype=int)
for i in range(3):
    img2[:, :, i] = matrixConv(Kx, img[:, :, i]) % 255
    img2[:, :, i].astype(int)

plt.imshow(img2)

# now, we add the horizontal edge
img3 = np.ones((6, 6, 3), dtype=int)*255
plt.imshow(img3)  # it is just a white canvars
img3[3:4, :, :] = np.ones((1, 6, 3), dtype=int)
plt.imshow(img3)

img4 = np.ones((4, 4, 3), dtype=int)
for i in range(3):
    img4[:, :, i] = matrixConv(Ky, img3[:, :, i]) % 255
    img4[:, :, i].astype(int)

plt.imshow(img4)


# Padding
img5 = np.zeros((8, 8, 3), dtype=int)  # 8 by 8 matrix
img5[1:-1, 1:-1, :] = img
plt.imshow(img5)

# convolution after padding
img6 = np.ones((6, 6, 3), dtype=int)
for i in range(3):
    img6[:, :, i] = matrixConv(Kx, img5[:, :, i]) % 255
    img6[:, :, i].astype(int)

plt.imshow(img6)































#
