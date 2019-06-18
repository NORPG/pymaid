import cv2
import numpy as np,sys
import hwconv


it = 6
A = cv2.imread('apple.jpg', cv2.IMREAD_GRAYSCALE)
B = cv2.imread('orange.jpg', cv2.IMREAD_GRAYSCALE)
# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]

for i in xrange(it):
  G = cv2.copyMakeBorder(G,2,2,2,2,cv2.BORDER_REFLECT)
  G = hwconv.pyrDown(G)
  gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]

for i in xrange(it):
  G = cv2.copyMakeBorder(G,2,2,2,2,cv2.BORDER_REFLECT)
  G = hwconv.pyrDown(G)
  gpB.append(G)

# generate Laplacian Pyramid for A
lpA = [gpA[it-1]]
for i in xrange((it-1),0,-1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[it-1]]
for i in xrange((it-1),0,-1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols = la.shape
    ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in xrange(1,(it)):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

# image with direct connecting each half
real = np.hstack((A[:,:cols/2],B[:,cols/2:]))
cv2.imwrite('Pyramid_blending_hw.jpg',ls_)
