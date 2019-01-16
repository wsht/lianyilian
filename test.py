import numpy as np
# import imutils

import cv2

img_rgb = cv2.imread('1.png')

Conv_hsv_Gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
mask = cv2.inRange(img_rgb,(0,0,0), (204,204,204))

img_rgb[mask != 0] = [204, 204, 204]#[0, 0, 255]

ret = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

cv2.imwrite("./test/output2.png", ret)  # show windows

ret = ret[398:1340, 0:]

cv2.imwrite("./test/imgOriginal.png", img_rgb)  # show windows

cv2.imwrite("./test/output.png", ret)  # show windows

cv2.imwrite("./test/mask.png", mask)  # show windows

template2 = cv2.imread('search.png', 0)
template2 = cv2.resize(template2, (109, 109))

print cv2.minMaxLoc(ret)

w, h = template2.shape[::-1]
print w, h
res = cv2.matchTemplate(ret, template2, cv2.TM_CCOEFF_NORMED)
threshold = 0.8

loc = np.where(res >= threshold)
print len(loc[1])
for pt in zip(*loc[::-1]):
    # print pt
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

cv2.imwrite('./test/res.png', img_rgb)


exit()


