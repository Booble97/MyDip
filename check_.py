import cv2, numpy as np
image = cv2.imread('Pek.jpg')[100:1300, 100:1100]

#градации серого
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
cv2.waitKey(0)

#строго черно-белое инвертированное
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('binary',thresh)
cv2.waitKey(0)

#раздутое
kernel = np.ones((5,15), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow('dilated',img_dilation)
cv2.waitKey(0)

#ищем и сортируем контуры
im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_ctrs = sorted(ctrs, key=lambda ctr: (cv2.boundingRect(ctr)[1]//10, cv2.boundingRect(ctr)[0]))

for i, ctr in enumerate(sorted_ctrs):
    # Рисуем прямоугольник вокруг каждого контура
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = image[y:y+h, x:x+w]

    # show ROI
    # cv2.imshow('segment no:'+str(i),roi)
    cv2.rectangle(image,(x,y),( x + w, y + h ),(90, 0, 255), 2)
    cv2.putText(image, str(i), (x, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 0, 255), 2)

cv2.imshow('marked areas',image)
cv2.waitKey(0)
cv2.destroyAllWindows()