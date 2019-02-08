import cv2, numpy as np

def main():

    image = cv2.imread('Pek.jpg')#[:1000, :1000]
    image2 = image.copy()

    #градации серого
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #

    #строго черно-белое инвертированное
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #

    #раздутое
    kernel = np.ones((5,2), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    #

    #ищем и сортируем контуры
    im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: (cv2.boundingRect(ctr)[1]//10, cv2.boundingRect(ctr)[0]))

    for i, ctr in enumerate(sorted_ctrs):
        # Рисуем прямоугольник вокруг каждого контура
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = image[y:y+h, x:x+w]

        # show ROI
        # cv2.imshow('segment no:'+str(i),roi)
        cv2.imwrite("file_"+str(i)+".png", roi)
    #