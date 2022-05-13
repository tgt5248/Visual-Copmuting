import cv2 as cv



for i in range(1):
    a=cv.imread("m1.jpg")
    print(a.shape)
    b=cv.resize(a,(700,700))
    cv.imwrite("m111.jpg",b)