import cv2
import numpy as np

img=cv2.imread('demo.png',cv2.IMREAD_GRAYSCALE)

kernel = np.array((
    [0, -2, 0],
    [-2, 10, -2],
    [0, -2, 0]))

def cv(img, kernel):
    img_h = img.shape[0]
    img_w = img.shape[1]

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]
   
    #Bo hang giua, lay gia tri doi xung hai ben
    H = (kernel_h - 1) // 2
    W = (kernel_w - 1) // 2

    #Tao ma tran bang ma tran ban dau
    rs = np.zeros((img_h, img_w))

    #Tinh ket qua
    for i in np.arange(H, img_h - H):
        for j in np.arange(W, img_w - W):
            sum = 0
            for k in np.arange(-H, H + 1):
                for l in np.arange(-W, W + 1):
                    n = img[i + k, j + l]
                    m = kernel[H + k, W + l]
                    sum += (m * n)
            rs[i, j] = sum
    return rs

#img_rs=cv(img, kernel)
img_rs=cv2.filter2D(img, -1, kernel)
cv2.imshow('Demo', img)
cv2.imshow('Result', img_rs)


cv2.waitKey(0)
cv2.destroyAllWindows()
