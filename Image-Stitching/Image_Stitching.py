from PIL.Image import NONE
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

def plot_img(rows, cols, index, img, title):
    ax = plt.subplot(rows,cols,index)
    if(len(img.shape) == 3):
        ax_img = plt.imshow(img[...,::-1]) # same as img[:,:,::-1]), RGB image is displayed without cv.cvtColor
    else:
        ax_img = plt.imshow(img, cmap='gray')
    plt.axis('on')
    if(title != None): plt.title(title) 
    return ax_img, ax

def resizing(img,size1,size2):
    img=cv.resize(img,dsize=(size1,size2),interpolation=cv.INTER_AREA)
    return img

img1 = cv.imread("2.jpg")
img2 = cv.imread("1.jpg")
img3 = cv.imread("3.jpg")


kp=[]
des=[]
matches=[]
good_correspondences = []
a=[]
count=0

img=[img1,img2,img3]

def kp_des (img,num): #kp, des 구하기
    sift = cv.SIFT_create()
    global kp, des
    kp.clear()
    des.clear()

    for i in range(num):
        kp.append(sift.detectAndCompute(img[i], None)[0])
        des.append(sift.detectAndCompute(img[i], None)[1])

    flann = cv.FlannBasedMatcher({"algorithm":1, "trees":5}, {"checks":50})
    global matches
    matches.clear()
    for i in range(1,num):
        matches.append(flann.knnMatch(des[0], des[i], k=2))
    

def update_good_correpondences(ratio_dist,num): #good_correspondences 구하기
    global good_correspondences, a, count
    good_correspondences.clear()
    a.clear()
    for i in range (num-1):
        for m,n in matches[i]:
            if m.distance/n.distance < ratio_dist:
                good_correspondences.append(m)
        a.insert(i,good_correspondences[:])
        good_correspondences.clear()
    max=len(a[0])
    count=0
    if num-1!=0:
        for i in range(1,num-1):
            if len(a[i])>max:
                max=len(a[i])
                count=i

def matchline(): 
    img_matches = cv.drawMatches(img[0],kp[0],img[count+1],kp[count+1],a[count],None,matchColor=(0,255,0),singlePointColor=None,matchesMask=None,flags=2)
    #img_matches 그릴 캔버스 생성 후 그리기
    fig = plt.figure(1)
    ax_img, ax = plot_img(1,1,1,img_matches,"matching result")
    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    tx = ax.text(0.05, 0.95, "# good correspondences: " + str(len(a[count])), transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox={'boxstyle':'round', 'facecolor':'wheat', 'alpha':0.5})
    
    #stitch 버튼 추가
    button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(button_ax, 'Stitch', color='lightgoldenrodyellow', hovercolor='0.975')
    button.on_clicked(stitch)
    plt.show()

def next(event):
        plt.close('all')
    
def stitch(event):
    src_pts = np.float32([ kp[0][m.queryIdx].pt for m in a[count] ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp[count+1][m.trainIdx].pt for m in a[count] ]).reshape(-1,1,2)
    H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)

    img2_h,img2_w = img[count+1].shape[:2]
    img2_pts = np.array([[0,0],[0,img2_h],[img2_w,img2_h],[img2_w,0]]).astype(np.float32).reshape(-1,1,2)
    img2_pts_dst = cv.perspectiveTransform(img2_pts,H) #이미지 변환좌표 계산

    img1_shift_x, img1_shift_y = 0,0
    if(np.min(img2_pts_dst[:,:,0]) < 0):  #화면 밖으로 튀어나가는 음수 좌표값있을시 양수값으로 밀어주기
        img1_shift_x = -int(np.min(img2_pts_dst[:,:,0]))
        img2_pts_dst[:,:,0] -= np.min(img2_pts_dst[:,:,0])
                                    
    if(np.min(img2_pts_dst[:,:,1]) < 0):
        img1_shift_y = -int(np.min(img2_pts_dst[:,:,1]))
        img2_pts_dst[:,:,1] -= np.min(img2_pts_dst[:,:,1])
   
    H_new = cv.getPerspectiveTransform(img2_pts, img2_pts_dst)
    
    stitch_plane_rows = int(np.max(img2_pts_dst[:,:,1]))
    if stitch_plane_rows < img[0].shape[0]+img1_shift_y: #이미지 변형이 너무 적게 일어나면 간혹 원본 이미지보다 완성 이미지 세로길이가 작아지는 문제 처리
        stitch_plane_rows = img[0].shape[0]+img1_shift_y

    stitch_plane_cols = int(np.max(img2_pts_dst[:,:,0]))
    if stitch_plane_cols < img[0].shape[1]+img1_shift_x: #이미지 변형이 너무 적게 일어나면 간혹 원본 이미지보다 완성 이미지 가로길이가 작아지는 문제 처리
        stitch_plane_cols = img[0].shape[1]+img1_shift_x
        
    result1 = cv.warpPerspective(img[count+1],H_new, (stitch_plane_cols, stitch_plane_rows), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_TRANSPARENT)
    result2 = np.zeros((stitch_plane_rows, stitch_plane_cols,3), np.uint8)
    result2[0+img1_shift_y:img[0].shape[0]+img1_shift_y, 0+img1_shift_x:img[0].shape[1]+img1_shift_x] = img[0]

    # plt.figure(2)
    # plot_img(3, 1, 1, result1, None)
    # plot_img(3, 1, 2, result2, None)
    
    and_img = cv.bitwise_and(result1, result2)
    and_img_gray = cv.cvtColor(and_img, cv.COLOR_BGR2GRAY)
    th, mask = cv.threshold(and_img_gray, 1, 255, cv.THRESH_BINARY)
    # plot_img(3, 1, 3, mask, None)
    global result3 
    result3 = np.zeros((stitch_plane_rows, stitch_plane_cols,3), np.uint8)
    
    for y in range(stitch_plane_rows):
        for x in range(stitch_plane_cols):
            mask_v = mask[y, x]
            if(mask_v > 0):
                result3[y, x] = np.uint8(result1[y,x] * 0.5 + result2[y,x] * 0.5)
            else:
                result3[y, x] = result1[y,x]+result2[y,x]

    plt.figure(2)
    plot_img(1, 1, 1, result3, None)
    button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(button_ax, 'next', color='lightgoldenrodyellow', hovercolor='0.975')
    button.on_clicked(next)
    plt.show()
    
def finalstitch(num):
    for i in range (num-1):
        kp_des(img,num-i)
        update_good_correpondences(0.7,num-i)
        matchline()
        if i!=num-2:
            del img[0]
            del img[count]
        img.insert(0,result3)

finalstitch(len(img))
