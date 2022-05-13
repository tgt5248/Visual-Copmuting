import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def plot_img(rows, cols, index, img, title, axis='on'):
    ax=plt.subplot(rows,cols,index)
    if(len(img.shape)==3):
        ax_img=plt.imshow(img[...,::-1])
    else:
        ax_img=plt.imshow(img,cmap='gray')
    plt.axis(axis)
    if(title !=None): plt.title(title)
    return ax_img,ax

def display_untilKey(Pimg,titles,file_out=False):
    for img,title in zip(Pimg,titles):
        cv.imshow(title,img)
        if file_out==True:
            cv.imwrite(title+".jpg",img)
    cv.waitKey(0)
    
def detect_2d_points_from_cbimg(file_name,pattern_size): #calibration image로부터 2d 포인트를 detect 하는 function
    img=cv.imread(file_name) #해당파일을 인풋으로 받아서 파일을 불러온다.
    img_gray=cv.cvtColor(img,cv.COLOR_RGB2GRAY) #그레이스케일로 변환, 많은 오픈cv가 그레이스케일에서 되는 경우가 많음
    
    # found = point를 찾으면 true, corner = corner point
    # Find the chess board corners
    found,corner=cv.findChessboardCorners(img_gray,pattern_size,None) #이미지그레이에서 패턴사이즈의 체스보드를 찾아줌, found에는 해당 체스보드의 포인트를 잘 찾았는가하는 데이터, 우리가 찾는 포인터는 corner에
    if found: # 정밀한 코너를 찾기 위한 과정
        criteria=(cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_COUNT,30,0.1) # 0.001
        corner=cv.cornerSubPix(img_gray,corner,(5,5),(-1,-1),criteria)  #서브픽셀단위로 더 정밀하게 코너를 찾아줌, 11x11에 해당하는 마스크 내에서 interactive하게 어떤 feature point의 intensity가 가장 커지는 포인트를 찾아줌
        # 읽혀진 영상의 패턴을 그림
        cv.drawChessboardCorners(img,pattern_size,corner,found) 
    
    if not found:
        print('chessboard not found')
        return None

    return (corner,img)    

if __name__=="__main__":
     # 본인의 사진에 맞게 설정해줘야해요!
    pattern_size=(7,6)
    square_size=24
    
    #pattern_point=[]
    #idx=0
    #for y in range(pattern_size[1]):
    #    for x in range(pattern_size[0]):
    #        pattern_point.append[(x,y,0)]
    #        idx+=1
    #pattern_point = np.array(pattern_point).reshape(-1,3).astype(np.float32)
    
    #두 줄의 np 코드로 위 주석문 처리 가능
    pattern_point=np.zeros((pattern_size[0]*pattern_size[1],3),np.float32)
    pattern_point[:,:2]=np.indices(pattern_size).T.reshape(-1,2)
    
    # corner 3D point
    pattern_point*=square_size
    
    import glob
    image_names=glob.glob('./cal/*.JPG')
    print(image_names)
    
    # corner 2D point
    chessboard_imgs=[] # 체스보드 이미지들을  2d position들을 넣을 곳
    chessboard_imgs=[detect_2d_points_from_cbimg(file_name,pattern_size) for file_name in image_names] #이미지와 모든 이미지들로부터 찾은 chessboard 포인트를 리스트에 저장
    
    # Arryas to store object points and image points from all the images
    obj_point=[] # 3d point in real world space
    img_point=[] # 2d points in image plane
    #idx=1
    #plt.figure(1)
    for x in chessboard_imgs:
        if x is not None: #체스보드
            img_point.append(x[0]) #2d포인트를 저장
            obj_point.append(pattern_point) #3d포인트를 저장
            #plot_img(4,4,idx,x[1],None,'off')
            #idx+=1
    #plt.show()
    h,w=cv.imread(image_names[0]).shape[:2] #같은 해상도의 이미지기 떄문에 이미지의 width, height를 구함
    print('image size : %d' %w + ', %d' %h + ' x%d images' %len(image_names))
    
    # rms_err = reprojetion error
    # intrinsic_mtx = K matrix
    # dist_coefs = distortion coefficients
    # rvecs = R vector
    # tvecs = translation vector
    # 핀홀 카메라의 특성상 왜곡이 일어나 이를 펴주기 위한 함수
    rms_err,intrinsic_mtx,dist_coefs,rvecs,tvecs=cv.calibrateCamera(obj_point, img_point,(w,h),None, None) #re-projection error, K matrix, lens distortion을 correction 하기 위한 coefficients값, rotation이 vector로 표현,traslation vector 
    newcameramtx,roi=cv.getOptimalNewCameraMatrix(intrinsic_mtx,dist_coefs,(w,h),0,(w,h)) #계산된 coefficients값을 이용해 왜곡을 펴서 새로운 이미지를 저장 , 0은 주변부 자르는것, 원본과 같은 크기로해서 해상도 손실 줄임 
    
     # 본인의 사진에 맞게 설정해줘야해요!
    pattern_size=(2,2)
    square_size=62.5
    pattern_point=np.zeros((pattern_size[0]*pattern_size[1],3),np.float32)
    pattern_point[:,:2]=np.indices(pattern_size).T.reshape(-1,2)
    pattern_point *= square_size
    
    import cv2.aruco as aruco
    # aruco마커를 생성하는 딕셔너리
    arucoDict=cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_50)
    marker_img=np.zeros((300,300,1),dtype="uint8")
    # 두 번째 파라미터의 번호 변경시 마커 모양 바뀜
    cv.aruco.drawMarker(arucoDict,1,300,marker_img,1) 
    cv.imwrite('./m1.bmp',marker_img) # 마커 생성
    arucoParams=cv.aruco.DetectorParameters_create()
    
    img=cv.imread('m1.jpg')
    undist_img=cv.undistort(img,intrinsic_mtx,dist_coefs,None,newcameramtx)
    #plt.figure(2)
    #plot_img(1,2,1,img,'distorted','off')
    #plot_img(1,2,2,undist_img,'undistorted','off')
    #plt.show()
    
    (corners,ids,regected)=cv.aruco.detectMarkers(undist_img,arucoDict,parameters=arucoParams)
    
    # verify "at least" one Aruco marker was detected
    if len(corners)>0:
        # flatten the aruco IDs lsit
        ids=ids.flatten()
        # loop over the detected Aruco corners
        for(markerCorner,markerID) in zip(corners, ids):
            print("[INFO] ArUco marker ID: {}".format(markerID))
            if markerID==1:
                #extract the marker corners (which are always returned in 
                # top-left, top-right, bottom-right, and bottom-left order)
                corners=markerCorner.reshape((4,2))
                (topLeft, topRight, bottomRight, bottomLeft)=corners
                
                ret,rvecs,tvecs=cv.solvePnP(pattern_point,np.asarray([topLeft,topRight,bottomLeft,bottomRight]).reshape(-1,2),newcameramtx,None)
                
                 # None : undistort 되었기 때문에 None으로 표기
                axis=np.float32([[square_size,0,0],[0,square_size,0],[0,0,square_size]]).reshape(-1,3)
                # project 3D points to image plane
                imgpts, jac=cv.projectPoints(axis,rvecs,tvecs,newcameramtx,None)
                
                axix_center=tuple(topLeft.ravel().astype(int))
                cv.line(undist_img,axix_center,tuple(imgpts[0].ravel().astype(int)),(0,0,255),5)
                cv.line(undist_img,axix_center,tuple(imgpts[1].ravel().astype(int)),(0,255,0),5)
                cv.line(undist_img,axix_center,tuple(imgpts[2].ravel().astype(int)),(255,0,0),5)
                display_untilKey([undist_img],["pose"])
