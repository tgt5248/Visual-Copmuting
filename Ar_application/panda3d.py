from direct.showbase.ShowBase import ShowBase
import panda3d.core as p3c
from direct.actor.Actor import Actor
from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.OnscreenText import OnscreenText
import direct.gui.DirectGui as dui

import cv2 as cv
import numpy as np
import cv2.aruco as aruco
import math
import copy
fr = cv.FileStorage("./camera_parameters.txt", cv.FileStorage_READ)
if not fr.isOpened():
    raise IOError("Cannot open cam parameters")
intrisic_mtx = fr.getNode('camera intrinsic matrix').mat()
dist_coefs = fr.getNode('distortion coefficients').mat()
newcameramtx = fr.getNode('camera new intrinsic matrix').mat()
fr.release()

vid_cap = cv.VideoCapture(0, cv.CAP_DSHOW) 
# Check if the webcam is opened correctly
if not vid_cap.isOpened():
    raise IOError("Cannot open webcam")

vid_cap.set(cv.CAP_PROP_FRAME_WIDTH,1024)
vid_cap.set(cv.CAP_PROP_FRAME_HEIGHT,768)
frame_w = int(vid_cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_h = int(vid_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print(frame_w, frame_h)

base = ShowBase()
base.disableMouse()
winProp = p3c.WindowProperties()
winProp.setSize(frame_w, frame_h)
base.win.requestProperties(winProp)


# Load and transform the panda actor.
pandaActor = Actor("models/panda-model",{"walk": "models/panda-walk4"})
# Loop its animation.
pandaActor.loop("walk")
lMinPt, lMaxPt = p3c.Point3(), p3c.Point3()
 # in the object space

pandaActor.setMat(p3c.LMatrix4(0.05,0,0,0,
                               0,0.05,0,0,
                               0,0,0.05,0,
                               20,20,0,1)) #scale + 이동 
pandaActor.calcTightBounds(lMinPt, lMaxPt)
print(lMinPt, lMaxPt)

pandaActor2 = Actor("models/panda",{"walk": "models/panda-walk"})
pandaActor2.loop("walk")
#lMinPt, lMaxPt = p3c.Point3(), p3c.Point3()
#pandaActor2.calcTightBounds(lMinPt, lMaxPt) # in the object space
m_os2ls = p3c.LMatrix4.scaleMat(1/10.0, 1/10.0, 1/10.0)
pandaActor2.setMat(m_os2ls)

pandaActor3 = base.loader.loadModel('bunny.egg')
#lMinPt, lMaxPt = p3c.Point3(), p3c.Point3()
#pandaActor2.calcTightBounds(lMinPt, lMaxPt) # in the object space
m_os2ls = p3c.LMatrix4.scaleMat(1/10.0, 1/10.0, 1/10.0)
pandaActor3.setMat(m_os2ls)

plight= p3c.PointLight('plight') #광원 추가
plnp = base.render.attachNewNode(plight)
plnp.setPos(100,20,200)
base.render.setLight(plnp)
alight = p3c.AmbientLight('alight') #Ambient light : 기본으로 깔려있는 빛
alight.setColor((0.2,0.2,0.2,1))
alnp = base.render.attachNewNode(alight)
base.render.setLight(alnp)


# set up a texture for (h by w) rgb image
tex = p3c.Texture()
tex.setup2dTexture(frame_w, frame_h, p3c.Texture.T_unsigned_byte, p3c.Texture.F_rgb)

background = OnscreenImage(image=tex) # Load an image object
background.reparentTo(base.render2dp)
base.cam2dp.node().getDisplayRegion(0).setSort(-20)


# tracking the aruco marker
pattern_size = (2, 2)
pattern_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
square_size = 40 # mm unit
pattern_points *= square_size

arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_50)
arucoParams = cv.aruco.DetectorParameters_create()

fov_x = 2 * math.atan(frame_w/(2 * intrisic_mtx[0][0])) * 180 / math.pi
fov_y = 2 * math.atan(frame_h/(2 * intrisic_mtx[1][1])) * 180 / math.pi
base.camLens.setNearFar(10, 10000)
base.camLens.setFov(fov_x, fov_y)

# Panda3D local line
linesx = p3c.LineSegs()
linesy = p3c.LineSegs()
linesz = p3c.LineSegs()

linesx.moveTo(0,0,0)
linesy.moveTo(0,0,0)
linesz.moveTo(0,0,0)

linesx.drawTo(50,0,0)
linesy.drawTo(0,50,0)
linesz.drawTo(0,0,50)

linesx.setThickness(5)
linesy.setThickness(5)
linesz.setThickness(5)

nodex = linesx.create()
nodey = linesy.create()
nodez = linesz.create()

npx = p3c.NodePath(nodex)
npy = p3c.NodePath(nodey)
npz = p3c.NodePath(nodez)

npx.setColorScale(1, 0, 0, 1.0) 
npy.setColorScale(0, 1, 0, 1.0) 
npz.setColorScale(0, 0, 1, 1.0)

panda3d_axis_main = p3c.NodePath('panda3d-axis1')
panda3d_axis_marker2 = p3c.NodePath('panda3d-axis2')
panda3d_axis_marker3 = p3c.NodePath('panda3d-axis3')

npx.reparentTo(panda3d_axis_main)
npy.reparentTo(panda3d_axis_main)
npz.reparentTo(panda3d_axis_main)

panda3d_axis_main.copyTo(panda3d_axis_marker2)
panda3d_axis_main.copyTo(panda3d_axis_marker3)

panda3d_axis_main.reparentTo(base.render)
panda3d_axis_marker2.reparentTo(base.render)
panda3d_axis_marker3.reparentTo(base.render)


def updateBg(task):
    success, frame = vid_cap.read()
    frame = cv.undistort(frame,intrisic_mtx,dist_coefs,None,newcameramtx)
    if success:
        textObject.setText("No AR Marker Detected")
        
        (corners, ids, rejected) = cv.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            matViewMain = None
            matViewInvMain = None
            for (markerCorner, markerID) in zip(corners, ids):
                if markerID == 1:
                    textObject.setText("[INFO] ArUco marker ID: {}".format(markerID))
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    # convert each of the (x, y)-coordinate pairs to integers
                    axis = np.float32([[square_size,0,0], [0,square_size,0], [0,0,square_size]]).reshape(-1,3)

                    ret, rvecs, tvecs = cv.solvePnP(pattern_points, 
                                                    np.asarray([topLeft, topRight, bottomLeft, bottomRight]).reshape(-1, 2), 
                                                    intrisic_mtx, None)

                    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, intrisic_mtx, None)
                    def draw(img, axis_center, imgpts):
                        cv.line(img, axis_center, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 4) # X
                        cv.line(img, axis_center, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 4) # Y
                        cv.line(img, axis_center, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 4) # Z

                    draw(frame, tuple(topLeft.ravel().astype(int)), imgpts)
                    
                    # from rvecs and tvecs
                    def getViewMatrix(rvecs, tvecs):
                        # build view matrix
                        rmtx = cv.Rodrigues(rvecs)[0]
                        
                        view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],tvecs[0]],
                                                [rmtx[1][0],rmtx[1][1],rmtx[1][2],tvecs[1]],
                                                [rmtx[2][0],rmtx[2][1],rmtx[2][2],tvecs[2]],
                                                [0.0       ,0.0       ,0.0       ,1.0    ]])
                        
                        inverse_matrix = np.array([[ -1.0, 1.0, -1.0, 1.0],
                                                [1.0,-1.0,1.0,-1.0],
                                                [1.0,-1.0,1.0,-1.0],
                                                [ 1.0, 1.0, 1.0, 1.0]])

                        view_matrix = view_matrix * inverse_matrix
                        view_matrix = np.transpose(view_matrix)
                        
                        return p3c.LMatrix4(view_matrix[0][0],view_matrix[0][1],view_matrix[0][2],view_matrix[0][3],
                                view_matrix[1][0],view_matrix[1][1],view_matrix[1][2],view_matrix[1][3],
                                view_matrix[2][0],view_matrix[2][1],view_matrix[2][2],view_matrix[2][3],
                                view_matrix[3][0],view_matrix[3][1],view_matrix[3][2],view_matrix[3][3])

                    matView = getViewMatrix(rvecs, tvecs)#world space -> camera space 변환matrix임.
                    matViewMain = getViewMatrix(rvecs, tvecs)
                    matViewInvMain = getViewMatrix(rvecs, tvecs)
                    matViewInvMain.invertInPlace()
                    matViewInv = matView
                    matViewInv.invertInPlace() # 역변환(camera space -> world space)해주는 matrix
                    cam_pos = matViewInv.xformPoint(p3c.LPoint3(0, 0, 0))                    
                    cam_view = matViewInv.xformVec(p3c.LVector3(0, 0, -1))
                    cam_up = matViewInv.xformVec(p3c.LVector3(0, 1, 0))
                    

                    marker_x_end = np.dot(np.dot(np.array([-square_size,0,0,1]),np.array(matViewMain)),matViewInvMain)
                    marker_y_end = np.dot(np.dot(np.array([0,square_size,0,1]),np.array(matViewMain)),matViewInvMain)
                    marker_center_pos = (marker_x_end + marker_y_end)/2
                    pandaActor.setPos((marker_center_pos[0],marker_center_pos[1],marker_center_pos[2]))
                    pandaActor.reparentTo(base.render)


                    # camera
                    base.camera.setPos(cam_pos)
                    base.camera.lookAt(cam_pos + cam_view, cam_up)

                    fov_x = 2 * math.atan(frame_w/(2 * intrisic_mtx[0][0])) * 180 / math.pi
                    fov_y = 2 * math.atan(frame_h/(2 * intrisic_mtx[1][1])) * 180 / math.pi
                    base.camLens.setNearFar(10, 10000)
                    base.camLens.setFov(fov_x, fov_y)
            

                if markerID == 2:
                    textObject.setText("[INFO] ArUco marker ID: {}".format(markerID))
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    # convert each of the (x, y)-coordinate pairs to integers
                    axis = np.float32([[square_size,0,0], [0,square_size,0], [0,0,square_size]]).reshape(-1,3)

                    ret, rvecs, tvecs = cv.solvePnP(pattern_points, 
                                                    np.asarray([topLeft, topRight, bottomLeft, bottomRight]).reshape(-1, 2), 
                                                    intrisic_mtx, None)
                    
                    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, intrisic_mtx, None)
                    def draw(img, axis_center, imgpts):
                        cv.line(img, axis_center, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 4) # X
                        cv.line(img, axis_center, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 4) # Y
                        cv.line(img, axis_center, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 4) # Z

                    draw(frame, tuple(topLeft.ravel().astype(int)), imgpts)
                    

                    # from rvecs and tvecs
                    def getViewMatrix(rvecs, tvecs):
                        # build view matrix
                        rmtx = cv.Rodrigues(rvecs)[0]
                        
                        view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],tvecs[0]],
                                                [rmtx[1][0],rmtx[1][1],rmtx[1][2],tvecs[1]],
                                                [rmtx[2][0],rmtx[2][1],rmtx[2][2],tvecs[2]],
                                                [0.0       ,0.0       ,0.0       ,1.0    ]])

                        inverse_matrix = np.array([[ -1.0, 1.0, -1.0, 1.0],
                                                [1.0,-1.0,1.0,-1.0],
                                                [1.0,-1.0,1.0,-1.0],
                                                [ 1.0, 1.0, 1.0, 1.0]])


                        view_matrix = view_matrix * inverse_matrix
                        view_matrix = np.transpose(view_matrix)
                        
                        return p3c.LMatrix4(view_matrix[0][0],view_matrix[0][1],view_matrix[0][2],view_matrix[0][3],
                                view_matrix[1][0],view_matrix[1][1],view_matrix[1][2],view_matrix[1][3],
                                view_matrix[2][0],view_matrix[2][1],view_matrix[2][2],view_matrix[2][3],
                                view_matrix[3][0],view_matrix[3][1],view_matrix[3][2],view_matrix[3][3])

                    matView = getViewMatrix(rvecs, tvecs)
                    mv2=getViewMatrix(rvecs, tvecs)
                    matViewInv = matView
                    matViewInv.invertInPlace()

                    if(matViewMain != None):
                        marker_start = np.dot(np.dot(np.array([0,0,0,1]),np.array(mv2)),matViewInvMain)
                        marker_x_end = np.dot(np.dot(np.array([-square_size,0,0,1]),np.array(mv2)),matViewInvMain)
                        marker_y_end = np.dot(np.dot(np.array([0,square_size,0,1]),np.array(mv2)),matViewInvMain)
                        marker_center_pos = (marker_x_end + marker_y_end)/2
                        
                    
                        #mv3 =np.dot(np.dot(np.array(matViewInv),np.array(matViewMain)),np.array(pandaActor.getMat()))
                        mv3 =np.dot(np.dot(np.array(pandaActor.getMat()),np.array(matViewMain)),np.array(matViewInv))
                        mv3 = p3c.LMatrix4(mv3[0][0],-mv3[0][1],-mv3[0][2],mv3[0][3],
                                           -mv3[1][0],mv3[1][1],mv3[1][2],-mv3[1][3],
                                           -mv3[2][0],mv3[2][1],mv3[2][2],-mv3[2][3],
                                           marker_center_pos[0],marker_center_pos[1],marker_center_pos[2],1)
                        pandaActor2.setMat(mv3)
                        
                        mv3[3][0], mv3[3][1], mv3[3][2] = marker_start[0], marker_start[1], marker_start[2]
                        panda3d_axis_marker2.setMat(mv3)
                        panda3d_axis_marker2.setScale(1,1,1)
                        pandaActor2.setScale(2)
                        pandaActor2.reparentTo(base.render)


                        #pandaActor.hide()
                
                if markerID == 3:
                    textObject.setText("[INFO] ArUco marker ID: {}".format(markerID))
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    # convert each of the (x, y)-coordinate pairs to integers
                    axis = np.float32([[square_size,0,0], [0,square_size,0], [0,0,square_size]]).reshape(-1,3)

                    ret, rvecs, tvecs = cv.solvePnP(pattern_points, 
                                                    np.asarray([topLeft, topRight, bottomLeft, bottomRight]).reshape(-1, 2), 
                                                    intrisic_mtx, None)
                    
                    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, intrisic_mtx, None)
                    def draw(img, axis_center, imgpts):
                        cv.line(img, axis_center, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 4) # X
                        cv.line(img, axis_center, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 4) # Y
                        cv.line(img, axis_center, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 4) # Z

                    draw(frame, tuple(topLeft.ravel().astype(int)), imgpts)
                    

                    # from rvecs and tvecs
                    def getViewMatrix(rvecs, tvecs):
                        # build view matrix
                        rmtx = cv.Rodrigues(rvecs)[0]
                        
                        view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],tvecs[0]],
                                                [rmtx[1][0],rmtx[1][1],rmtx[1][2],tvecs[1]],
                                                [rmtx[2][0],rmtx[2][1],rmtx[2][2],tvecs[2]],
                                                [0.0       ,0.0       ,0.0       ,1.0    ]])

                        inverse_matrix = np.array([[ -1.0, 1.0, -1.0, 1.0],
                                                [1.0,-1.0,1.0,-1.0],
                                                [1.0,-1.0,1.0,-1.0],
                                                [ 1.0, 1.0, 1.0, 1.0]])


                        view_matrix = view_matrix * inverse_matrix
                        view_matrix = np.transpose(view_matrix)
                        
                        return p3c.LMatrix4(view_matrix[0][0],view_matrix[0][1],view_matrix[0][2],view_matrix[0][3],
                                view_matrix[1][0],view_matrix[1][1],view_matrix[1][2],view_matrix[1][3],
                                view_matrix[2][0],view_matrix[2][1],view_matrix[2][2],view_matrix[2][3],
                                view_matrix[3][0],view_matrix[3][1],view_matrix[3][2],view_matrix[3][3])

                    matView = getViewMatrix(rvecs, tvecs)
                    mv4=getViewMatrix(rvecs, tvecs)
                    matViewInv = matView
                    matViewInv.invertInPlace()

                    if(matViewMain != None):
                        marker_start = np.dot(np.dot(np.array([0,0,0,1]),np.array(mv4)),matViewInvMain)
                        marker_x_end = np.dot(np.dot(np.array([-square_size,0,0,1]),np.array(mv4)),matViewInvMain)
                        marker_y_end = np.dot(np.dot(np.array([0,square_size,0,1]),np.array(mv4)),matViewInvMain)
                        marker_center_pos = (marker_x_end + marker_y_end)/2
                        
                        
                        #mv5 =np.dot(np.dot(np.array(matViewInv),np.array(matViewMain)),np.array(pandaActor.getMat()))
                        mv5 =np.dot(np.dot(np.array(pandaActor.getMat()),np.array(matViewMain)),np.array(matViewInv))
                        mv5 = p3c.LMatrix4(mv5[0][0],-mv5[0][1],-mv5[0][2],mv5[0][3],
                                           -mv5[1][0],mv5[1][1],mv5[1][2],-mv5[1][3],
                                           -mv5[2][0],mv5[2][1],mv5[2][2],-mv5[2][3],
                                           marker_center_pos[0],marker_center_pos[1],marker_center_pos[2],1)
                        
                        
                        pandaActor3.setMat(mv5)
                        
                        mv5[3][0], mv5[3][1], mv5[3][2] = marker_start[0], marker_start[1], marker_start[2]
                        panda3d_axis_marker3.setMat(mv5)
                        panda3d_axis_marker3.setScale(1,1,1)
                        pandaActor3.setScale(3)
                        pandaActor3.reparentTo(base.render)
                        #pandaActor.hide()

        
        # positive y goes down in openCV, so we must flip the y coordinates
        frame = cv.flip(frame, 0)
        # overwriting the memory with new frame
        tex.setRamImage(frame)
        return task.cont

base.taskMgr.add(updateBg, 'video frame update')

aspect = frame_w / frame_h
textObject = OnscreenText(text="No AR Marker Detected", pos=(aspect - 0.05, -0.95), 
                        scale=(0.07, 0.07),
                        fg=(1, 0.5, 0.5, 1), 
                        align=p3c.TextNode.A_right,
                        mayChange=1)
textObject.reparentTo(base.aspect2d)

base.run()
