import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

plot_grid_size = (6, 6)
plt.figure(figsize=(11,11))
def plot_img(index, img, title):
    plt.subplot(plot_grid_size[0],plot_grid_size[1],index)
    plt.imshow(img[...,::-1]) 
    plt.axis('off'), plt.title(title) 
    
def __pyrUp(img, size = None):
    nt = tuple([x*2 for x in img.shape[:2]])
    if size == None:
        size = nt
    if nt != size:
        upscale_img = cv.pyrUp(img, None, size)
    else:
        upscale_img = cv.pyrUp(img)
    return upscale_img

def generate_gaussian_pyramid(img, levels):
    GP = [img]
    for i in range(1, levels): 
        img = cv.pyrDown(img)
        GP.append(img)
    return GP

def generate_laplacian_pyramid(GP):
    levels = len(GP)
    LP = [] #[GP[levels + 1]]
    for i in range(levels - 1, 0, -1):
        upsample_img = __pyrUp(GP[i], GP[i-1].shape[:2])
        
        laplacian_img = cv.subtract(GP[i-1], upsample_img)
        LP.append(laplacian_img)
    LP.reverse()
    return LP

def generate_pyramid_composition_image(Pimgs):
    levels = len(Pimgs)
    rows, cols = Pimgs[0].shape[:2] 
    composite_image = np.zeros((rows, cols + int(cols / 2 + 0.5), 3), dtype=Pimgs[0].dtype)
    composite_image[:rows, :cols, :] = Pimgs[0]
    i_row = 0
    for p in Pimgs[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows
    return composite_image

def stitch_vertical(P_side, P_center):
    P_stitch = []
    for la,lb in zip(P_side, P_center):
        rows = la.shape[0]
        cols = la.shape[1]
        lpimg_stitch = np.vstack((la[0:(rows//16)*11,:],lb[(rows//16)*11:(rows//4)*3,:],la[(rows//4)*3:,:]))
        P_stitch.append(lpimg_stitch)
    return P_stitch

def stitch_horizontal(P_side, P_center):
    P_stitch = []
    for la,lb in zip(P_side, P_center):
        rows = la.shape[0]
        cols = la.shape[1]
        lpimg_stitch = np.hstack((la[:,0:(cols//16)*7], lb[:,(cols//16)*7:(cols//8)*5],la[:,(cols//8)*5:]))
        P_stitch.append(lpimg_stitch)
    return P_stitch

img_hand = cv.imread("hand.jpg")
img_eye = cv.imread("eye3.jpg")

# 두 이미지의 사이즈를 맞춰줌
img_eye = cv.resize(img_eye, img_hand.shape[:2])

plot_img(1, img_hand, "hand")
plot_img(2, img_eye, "eye") 

# 가우시안 피라미드와 라플라시안 피라미드를 구한다. 숫자가 작을수록 원본에 가까워짐
GP_hand = generate_gaussian_pyramid(img_hand, 6)
GP_eye = generate_gaussian_pyramid(img_eye, 6)
LP_hand = generate_laplacian_pyramid(GP_hand)
LP_eye = generate_laplacian_pyramid(GP_eye)

plot_img(3, generate_pyramid_composition_image(GP_hand), "GP_hand")
plot_img(4, generate_pyramid_composition_image(GP_eye) , "GP_eye") 
plot_img(5, generate_pyramid_composition_image(LP_eye) , "LP_hand")
plot_img(6, generate_pyramid_composition_image(LP_eye) , "LP_eye") 

# 먼저 수직방향으로 사용할 범위만큼 LP_hand와 LP_eye를 추가하여 라플라시안 이미지를 만들고
LP_stitch = stitch_vertical(LP_hand, LP_eye)
plot_img(7, generate_pyramid_composition_image(LP_stitch), "Comp LP(only v)")
# 다음으로 수평방향으로 정해놓은 범위만큼 LP_hand와 방금 만들었던 LP_stitch를 이용해 라플라시안 이미지를 만든다.
LP_stitch = stitch_horizontal(LP_hand, LP_stitch)
plot_img(8, generate_pyramid_composition_image(LP_stitch), "Comp LP(v+h)")
# 수직방향으로 사용할 범위만큼 GP_hand와 GP_eye를 추가하여 가우시안 피라미드 이미지를 만들고
GP_stitch = stitch_vertical(GP_hand, GP_eye)
plot_img(9, generate_pyramid_composition_image(GP_stitch), "Comp GP(only v)")
# 다음으로 수평방향으로 정해놓은 범위만큼 GP_hand와 방금 만들었던 GP_stitch를 이용해 가우시안 이미지를 만든다.
GP_stitch = stitch_horizontal(GP_hand, GP_stitch)
plot_img(10, generate_pyramid_composition_image(GP_stitch), "Comp GP(v+h)")

# 점점 자연스럽게
recon_img = GP_stitch[-1] 
lp_maxlev = len(LP_stitch) - 1
plot_img(18, recon_img.copy(), "level: " + str(6))
for i in range(lp_maxlev, -1, -1):
    recon_img = __pyrUp(recon_img, LP_stitch[i].shape[:2])
    plot_img(i + 1 + 24, recon_img.copy(), "level: " + str(i))
    recon_img = cv.add(recon_img, LP_stitch[i])
    plot_img(i + 1+12, recon_img.copy(), "level: " + str(i))
    plot_img(i + 1 + 18, LP_stitch[i].copy(), "level: " + str(i))

# 원본 이미지를 연결 시킨 것
rows = img_hand.shape[0]
cols = img_hand.shape[1]
naive_mix = np.vstack((img_hand[0:(rows//32)*21,:],img_eye[(rows//32)*21:(rows//4)*3,:],img_hand[(rows//4)*3:,:]))
naive_mix = np.hstack((img_hand[:,0:(cols//32)*13], naive_mix[:,(cols//32)*13:(cols//32)*19],img_hand[:,(cols//32)*19:]))

plot_img(31, recon_img, "blending")
plot_img(32, naive_mix, "direct connecting")
plt.show()
