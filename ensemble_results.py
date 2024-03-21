from PIL import Image
import os
import cv2

input_dir0 = r'results\02_HAT_baseline\test'
input_dir1 = r'results\02_RepRLFN_baseline\test'
save_dir = r'results\02_ensemble\test'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_list = os.listdir(input_dir0)


for im in img_list:
    # print("im: {}".format(im))
    # print("im2: {}".format(im[:6]+'.png'))
    img_path0 = os.path.join(input_dir0, im)
    img_path1 = os.path.join(input_dir1, im)
    save_path = os.path.join(save_dir, im)
    # print(img_path)
    img0 = cv2.imread(img_path0)
    img1 = cv2.imread(img_path1)

    ensemble_img = 0.95*img0+0.05*img1

    cv2.imwrite(save_path, ensemble_img)

    print("img_path: {}".format(img_path0))
    # exit(0)