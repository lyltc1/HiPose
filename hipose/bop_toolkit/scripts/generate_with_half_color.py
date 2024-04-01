""" for set of images, output the half color and save as jpg format """

import cv2

images = ['/home/z3d/data/lmo/test/000003/rgb/000598.png', 
          '/home/z3d/data/lmo/test/000003/rgb/000603.png']
for image in images:
    rgb = cv2.imread(image)
    rgb = rgb // 2
    cv2.imwrite(image.replace('.png', '.jpg'), rgb)