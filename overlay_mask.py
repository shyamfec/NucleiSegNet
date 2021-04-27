# from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
# path to the test images
Test_image_path = '/content/drive/My Drive/intern_pyth/monuseg/TestData/tis/'
# path to the segmented mask
mask_path = '/content/results/'
# path to the overlay output
if not os.path.exists("/content/overlay"):
    os.mkdir("/content/overlay")
overlay = '/content/overlay/'

test_path_ids1 = next(os.walk(Test_image_path))[2]
mask_path_ids1 = next(os.walk(mask_path))[2]

test_path_ids = sorted(test_path_ids1,key=lambda x: (os.path.splitext(x)[0]))
mask_path_ids = sorted(mask_path_ids1,key=lambda x: (os.path.splitext(x)[0]))

print('Creating overlay images ===========>')
for _,i in tqdm(enumerate(test_path_ids),total=len(test_path_ids)):

    test = Test_image_path+i

    mask = mask_path+os.path.splitext(i)[0]+'.png'

    seg   = cv2.imread(mask,cv2.IMREAD_GRAYSCALE)
    main = cv2.imread(test,cv2.COLOR_BGR2RGB)

    contours, hierarchy = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image = cv2.drawContours(main, contours, -1, (0, 255, 0), 3)
    loc = overlay+os.path.splitext(i)[0]+'.png'
    # Save result
    cv2.imwrite(loc,image)
print('Created Overlay images ===========>')
