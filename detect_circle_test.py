
import cv2
import numpy as np
  
# Read image.
from os import walk

filenames = next(walk("test_images"), (None, None, []))[2] 
# print(filenames)



for name in filenames:
    img = cv2.imread('test_images/' + name, cv2.IMREAD_COLOR)
    # print(img.shape[1])
    # img = cv2.resize(img, (500,150), interpolation = cv2.INTER_AREA)
    img = img[:,:img.shape[1]//2,:]
    # break
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))
      
    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred, 
                       cv2.HOUGH_GRADIENT, 1.2, 15, param1 = 50,
                   param2 = 35, minRadius = 1, maxRadius = 90)
      
    # Draw circles that are detected.
    if detected_circles is not None:
      
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
      
            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)
      
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
            cv2.imshow("Detected Circle", img)
            cv2.waitKey(0)
            break
    else:
        print(name)


# import numpy as np
# from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt

# for name in filenames:

#     # Open the input image as numpy array, convert to RGB
#     img=Image.open("data_train_croped_circle/" + name).convert("RGB")
# qq
#     npImage=np.array(img)
#     h,w=img.size

#     # Create same size alpha layer with circle
#     alpha = Image.new('L', img.size,0)
#     draw = ImageDraw.Draw(alpha)
#     draw.pieslice([0,0,h,w],0,360,fill=255)

#     # Convert alpha Image to numpy array
#     npAlpha=np.array(alpha)

#     # Add alpha layer to RGB
#     npImage=np.dstack((npImage,npAlpha))
#     # Save with alpha
#     Image.fromarray(npImage).save('data/' + name.split(".")[0].split("_")[0] + '.png')
#     cv2.imshow("Detected Circle", npImage)
#     cv2.waitKey(0)
#     # img = cv2.imread("data/" + name.split(".")[0] + '.png')
#     # # img = cv2.resize(img, (100,100))
#     # cv2.imwrite('data/' + name.split(".")[0] + '.png', img)


# import cv2
# import numpy as np

# # Load image, grayscale, median blur, Otsus threshold
# image = cv2.imread('test_images/1.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.medianBlur(gray, 77)
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# # Morph open 
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

# # Find contours and filter using contour area and aspect ratio
# cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.04 * peri, True)
#     area = cv2.contourArea(c)
#     if len(approx) > 5 and area > 1000 and area < 500000:
#         ((x, y), r) = cv2.minEnclosingCircle(c)
#         cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)

# cv2.imshow('thresh', thresh)
# cv2.imshow('opening', opening)
# cv2.imshow('image', image)
# cv2.waitKey()   