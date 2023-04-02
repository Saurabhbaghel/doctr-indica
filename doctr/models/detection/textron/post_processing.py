from .src.config import *
import cv2
import skimage.io as io
import pandas as pd
import numpy as np
from numpy import invert

### Postprocessing Step
def get_bboxes(file):
    # TODO change this saving of the images and then retreiving, or del them afterwards
    img = cv2.imread(RESULTS_DIR + file)

    ht_img, w_img = img.shape[:2]
    
    img = invert(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to binary
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)

    # To detect object contours, we want a black background and a white foreground, so we invert the image (i.e. 255 - pixel value)
    inverted_binary = ~binary

    # Find the contours on the inverted binary image, and store them in a list
    # Contours are drawn around white blobs. hierarchy variable contains info on the relationship between the contours
    contours, hierarchy = cv2.findContours(inverted_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    bboxes = []
    # Draw a bounding box around all contours
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        w = int(w*(1/WIDTH_THRESHOLD))
        h = int(h*(1/HEIGHT_THRESHOLD))
        # Make sure contour area is large enough
        if cv2.contourArea(c) > 30:
            bboxes.append(['text',1,x, y, w, h])

    final_img = cv2.imread(INPUT_IMG_DIR + file)
    for b in bboxes:
        x = b[2]
        y = b[3]
        w = int(b[4])
        h = int(b[5])
        cv2.rectangle(final_img,(x,y), (x+w,y+h), (0, 255, 0),1)

    # TODO develop a scoring mechanism
    SCORE = 1.0
    
    bboxes_for_doctr = [
        [b[2]/w_img, b[3]/ht_img, (b[2]+b[4])/w_img, (b[3]+b[5])/ht_img, SCORE] 
        for b in bboxes
        ]
    # bboxes_for_doctr = np.array(bboxes_for_doctr)
    # df = pd.DataFrame(bboxes, columns = ['label', 'confidence', 'X', 'Y', 'W', 'H'])
    name = file[:len(file) - 4]
    return (np.clip(np.expand_dims(np.asarray(bboxes_for_doctr), axis=0), 0, 1)) if len(bboxes_for_doctr) > 0 else np.zeros((0, 5)) 
    # io.imsave(PREDICTIONS_DIR + name + '_pred.jpg', final_img)
    # df.to_csv(OUT_TXT_DIR + name + '.txt', sep=' ',index=False, header=False)