from .src.config import *
import skimage.io as io
from .src.lf_utils import *
from .src.utils import get_pixels
from PIL import Image
from torchvision.transforms import ToPILImage

import enum

from .spear.spear.labeling import labeling_function, ABSTAIN, preprocessor


class pixelLabels(enum.Enum):
    TEXT = 1
    NOT_TEXT = 0


class Labeling:
    def __init__(self,imgfile, model) -> None:
        # TODO change INPUT_IMG_DIR
        self.imgfile = INPUT_IMG_DIR + imgfile
        image = io.imread(self.imgfile) # skimage gives RGB so does torchvision.io imgfile.squeeze().permute(1, 2, 0).cpu().numpy() 
        image2 = Image.open(self.imgfile)  # ToPILImage(imgfile) 
        self.image3 = cv2.imread(self.imgfile) # making the channel last imgfile.squeeze().permute(1, 2, 0).cpu().numpy() # 
        self.CHULL        = get_convex_hull(image)
        self.EDGES        = get_image_edges(image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD, THICKNESS)
        # self.PILLOW_EDGES = get_pillow_image_edges(image2, WIDTH_THRESHOLD, HEIGHT_THRESHOLD)
        self.CONTOUR      = get_contour_labels(self.image3, WIDTH_THRESHOLD, HEIGHT_THRESHOLD, THICKNESS)
        self.TITLE_CONTOUR = get_title_contour_labels(self.image3, WIDTH_THRESHOLD, HEIGHT_THRESHOLD, 7)
        self.DOCTR        = get_doctr_labels(model, self.imgfile, image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD)
        # self.DOCTR        = get_existing_doctr_labels(ANN_DOCTR_DIR, imgfile, image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD)
        self.TESSERACT    = get_tesseract_labels(image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD)
        self.MASK_HOLES   = get_mask_holes_labels(image)
        self.MASK_OBJECTS = get_mask_objects_labels(image, LUMINOSITY)
        self.SEGMENTATION = get_segmentation_labels(image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD, THICKNESS)
        self.pixels = get_pixels(image)
        self.image = image
        
    @preprocessor()
    def get_chull_info(self, x):
        return self.CHULL[x[0]][x[1]]


    @preprocessor()
    def get_edges_info(self, x):
        return self.EDGES[x[0]][x[1]]


    # @preprocessor()
    # def get_pillow_edges_info(self, x):
    #     return self.PILLOW_EDGES[x[0]][x[1]]


    @preprocessor()
    def get_doctr_info(self, x):
        return self.DOCTR[x[0]][x[1]]


    @preprocessor()
    def get_tesseract_info(self, x):
        return self.TESSERACT[x[0]][x[1]]


    @preprocessor()
    def get_contour_info(self, x):
        return self.CONTOUR[x[0]][x[1]]


    @preprocessor()
    def get_title_contour_info(self, x):
        return self.TITLE_CONTOUR[x[0]][x[1]]


    @preprocessor()
    def get_mask_holes_info(self, x):
        return self.MASK_HOLES[x[0]][x[1]]


    @preprocessor()
    def get_mask_objects_info(self, x):
        return self.MASK_OBJECTS[x[0]][x[1]]


    @preprocessor()
    def get_segmentation_info(self, x):
        return self.SEGMENTATION[x[0]][x[1]]
    
        
    def __repr__(self) -> str:
        return f"shape of the image : {self.image3.shape}"