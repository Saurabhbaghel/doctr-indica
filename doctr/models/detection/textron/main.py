from .src.lf_utils import *
from .src.config import *
from .src.utils import get_label

from PIL import Image
from tqdm import tqdm

import subprocess
import pandas as pd

import torch
import torch.nn as nn
# from doctr.models import ocr_predictor
# from ...zoo import ocr_predictor
from doctr.file_utils import is_tf_available, is_torch_available
from ... import detection
from ...preprocessor import PreProcessor
from ..predictor import DetectionPredictor
from ...predictor import OCRPredictor
from ...recognition.zoo import recognition_predictor

from .spear.spear.labeling import labeling_function, ABSTAIN, preprocessor
from .spear.spear.labeling import LFSet, PreLabels
from .spear.spear.cage import Cage

from .data_processing import Labeling, pixelLabels
from .post_processing import get_bboxes

import warnings

warnings.filterwarnings("ignore")

__all__ = ["textron"]

imgfile = None
Y = None
# global lf
# MODEL = ocr_predictor(pretrained=True)
key_params = {}
det_arch = "db_resnet50"
reco_arch = "crnn_vgg16_bn"
det_model = detection.__dict__[det_arch](
    pretrained=True,
    pretrained_backbone=True,
    assume_straight_pages=True
)

key_params["mean"] = det_model.cfg["mean"]
key_params["std"] = det_model.cfg["std"]
key_params["batch_size"] = 1

det_predictor = DetectionPredictor(
    PreProcessor(det_model.cfg["input_shape"][:-1] if is_tf_available() else det_model.cfg["input_shape"][1:], **key_params),
            det_model,
)

rec_model = recognition_predictor(
    reco_arch, pretrained=True, pretrained_backbone=True, batch_size=1
)


MODEL = OCRPredictor(
    det_predictor,
    rec_model,
    # assume_straight_pages=True,
    # preserve_aspect_ratio=True,
    # symmetric_aspect_ratio=True,
    # symmetric_pad=True,
    # detect_orientation=False,
    # detect_language=True
)



@preprocessor()
def get_chull_info(x):
    return lf.CHULL[x[0]][x[1]]


@preprocessor()
def get_edges_info(x):
    return lf.EDGES[x[0]][x[1]]


@preprocessor()
def get_pillow_edges_info(x):
    return lf.PILLOW_EDGES[x[0]][x[1]]


@preprocessor()
def get_doctr_info(x):
    return lf.DOCTR[x[0]][x[1]]


@preprocessor()
def get_tesseract_info(x):
    return lf.TESSERACT[x[0]][x[1]]


@preprocessor()
def get_contour_info(x):
    return lf.CONTOUR[x[0]][x[1]]


@preprocessor()
def get_title_contour_info(x):
    return lf.TITLE_CONTOUR[x[0]][x[1]]


@preprocessor()
def get_mask_holes_info(x):
    return lf.MASK_HOLES[x[0]][x[1]]


@preprocessor()
def get_mask_objects_info(x):
    return lf.MASK_OBJECTS[x[0]][x[1]]


@preprocessor()
def get_segmentation_info(x):
    return lf.SEGMENTATION[x[0]][x[1]]


@labeling_function(label=pixelLabels.NOT_TEXT, pre=[get_chull_info], name="CHULL_PURE")
def CONVEX_HULL_LABEL_PURE(pixel):
    if pixel:
        return pixelLabels.NOT_TEXT
    else:
        return ABSTAIN


@labeling_function(label=pixelLabels.TEXT, pre=[get_chull_info], name="CHULL_NOISE")
def CONVEX_HULL_LABEL_NOISE(pixel):
    if pixel:
        return pixelLabels.TEXT
    else:
        return ABSTAIN


@labeling_function(label=pixelLabels.TEXT, pre=[get_edges_info], name="SKIMAGE_EDGES")
def EDGES_LABEL(pixel):
    if pixel:
        return pixelLabels.TEXT
    else:
        return ABSTAIN


@labeling_function(
    label=pixelLabels.NOT_TEXT, pre=[get_edges_info], name="SKIMAGE_EDGES_REVERSE"
)
def EDGES_LABEL_REVERSE(pixel):
    if pixel:
        return ABSTAIN
    else:
        return pixelLabels.NOT_TEXT


@labeling_function(
    label=pixelLabels.TEXT, pre=[get_pillow_edges_info], name="PILLOW_EDGES"
)
def PILLOW_EDGES_LABEL(pixel):
    if pixel:
        return pixelLabels.TEXT
    else:
        return ABSTAIN


@labeling_function(
    label=pixelLabels.NOT_TEXT, pre=[get_pillow_edges_info], name="PILLOW_EDGES_REVERSE"
)
def PILLOW_EDGES_LABEL_REVERSE(pixel):
    if pixel:
        return ABSTAIN
    else:
        return pixelLabels.NOT_TEXT


@labeling_function(label=pixelLabels.TEXT, pre=[get_doctr_info], name="DOCTR")
def DOCTR_LABEL(pixel):
    if pixel:
        return pixelLabels.TEXT
    else:
        return ABSTAIN


@labeling_function(label=pixelLabels.TEXT, pre=[get_doctr_info], name="DOCTR2")
def DOCTR_LABEL2(pixel):
    if pixel:
        return pixelLabels.TEXT
    else:
        return ABSTAIN


@labeling_function(label=pixelLabels.TEXT, pre=[get_tesseract_info], name="TESSERACT")
def TESSERACT_LABEL(pixel):
    if pixel:
        return pixelLabels.TEXT
    else:
        return ABSTAIN


@labeling_function(label=pixelLabels.TEXT, pre=[get_contour_info], name="CONTOUR")
def CONTOUR_LABEL(pixel):
    if pixel:
        return pixelLabels.TEXT
    else:
        return ABSTAIN


@labeling_function(
    label=pixelLabels.TEXT, pre=[get_title_contour_info], name="CONTOUR_TITLE"
)
def CONTOUR_TITLE_LABEL(pixel):
    if pixel:
        return pixelLabels.TEXT
    else:
        return ABSTAIN


@labeling_function(
    label=pixelLabels.NOT_TEXT, pre=[get_mask_holes_info], name="MASK_HOLES"
)
def MASK_HOLES_LABEL(pixel):
    if pixel:
        return pixelLabels.NOT_TEXT
    else:
        return ABSTAIN


@labeling_function(
    label=pixelLabels.NOT_TEXT, pre=[get_mask_objects_info], name="MASK_OBJECTS"
)
def MASK_OBJECTS_LABEL(pixel):
    if pixel:
        return pixelLabels.NOT_TEXT
    else:
        return ABSTAIN


@labeling_function(
    label=pixelLabels.TEXT, pre=[get_segmentation_info], name="SEGMENTATION"
)
def SEGMENTATION_LABEL(pixel):
    if pixel:
        return pixelLabels.TEXT
    else:
        return ABSTAIN


### Get LF Analysis of the input images
def analysis(img):

    ### Labeling Functions which should be run
    LFS = [globals()[LF] for LF in lab_funcs]

    rules = LFSet("DETECTION_LF")
    rules.add_lf_list(LFS)

    R = np.zeros((lf.pixels.shape[0],len(rules.get_lfs())))
    Y = io.imread(INPUT_IMG_DIR + img)
    
    name = img[:len(img) - 8]
    
    if GROUND_TRUTH_AVAILABLE:
        df = pd.read_csv(GROUND_TRUTH_DIR+name+'_pro.txt', delimiter=' ',
                     names=["token", "x0", "y0", "x1", "y1", "R", "G", "B", "font name", "label"])

        height, width, _ = Y.shape
        for i in range(df.shape[0]):
            x0, y0, x1, y1  = (df['x0'][i], df['y0'][i], df['x1'][i], df['y1'][i])
            x0, y0, x1, y1 = (int(x0*width/1000), int(y0*height/1000), int(x1*width/1000), int(y1*height/1000))
            w = int((x1-x0)*WIDTH_THRESHOLD)
            h = int((y1-y0)*HEIGHT_THRESHOLD)
            cv2.rectangle(Y, (x0, y0), (x0+w, y0+h), (0, 0, 0), cv2.FILLED)

    gold_label = get_label(Y)

    td_noisy_labels = PreLabels(
        name="TD",
        data=lf.pixels,
        rules=rules,
        gold_labels=gold_label,
        labels_enum=pixelLabels,
        num_classes=2,
    )

    L, S = td_noisy_labels.get_labels()

    analyse = td_noisy_labels.analyse_lfs(plot=True)

    result = analyse.head(16)
    result["image"] = img

    print(result)
    return result


### Get CAGE based output predictions
def cage(file, X):

    ### Labeling Functions which should be run
    LFS = [globals()[LF] for LF in lab_funcs]
    # LFS = lab_funcs

    prob_arr = np.array(QUALITY_GUIDE)

    rules = LFSet("DETECTION_LF")
    rules.add_lf_list(LFS)

    n_lfs = len(rules.get_lfs())

    # file has to be complete name
    Y = io.imread(file) #io.imread(INPUT_IMG_DIR + file)
    height, width, _ = Y.shape
    # TODO do something for the GROUND_TRUTH_AVAILABLE currently it is False
    if GROUND_TRUTH_AVAILABLE:
        if ("docbank" in INPUT_DATA_DIR) or "testing_sample" in INPUT_DATA_DIR:
            name = file[: len(file) - 4]
            df = pd.read_csv(
                GROUND_TRUTH_DIR + name + ".txt",
                delimiter=" ",
                names=[
                    "token",
                    "x0",
                    "y0",
                    "x1",
                    "y1",
                    "R",
                    "G",
                    "B",
                    "font name",
                    "label",
                ],
            )

            for i in range(df.shape[0]):
                x0, y0, x1, y1 = (df["x0"][i], df["y0"][i], df["x1"][i], df["y1"][i])
                x0, y0, x1, y1 = (
                    int(x0 * width / 1000),
                    int(y0 * height / 1000),
                    int(x1 * width / 1000),
                    int(y1 * height / 1000),
                )
                w = int((x1 - x0) * WIDTH_THRESHOLD)
                h = int((y1 - y0) * HEIGHT_THRESHOLD)
                cv2.rectangle(Y, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), cv2.FILLED)

        else:
            #     name = file[:len(file) - 4]
            #     df = pd.read_csv(GROUND_TRUTH_DIR+name+'.txt', delimiter=' ',
            #                     names=["label","confidence","x0","y0",'w','h'])

            #     for i in range(df.shape[0]):
            #         x0, y0, w, h  = (df['x0'][i], df['y0'][i], df['w'][i], df['h'][i])
            #         w = int(w*WIDTH_THRESHOLD)
            #         h = int(h*HEIGHT_THRESHOLD)
            #         cv2.rectangle(Y, (x0, y0), (x0+w, y0+h), (0, 0, 0), cv2.FILLED)

            name = file[: len(file) - 4]
            df = pd.read_csv(
                GROUND_TRUTH_DIR + name + ".txt",
                delimiter=" ",
                names=["label", "x", "y", "x2", "y2"],
            )

            for i in range(df.shape[0]):
                x0, y0, x2, y2 = (df["x"][i], df["y"][i], df["x2"][i], df["y2"][i])
                x2 = int(x2 * WIDTH_THRESHOLD)
                y2 = int(y2 * HEIGHT_THRESHOLD)
                cv2.rectangle(Y, (int(x0), int(y0)), (x2, y2), (0, 0, 0), cv2.FILLED)

    gold_label = get_label(Y)

    path_json = "sms_json.json"
    T_path_pkl = "pickle_T.pkl"  # test data - have true labels
    U_path_pkl = "pickle_U.pkl"  # unlabepickle_Tlled data - don't have true labels

    log_path_cage_1 = "sms_log_1.txt"  # cage is an algorithm, can be found below

    sms_noisy_labels = PreLabels(
        name="sms",
        data=X,
        gold_labels=gold_label,
        rules=rules,
        labels_enum=pixelLabels,
        num_classes=2,
    )
    sms_noisy_labels.generate_pickle(T_path_pkl)
    sms_noisy_labels.generate_json(path_json)  # generating json files once is enough

    sms_noisy_labels = PreLabels(
        name="sms", data=X, rules=rules, labels_enum=pixelLabels, num_classes=2
    )  # note that we don't pass gold_labels here, for the unlabelled data
    sms_noisy_labels.generate_pickle(U_path_pkl)

    cage = Cage(path_json=path_json, n_lfs=n_lfs)

    probs = cage.fit_and_predict_proba(
        path_pkl=U_path_pkl,
        path_test=T_path_pkl,
        path_log=log_path_cage_1,
        qt=prob_arr,
        qc=prob_arr,
        metric_avg=["binary"],
        n_epochs=50,
        lr=0.01,
    )
    labels = np.argmax(probs, 1)
    x, y, _ = Y.shape

    labels = labels.reshape(x, y)
    io.imsave(RESULTS_DIR + file, labels)


### Main Code
# if __name__ == "__main__":
# def run(img_file: str):
# dir_list = os.listdir(INPUT_IMG_DIR)

# gt_format = GroundTruthFormat.DocTR

### CAGE Execution
# for img_file in tqdm(dir_list):
# if not (os.path.exists(RESULTS_DIR + img_file)):
# if "__name__" == "__main__":
# img_file = "ABBv1_1-109_0_ori.jpg"
# print(f"File name is {file}")


# lf = Labeling(imgfile=img_file, model=MODEL)
# # print(f"image shape is {}")
# print(lf)
# cage(img_file, lf.pixels)
# bboxes = get_bboxes(img_file)

# subprocess.run(
#     [
#         "python",
#         "./iou-results/pascalvoc.py",
#         "-gt",
#         "../" + GROUND_TRUTH_DIR,
#         "-det",
#         "../" + OUT_TXT_DIR,
#     ]
# )

# print("\n\nPASCALVOC RAN SUCCESSFULLY\n\n")
# ### SPEAR EXECUTION
# df = pd.DataFrame()
# # for img in tqdm(dir_list):
# lf = Labeling(imgfile=img_file, model=MODEL)
# result = analysis(img_file)
# df = df.append(result)

# df.to_csv("results_only_some.csv", index=False)


class Textron(nn.Module):
    def __init__(
        self, 
        model=MODEL, 
        pretrained:bool=True, 
        pretrained_backbone:bool=True, 
        assume_straight_pages:bool=True
        ) -> None:
        super().__init__()
        self.model = model
        self.pretrained = pretrained
        self.pretrained_backbone = pretrained_backbone
        self.assume_straight_pages = assume_straight_pages
    def postprocess(self, imgfile: str):
        preds = get_bboxes(imgfile)
        return preds
    
    @torch.enable_grad()
    def forward(self, imgfile):
        global lf
        lf = Labeling(imgfile=imgfile, model=self.model)
        # print(f"image shape is {}")
        # print(lf)
        cage(imgfile, lf.pixels)
        
        
        @preprocessor()
        def get_chull_info(x):
            return lf.CHULL[x[0]][x[1]]


        @preprocessor()
        def get_edges_info(x):
            return lf.EDGES[x[0]][x[1]]


        @preprocessor()
        def get_pillow_edges_info(x):
            return lf.PILLOW_EDGES[x[0]][x[1]]


        @preprocessor()
        def get_doctr_info(x):
            return lf.DOCTR[x[0]][x[1]]


        @preprocessor()
        def get_tesseract_info(x):
            return lf.TESSERACT[x[0]][x[1]]


        @preprocessor()
        def get_contour_info(x):
            return lf.CONTOUR[x[0]][x[1]]


        @preprocessor()
        def get_title_contour_info(x):
            return lf.TITLE_CONTOUR[x[0]][x[1]]


        @preprocessor()
        def get_mask_holes_info(x):
            return lf.MASK_HOLES[x[0]][x[1]]


        @preprocessor()
        def get_mask_objects_info(x):
            return lf.MASK_OBJECTS[x[0]][x[1]]


        @preprocessor()
        def get_segmentation_info(x):
            return lf.SEGMENTATION[x[0]][x[1]]
        
        
        
        @labeling_function(label=pixelLabels.NOT_TEXT, pre=[get_chull_info], name="CHULL_PURE")
        def CONVEX_HULL_LABEL_PURE(pixel):
            if pixel:
                return pixelLabels.NOT_TEXT
            else:
                return ABSTAIN


        @labeling_function(label=pixelLabels.TEXT, pre=[get_chull_info], name="CHULL_NOISE")
        def CONVEX_HULL_LABEL_NOISE(pixel):
            if pixel:
                return pixelLabels.TEXT
            else:
                return ABSTAIN


        @labeling_function(label=pixelLabels.TEXT, pre=[get_edges_info], name="SKIMAGE_EDGES")
        def EDGES_LABEL(pixel):
            if pixel:
                return pixelLabels.TEXT
            else:
                return ABSTAIN


        @labeling_function(
            label=pixelLabels.NOT_TEXT, pre=[get_edges_info], name="SKIMAGE_EDGES_REVERSE"
        )
        def EDGES_LABEL_REVERSE(pixel):
            if pixel:
                return ABSTAIN
            else:
                return pixelLabels.NOT_TEXT


        @labeling_function(
            label=pixelLabels.TEXT, pre=[get_pillow_edges_info], name="PILLOW_EDGES"
        )
        def PILLOW_EDGES_LABEL(pixel):
            if pixel:
                return pixelLabels.TEXT
            else:
                return ABSTAIN


        @labeling_function(
            label=pixelLabels.NOT_TEXT, pre=[get_pillow_edges_info], name="PILLOW_EDGES_REVERSE"
        )
        def PILLOW_EDGES_LABEL_REVERSE(pixel):
            if pixel:
                return ABSTAIN
            else:
                return pixelLabels.NOT_TEXT


        @labeling_function(label=pixelLabels.TEXT, pre=[get_doctr_info], name="DOCTR")
        def DOCTR_LABEL(pixel):
            if pixel:
                return pixelLabels.TEXT
            else:
                return ABSTAIN


        @labeling_function(label=pixelLabels.TEXT, pre=[get_doctr_info], name="DOCTR2")
        def DOCTR_LABEL2(pixel):
            if pixel:
                return pixelLabels.TEXT
            else:
                return ABSTAIN


        @labeling_function(label=pixelLabels.TEXT, pre=[get_tesseract_info], name="TESSERACT")
        def TESSERACT_LABEL(pixel):
            if pixel:
                return pixelLabels.TEXT
            else:
                return ABSTAIN


        @labeling_function(label=pixelLabels.TEXT, pre=[get_contour_info], name="CONTOUR")
        def CONTOUR_LABEL(pixel):
            if pixel:
                return pixelLabels.TEXT
            else:
                return ABSTAIN


        @labeling_function(
            label=pixelLabels.TEXT, pre=[get_title_contour_info], name="CONTOUR_TITLE"
        )
        def CONTOUR_TITLE_LABEL(pixel):
            if pixel:
                return pixelLabels.TEXT
            else:
                return ABSTAIN


        @labeling_function(
            label=pixelLabels.NOT_TEXT, pre=[get_mask_holes_info], name="MASK_HOLES"
        )
        def MASK_HOLES_LABEL(pixel):
            if pixel:
                return pixelLabels.NOT_TEXT
            else:
                return ABSTAIN


        @labeling_function(
            label=pixelLabels.NOT_TEXT, pre=[get_mask_objects_info], name="MASK_OBJECTS"
        )
        def MASK_OBJECTS_LABEL(pixel):
            if pixel:
                return pixelLabels.NOT_TEXT
            else:
                return ABSTAIN


        @labeling_function(
            label=pixelLabels.TEXT, pre=[get_segmentation_info], name="SEGMENTATION"
        )
        def SEGMENTATION_LABEL(pixel):
            if pixel:
                return pixelLabels.TEXT
            else:
                return ABSTAIN
                
                
        return self.postprocess(imgfile)
            
    def __name__(self):
        return "textron"
    
def textron(
    pretrained: bool = True,
    pretrained_backbone: bool = True,
    assume_straight_pages: bool = True
    
) -> Textron:
    model = Textron(MODEL,pretrained, pretrained_backbone, assume_straight_pages)
    return model