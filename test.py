import torch
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ConvertImageDtype
from doctr.models.zoo import ocr_predictor
from doctr.models import db_resnet50

# model = indic_trocr(pretrained=True, model_path="doctr/models/trocr_files")
model = ocr_predictor(det_arch="textron", reco_arch="indic_trocr")
# img = read_image("/media/ashatya/Data/work/indic-doctr/ABBv3_4_ori.jpg")
# si = img.size()
# image = img.unsqueeze(0)
# # image = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
# transforms = Compose([
#     Resize([1024, 1024]),
#     ConvertImageDtype(torch.float)
# ])
image = "ABBv3_4_ori.jpg"
# image = transforms(image)
out = model([image])
# out = model([img])
print(out)
