import os

from transformers import VisionEncoderDecoderModel
from transformers import ViTFeatureExtractor, RobertaTokenizer, TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
from typing import Dict, Any, Callable, List, Optional, Tuple, Union

from doctr.datasets import VOCABS, decode_sequence
from ....utils.data import download_from_url

__all__ = ['indic_trocr','TROCR']

# NOTE add one config for trocr also
default_cfgs: Dict[str, Dict[str, Any]] = {
    "trocr": {
        "urls": {
            "config.json":"https://github.com/iitb-research-code/indic-trocr/releases/download/v0.0.1/config.json", # config
            "generation_config.json":"https://github.com/iitb-research-code/indic-trocr/releases/download/v0.0.1/generation_config.json", # generating config
            "optimizer.pt":"https://github.com/iitb-research-code/indic-trocr/releases/download/v0.0.1/optimizer.pt", # optimizer model
            "preprocessor_config.json":"https://github.com/iitb-research-code/indic-trocr/releases/download/v0.0.1/preprocessor_config.json", # preprocessor config
            "pytorch_model.bin":"https://github.com/iitb-research-code/indic-trocr/releases/download/v0.0.1/pytorch_model.bin", # model
            "rng_state.pth":"https://github.com/iitb-research-code/indic-trocr/releases/download/v0.0.1/rng_state.pth", # state
            "scheduler.pt":"https://github.com/iitb-research-code/indic-trocr/releases/download/v0.0.1/scheduler.pt", # scheduler
            "trainer_state.json":"https://github.com/iitb-research-code/indic-trocr/releases/download/v0.0.1/trainer_state.json", # trainer state
            "training_args.bin":"https://github.com/iitb-research-code/indic-trocr/releases/download/v0.0.1/training_args.bin" # training arguments 
        },
        "input_shape": (3, 224, 224),
        "vocab": VOCABS["devanagari"]
    }
}


class TROCR(nn.Module):
    def __init__(self, models_dir_path:Union[None,str] = "doctr/models/trocr_files") -> None:
        # NOTE change the models_dir_path to a more robust argument
        super().__init__()
        encode = 'google/vit-base-patch16-224-in21k'
        decode = 'flax-community/roberta-hindi'
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(encode)
        self.tokenizer = RobertaTokenizer.from_pretrained(decode)
        self.processor = TrOCRProcessor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
        self.model = VisionEncoderDecoderModel.from_pretrained(models_dir_path)
        
    def forward(self, x: torch.Tensor):
        # x is the image read as torch.Tensor
        pixel_values = self.processor(x, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        out: Dict[str, Any] = {}
        out["preds"] = generated_text
        
        return out
    
def indic_trocr(pretrained: bool = True, model_path = "doctr/models/trocr_files", **kwargs: Any) -> TROCR:
    """Transformer OCR"""
    if isinstance(model_path, str):
        if not os.path.isdir(model_path):
            raise NotADirectoryError(f"{model_path} not found.")
        models_dir_path = model_path 

    elif isinstance(model_path, dict):
        for file_name in model_path:
            url = model_path[file_name]
            path_downloaded_file = download_from_url(url, file_name)
            
            
            
    model = TROCR(models_dir_path = models_dir_path)
    return model