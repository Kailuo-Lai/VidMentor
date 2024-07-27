import os
from punctuators.models import PunctCapSegModelONNX
from punctuators.models.punc_cap_seg_model import PunctCapSegConfigONNX

from utils.utils import new_cd

parent_dir = os.path.dirname(__file__)
checkpoints_dir = os.path.abspath(os.path.join(parent_dir, '..', 'checkpoints'))

class Punctuator:
    def __init__(self, args):
        # Build configuration based on the provided parameters
        config = PunctCapSegConfigONNX(
            spe_filename="sp.model",  # Change the filename according to the actual situation
            model_filename="model.onnx",
            config_filename="config.yaml",
            directory=os.path.join(checkpoints_dir, args.punctuator_version)  # Use absolute path
        )

        # Initialize the model
        self.punct_model = PunctCapSegModelONNX(cfg=config)
            
    def infer(self, text):
        return self.punct_model.infer(text)