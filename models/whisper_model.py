import os
import whisper
from utils.utils import new_cd
parent_dir = os.path.dirname(__file__)

class WhisperModel():
    def __init__(self, args):
        with new_cd(parent_dir):
            model_state_file = os.listdir(f"../checkpoints/whisper-{args.whisper_version}")[0]
            self.model = whisper.load_model(f"../checkpoints/whisper-{args.whisper_version}/{model_state_file}", device="cpu")