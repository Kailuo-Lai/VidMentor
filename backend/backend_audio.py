import os
import time
import tqdm
from typing import List
import pandas as pd

from models.whisper_model import WhisperModel

class AudioBackend:
    def __init__(self, whisper_model: WhisperModel, args) -> None:
        '''
        Audio backend for VideoPal.
        whispper_model: Whisper model
        args: Arguments
        '''
        self.whisper_model = whisper_model
        self.args = args
        
    def get_audio_info(self, filepath: str, video_info: List, interval: int = 15):
        '''
        filepath: video file path e.g. ./data/test.mp4
        video_info: [video_name, file_name, video_length, file_path]
        n_sentences: number of sentences to be combined
        '''
        print(f'\033[1;33mStarting Extract ASR Information of Video: {video_info[0]}\033[0m')
        start_time = time.perf_counter()
        if os.path.isfile(f"./database/{video_info[0]}/audio_info.csv"):
            print(f'\033[1;33mVideo: {video_info[0]} asr info already exist\033[0m')
            return 
        
        pd.DataFrame(columns=["video_name", "start_time", "end_time", "content"]).to_csv(f"./database/{video_info[0]}/audio_info.csv", index=False)
        
        audio_result = self.whisper_model.model.transcribe(filepath, task = "transcribe")
        
        tmp_result = []
        for segment in tqdm.tqdm(audio_result["segments"]):
            if segment['no_speech_prob'] < 0.5:  
                tmp_result.append(segment)
            if len(tmp_result)>0 and tmp_result[-1]['end'] - tmp_result[0]['start'] >= interval:
                row = pd.DataFrame([[video_info[0],
                                     tmp_result[0]['start'],
                                     tmp_result[-1]['end'],
                                     ','.join([segment['text'] for segment in tmp_result])]])
                row.to_csv(f"./database/{video_info[0]}/audio_info.csv", mode="a", header=False, index=False)
                tmp_result = []
                
        # add last segment
        row = pd.DataFrame([[video_info[0],
                        tmp_result[0]['start'],
                        tmp_result[-1]['end'],
                        ','.join([segment['text'] for segment in tmp_result])]])
        row.to_csv(f"./database/{video_info[0]}/audio_info.csv", mode="a", header=False, index=False)
        
        print(f'\033[1;33mFinished After {time.perf_counter() - start_time} Seconds\033[0m')