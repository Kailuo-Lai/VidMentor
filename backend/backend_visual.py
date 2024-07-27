import os
import cv2
import time
from typing import List
import pandas as pd
from paddleocr import PaddleOCR
from models.keybert_model import KeyBertModel

class VisualBackend:
    def __init__(self, 
                 ocr: PaddleOCR,
                keybert: KeyBertModel,
                 args):
        '''
        Visual backend for VideoPal.
        ocr: OCR model
        translator: Translate model
        args: Arguments
            inter_seconds: The smallest time gap between successive clips, in seconds.
        '''
        self.args = args
        self.inter_seconds = args.inter_seconds
        self.ocr = ocr
        self.keybert = keybert
    
    def get_static_info_of_whole_video(self, filepath: str, video_info: List):
        '''
        Extract static information of one video per {inter_seconds} seconds.
        filepath: video file path e.g. ./data/test.mp4
        video_info: [video_name, file_name, video_length, file_path]
        '''
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = int(fps) * self.inter_seconds

        print(f'\033[1;33mStarting Extract Visual Information of Video: {video_info[0]}\033[0m')
        if os.path.isfile(f"./database/{video_info[0]}/visual_info.csv"):
            print(f'\033[1;33mVideo: {video_info[0]} visual info already exist\033[0m')
            return 

        start_time = time.perf_counter()
        
        pd.DataFrame(columns=["video_name", "time", "OCR"]).to_csv(f"./database/{video_info[0]}/visual_info.csv", index=False)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            # Get first frame and last frame
            if current_frame_pos % sample_rate == 0 or current_frame_pos == 1 or current_frame_pos == frame_count - 1:  
                # OCR part
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ocr_info = []
                ocr_result = self.ocr.ocr(image, cls=True)
                for res in ocr_result:
                    if res is not None:
                        for line in res:
                            try:
                                if line[1][0]:  # Change to 'if line[1][0]' which can automatically handle None and empty strings
                                    ocr_info.append(line[1][0])
                            except IndexError:
                                # Errors can be logged or passed
                                pass  
                ocr_info = ','.join(ocr_info)

                # Save to database
                row = pd.DataFrame([[video_info[0], cap.get(cv2.CAP_PROP_POS_FRAMES) / fps, ocr_info]])
                row.to_csv(f"./database/{video_info[0]}/visual_info.csv", mode="a", header=False, index=False)
        
        print(f'\033[1;33mFinished After {time.perf_counter() - start_time} Seconds\033[0m')
        
    def get_static_info_of_one_frame(self, filepath: str, second: int):
        '''
        Extract static information of one frame.
        filepath: video file path e.g. ./data/test.mp4
        second: second of the frame to be extracted
        '''
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = int(second * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # OCR part
            ocr_info = []
            ocr_result = self.ocr.ocr(image, cls=True)
            for res in ocr_result:
                if res is not None:
                    for line in res:
                        try:
                            if line[1][0]:  # Change to 'if line[1][0]' which can automatically handle None and empty strings
                                ocr_info.append(line[1][0])
                        except IndexError:
                            # Errors can be logged or passed
                            pass
            ocr_info = ','.join(ocr_info)
        
            return ocr_info
        
        else:
            return None
        
    def get_frame_keywords(self, filepath , second):
        ocr_info = self.get_static_info_of_one_frame(filepath , second)
        if len(ocr_info)==0:
            return []
        else:
            keywords = self.keybert.extract_keywords(ocr_info)
            return keywords