import time
import argparse
import os
import json
import numpy as np
from paddleocr import PaddleOCR
from models import whisper_model, bgemodel, punctuator_model, llm_model , keybert_model
from backend.backend_audio import AudioBackend
from backend.backend_visual import VisualBackend
from backend.backend_llm import LLMBackend
from backend.backend_search import StoreDataEmb,SearchSingleVideo,SearchMultipleVideos,SummarySingleVideo,AnswerSingleQuestion
from utils.template import template
from utils.utils import get_video_info, init_database, merge_files
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
parser = argparse.ArgumentParser()

# whisper model arguments
parser.add_argument("--whisper_version", default="medium", help="Whisper model version for video asr")

# # llm model arguments
parser.add_argument("--llm_version", default="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", help="LLM model version")
parser.add_argument("--max_context_length", default=8196, type=int, help="Maximum prompt length for LLM model")
parser.add_argument("--max_length", default=8196, type=int, help="Maximum prompt + reponse length for LLM model")

# KeyBert model arguments
parser.add_argument("--keybert_version", default="paraphrase-multilingual-MiniLM-L12-v2", help="KeyBert model version")

# bge model arguments
parser.add_argument("--bge_version", default="bge-base-en-v1.5", help="Embedding model version")
parser.add_argument("--pooling_method",default="cls",help = "Pooling method for text embedding model")
parser.add_argument("--query_instruction_for_retrieval", default="Generate a representation for this sentence to retrieve relevant articles:", help="Query instruction for retrieval")
parser.add_argument("--use_fp16", default=True, type=bool, help="Whether to use fp16")
parser.add_argument("--use_cuda", default=True, type=bool, help="Whether to use cuda")
parser.add_argument("--normalize_embeddings", default=True, type=bool, help="Whether to normalize embeddings")

# search backend arguments
parser.add_argument("--threshold", default=0.4, type=float, help="Threshold for search backend")
parser.add_argument("--timeout", default=60, type=int, help="Timeout for search backend")

# punctuator model arguments
parser.add_argument("--punctuator_version", default="xlm-roberta_punctuation_fullstop_truecase", help="Punctuator model version")

# visual backend arguments
parser.add_argument("--inter_seconds", default=5, type=int, help="The smallest time gap between successive clips, in seconds for static information.")

args = parser.parse_args()
print(args)


print('\033[1;32m' + "Initializing models...".center(50, '-') + '\033[0m')
start_time = time.perf_counter()
whisper = whisper_model.WhisperModel(args)
kerbert = keybert_model.KeyBertModel(args)
bge_model = bgemodel.BGEModel(args)
llm = llm_model.LLMModel(args)
ocr = PaddleOCR(use_angle_cls = True, lang = "ch", show_log = False) 
punctuator = punctuator_model.Punctuator(args)
print(f"\033[1;32mModel initialization finished after {time.perf_counter() - start_time}s".center(50, '-') + '\033[0m\n')

audio_backend = AudioBackend(whisper, args)
visual_backend = VisualBackend(ocr, kerbert ,args)
llm_backend = LLMBackend(llm, kerbert, bge_model, punctuator, args)

init_database()

store_data_emb = StoreDataEmb(bge_model, args)
search_single_video = SearchSingleVideo(bge_model, args)
answer_single_question = AnswerSingleQuestion(llm, template, args)
summary_single_video = SummarySingleVideo(llm, template["summary"], args)
search_multiple_videos = SearchMultipleVideos(bge_model, args)


data_dir = './database'
for file in os.listdir('./videos'):
    if file.endswith('.mp4'):
        print(f"\033[1;32mStart to process {file}".center(50, '-') + '\033[0m')

        video_info = get_video_info(f'./videos/{file}')
        audio_backend.get_audio_info(f'./videos/{file}', video_info)
        visual_backend.get_static_info_of_whole_video(f'./videos/{file}', video_info)

        markdown_result, segment_info = llm_backend.generate_segment_for_video(video_info)

        print(f"\033[1;32mAudio and visual info of {file} has been saved\033[0m")

        filepath = video_info[0]
        for csvpath in os.listdir(f"./database/{filepath}"):
            if csvpath.endswith("audio_info.csv"):
                
                store_data_emb.get_embedding(f"./database/{filepath}/{csvpath}")
                print(f"\033[1;32mEmbedding of {csvpath} has been saved\033[0m")
                
                summary_dict = {}
                summary = summary_single_video.summary(f"./database/{filepath}/{csvpath}")
                video_name = filepath
                summary_dict[video_name] = summary
                with open(f"./database/{video_name}/summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary_dict, f, ensure_ascii=False, indent=4)
                store_data_emb.get_embedding(f'./database/{video_name}/summary.json')
                print(f"\033[1;32mEmbedding of summary of {csvpath} has been saved\033[0m")
                print(f"\033[1;32m{file} has been processed\033[0m\n\n")

all_summary_json, all_summary_embeddings = merge_files(data_dir)
with open("./database/all_summary.json", "w", encoding="utf-8") as f:
    json.dump(all_summary_json, f, ensure_ascii=False, indent=4)
np.savez("./database/all_summary_embedding.npz", embedding=all_summary_embeddings)

print(f"\033[1;32mAll summary embeddings have been saved\033[0m")