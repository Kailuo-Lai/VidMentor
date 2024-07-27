from models.bgemodel import BGEModel
import csv
import numpy as np
import json
from models.llm_model import LLMModel
import os
import re
import threading

os.environ['CUDA_VISIBLE_DEVICES'] = "3"


class StoreDataEmb:
    def __init__(self,bge_model: BGEModel, args):
        self.bge_model = bge_model
        self.args = args
        
    def get_embedding(self, file_path: str):
        try:
            if file_path.endswith(".csv"):
                sentences = {}
                sentences["start_time"] = []
                sentences["end_time"] = []
                sentences["content"] = []
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        sentences["start_time"].append(row[1])
                        sentences["end_time"].append(row[2])
                        sentences["content"].append(row[3])

                # Save as json file
                output_path = f"./database/{file_path.split('/')[-2].split('.')[0]}"
                with open(f"{output_path}/audio_info.json", "w", encoding="utf-8") as f:
                    json.dump(sentences, f, ensure_ascii=False, indent=4)

                # Save as npz file
                embedding = self.bge_model.encode_corpus(sentences["content"])
                np.savez(f"./database/{file_path.split('/')[-2].split('.')[0]}/embedding.npz", embedding=embedding)
                print(f'\033[1;33mFinished Saving Embedding\033[0m')
            elif file_path.endswith(".json"):
                sentences = {}
                sentences["file"] = []
                sentences["content"] = []
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for key in data:
                        sentences["file"].append(key)
                        sentences["content"].append(data[key])
                embedding = self.bge_model.encode_corpus(sentences["content"])
                np.savez(f"./database/{file_path.split('/')[-2].split('.')[0]}/summary_embedding.npz", embedding=embedding)
                print(f'\033[1;33mFinished Saving Embedding\033[0m')
                
        except Exception as e:
            print(f"Error: {e}")
            return None
        return 
    
class SearchSingleVideo:
    def __init__(self,bge_model: BGEModel, args):
        self.bge_model = bge_model
        self.threshold = args.threshold
        self.ref_dict = {}
    def retrieval(self, question = None, json_path = None, embedding_path = None):
        if question is not None:
            self.question = question
        if json_path is not None:
            with open(json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        if embedding_path is not None:
            with open(embedding_path, "rb") as f:
                self.embedding = np.load(f)["embedding"]
        try:
            for term in self.question:
                self.ref_dict[term] = []

                scores = self.embedding @ self.bge_model.encode_queries([term]).T
                for i in range(len(scores)):
                    if scores[i] > self.threshold:

                        self.ref_dict[term].append({"start_time":self.data["start_time"][i],"end_time":self.data["end_time"][i],"content":self.data["content"][i]})
            print(f'\033[1;33mFinished Retrieval\033[0m')
            return self.ref_dict
        except Exception as e:
            print(e)
        

class AnswerSingleQuestion:
    def __init__(self,llm_model:LLMModel,template, args):
        self.llm_model = llm_model
        self.template = template
    
    def answer(self, question = None, reference = None):
        if question is not None:
            self.question = question

        self.reference = reference
        ref_answer = None
        if self.reference:
            ref_answer = self.llm_model.chat(self.template["ref"].format(reference=self.reference,question=self.question))
            print(f"\033[1;35mReference Answer: \033[0m{ref_answer.content}")
            no_ref_answer = self.llm_model.chat(self.template["no_ref"].format(question=self.question))
            print(f"\033[1;35mNo Reference Answer: \033[0m{no_ref_answer.content}")
        else:
            no_ref_answer = self.llm_model.chat(self.template["no_ref"].format(question=self.question))
            print(f"\033[1;35mNo Reference Answer: \033[0m{no_ref_answer.content}")
        if ref_answer:
            merge_ref = ref_answer.content + "." + no_ref_answer.content
            merge_answer = self.llm_model.chat(self.template["ref"].format(question = self.question,reference = merge_ref))
        else:
            merge_answer = no_ref_answer
        print(f"\033[1;35mFinal Answer: \033[0m{merge_answer.content}")
        return merge_answer.content
        

class SummarySingleVideo:
    def __init__(self,llm_model:LLMModel, template, args):
        self.llm_model = llm_model
        self.template = template
        self.max_context_length = args.max_context_length
        self.timeout = args.timeout

    def split_context(self,file_path: str, max_length=512):
        context = ""
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                text = row[3]
                context += "。 " + text
        segments = []
        while len(context) > max_length:
            position = context.rfind("。", 0, max_length)
            if position == -1:
                position = max_length
            segments.append(context[:position])
            context = context[position:]
        if context:
            segments.append(context)
        return segments
    
    def summary(self, file_path: str):
        segments = self.split_context(file_path)
        segment_summaries = []
        for segment in segments:
            summary_segment = self.run_summary(segment,timeout=self.timeout)
            if summary_segment is not None:
                segment_summaries.append(summary_segment)
        context_to_resummarize = ";".join(segment_summaries)
        final_summary = self.run_summary(context_to_resummarize, timeout=5 * self.timeout)
        return final_summary
    
    
    def run_summary(self, context,timeout = 60):
        prompt = self.template.format(query=context)
        summary = [None]  # Use a mutable object to allow modification inside the thread

        def model_call():
            successful = False
            while not successful:
                try:

                    response = self.llm_model.chat(prompt).content
                    response =  re.search(r'"summary": "([^"]+)"', response)

                    response = response.group(1)
                    summary[0] = response
                    successful = True
                    print(f"\033[1;35mSummary: \033[0m{summary[0]}")
                except Exception as e:
                    print(f"An error occurred: {e}")

        thread = threading.Thread(target=model_call)
        thread.start()
        thread.join(timeout=timeout)  # Wait for the thread to finish or timeout after 60 seconds
        if thread.is_alive():
            print("Operation timed out.")
            thread.join()  # Ensure the thread has finished before continuing
            return None

        return summary[0]
    
class SearchMultipleVideos:
    def __init__(self,bge_model: BGEModel, args):
        self.bge_model = bge_model
        self.threshold = args.threshold
        self.ref_dict = {}
    def retrieval(self, question = None, json_path = './database/all_summary.json', embedding_path = './database/all_summary_embedding.npz'):
        if question is not None:
            self.question = question
            

        if json_path is not None:
            with open(json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        if embedding_path is not None:
            with open(embedding_path, "rb") as f:
                self.embedding = np.load(f)["embedding"]
        try:
            for term in self.question:
                self.ref_dict[term] = []

                scores = self.embedding @ self.bge_model.encode_queries([term]).T
                for i in range(len(scores)):
                    if scores[i] > self.threshold:

                        self.ref_dict[term].append({"video_name":list(self.data.keys())[i],"summary":self.data[list(self.data.keys())[i]]})
            return self.ref_dict
        except Exception as e:
            print(e)
    
    
    
    
    
    
    
    
    
    
    
    
    
    