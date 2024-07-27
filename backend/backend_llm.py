import csv
import torch
import json
import pandas as pd
from collections import defaultdict
from langchain import PromptTemplate
from typing import List

from models.keybert_model import KeyBertModel
from models.llm_model import LLMModel
from models.bgemodel import BGEModel
from models.punctuator_model import Punctuator
from utils.tree import get_leaf_node_paths, parse_markdown_to_tree




class LLMBackend():
    def __init__(self,
                 llm_model: LLMModel,
                 keybert_model: KeyBertModel,
                 bge_model: BGEModel,
                 punct_model: Punctuator,
                 args):
        '''
        Args:
            llm_model: LLMModel
            keybert_model: KeyBERT
            text_embedding_model: TextEmbedding
            punct_model: Punctuator
        Returns:
            None
        '''
        self.llm_model = llm_model
        self.keybert_model = keybert_model
        self.bge_model = bge_model
        self.punct_model = punct_model
        
    
    def generate_segment_for_video(self, video_info):
        '''
        generate segment info for video and save to file
        input: video_info: [video_name, file_name, video_length, file_path]
        output: markdown_result, segment_info
        '''
        
        system_message = """
        You are an expert in extracting and summarizing information from educational videos. Your task is to summarize the information from the input text and extract key points, returning the results in markdown format.

        **Input format:**
        The input will be a text containing the description of the video and the dialogue of the characters in the video. You need to summarize this and return it in markdown format.

        **Output format:**
        A summary of the key points using markdown syntax, consisting only of headings from levels one to three. Level three headings can have additional explanations, while level one and two headings should contain only keywords.

        **Requirements:**
        1. Each key point should be returned in markdown heading format, like # Heading 1, ## Heading 2, ### Heading 3, with a line break between each key point.
        2. Headings from levels one to three should contain only keywords, and these keywords should be summarizing.
        3. Think about the hierarchical relationship between different key points before summarizing, ensuring the accuracy of the heading levels.
        4. A level one heading must precede a level two heading, and similarly for other heading levels.
        5. The level three headings under the same level two heading should be as similar as possible.
        6. If there are additional explanations, they should start on a new line without using "-" or "#" characters to avoid interfering with the key points. The explanations should be as brief and concise as possible.
        7. Each level three heading should have a corresponding brief description, while level one and two headings should not have descriptions.
        8. Answer strictly according to the requirements and do not add any additional explanations at the end or beginning, for example do not answer any explanations like "Note:...",
        """

        PromptTemplate = """
        **Text:**
        {query}

        **Your answer:**
        """

        
        reader = csv.reader(open(f'./database/{video_info[0]}/audio_info.csv',encoding='utf-8'))
        next(reader)
        document = ''

        data = []

        for line in reader:
            _, start,end,text = line
            start = round(float(start),2)
            end = round(float(end),2)
            dur = round(float(end)-start,2)
            
            patch = data[-1] if len(data) > 0 else {}
            if patch and patch.get('durs',1e6)<30:
                patch['content']+=text+','
                patch['durs']+=dur
            else:
                patch = defaultdict(any)
                patch['start']=start
                patch['durs']=dur
                patch['content']=text+','
                data.append(patch)

        for patch in data: 
            text = patch['content']
            patch['keyword']= self.keybert_model.extract_keywords(text)
            start = round(patch['start'],2)
            end = round(start+patch['durs'],2)
            document+=text+'\n'
            
        prompt = PromptTemplate.format(query=document)
        markdown_result = self.llm_model.chat(prompt,system_message)

        tree = parse_markdown_to_tree(markdown_result.content,tree_name=video_info[0])
        leaves_info = get_leaf_node_paths(tree)

        data_emb = self.bge_model.encode([item['content'] for item in data])
        leaves_emb = self.bge_model.encode([info for info in leaves_info])

        sims, idxs = torch.topk(torch.tensor(data_emb@leaves_emb.T),1,dim=-1)
        
        segment_info = []
        for i,idx in enumerate(idxs):
            idx = idx.numpy()[0]
            if segment_info and leaves_info[idx]==segment_info[-1][-1]:
                segment_info[-1][1]=round(data[i]['start']+data[i]['durs'],2)
            else:
                segment_info.append([data[i]['start'],round(data[i]['start']+data[i]['durs'],2),leaves_info[idx]])
        segment_info = {k.replace('*',''):[s,e] for s,e,k in segment_info}
        
        json.dump(segment_info, open(f"./database/{video_info[0]}/segment_info.json", "w", encoding='utf-8'), ensure_ascii=False, indent=4)
        with open(f"./database/{video_info[0]}/markdown_result.txt", "w", encoding='utf-8') as f:
            f.write(markdown_result.content)
        
        return markdown_result.content, segment_info
    
    def generate_question_by_clip(self, video_info: List, start: int, end: int, question_num: int = 5, question_type: str = "Multiple choice"):
        '''
        generate question for video clip
        input: 
            video_info: [video_name, file_name, video_length, file_path]
            start: start time of the clip (in seconds)
            end: end time of the clip (in seconds)
        output: question
        '''
        audio_info_tb = pd.read_csv(f"./database/{video_info[0]}/audio_info.csv")
        
        if question_type not in ["Multiple choice", "True or False", "Short answer"]:
            raise ValueError("question_type should be one of 'Multiple choice', 'True or False', 'Short answer'")
        
        prompt_question = """
        You are an excellent English question maker. Your task is to provide users with the given number of questions based on the given information.
        For example, if the question type is multiple choice and the number of questions is 3, then you should help the user to come up with 3 multiple choice questions, and each question should provide the correct answer, and so on.
        Make sure that the questions you provide match the user's requirements, question type, and number of questions. Do not make up random questions, and do not include irrelevant symbols.
        If the question type is true or false, then you need to provide questions that need to be judged right or wrong and provide the correct answer without saying anything extra.
        If the question type is a short answer question, then you need to ask questions in Chinese and provide the correct answer in Chinese as concisely as possible.
        If the given information below is not enough to ask questions, please reply directly with "Not related to the textbook, I cannot complete your task".
        If you can answer, please do not output the words in the prompt.
        
        Given information:
        {INFO}
        
        User requirements:
        {QUERY}
        
        Question type:
        {TYPE}
        
        Number of questions:
        {NUM}
        
        Please express in English.
        """
        
        prompt_question = PromptTemplate.from_template(prompt_question)
        audio_info = ["".join(audio_info_tb[(audio_info_tb['start_time'] >= start) & (audio_info_tb['end_time'] <= end)]["content"].to_list())]
        audio_info = self.punct_model.infer(text=audio_info)
        
        prompt_question = prompt_question.format(INFO="".join(audio_info[0]), QUERY="Please express in English", TYPE=question_type, NUM=question_num)
        
        questions = self.llm_model.chat(prompt_question)
        
        return questions.content
    
    def generate_questions(self, video_info, question_num: int = 5):
        '''
        generate question for video by clip
        Args:
            video_info: [video_name, file_name, video_length, file_path]
        Returns:
            question_tb: DataFrame(start, end, question)
        '''
        segment_info = json.load(open(f"./database/{video_info[0]}/segment_info.json", "r", encoding='utf-8'))
        question_tb = []
        for key, value in segment_info.items():
            questions = self.generate_question_by_clip(video_info, value[0], value[1], question_num)
            question_tb.append([value[0], value[1], key, questions])
        
        question_tb = pd.DataFrame(question_tb, columns=["start", "end", "key_title", "question"])
        question_tb.to_csv(f"./database/{video_info[0]}/questions.csv", index=False)
        
        return question_tb