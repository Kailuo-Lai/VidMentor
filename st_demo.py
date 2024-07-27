import time
import argparse
import os
import streamlit as st
from graphviz import Digraph
from paddleocr import PaddleOCR

from models import whisper_model, bgemodel, punctuator_model, llm_model , keybert_model
from backend.backend_audio import AudioBackend
from backend.backend_visual import VisualBackend
from backend.backend_llm import LLMBackend
from backend.backend_search import StoreDataEmb,SearchSingleVideo,SearchMultipleVideos,SummarySingleVideo,AnswerSingleQuestion
from utils.template import template
from utils.utils import retrieve_video_info
from utils.tree import parse_markdown_to_tree

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
parser = argparse.ArgumentParser()

# whisper model arguments
parser.add_argument("--whisper_version", default="medium", help="Whisper model version for video asr")

# # llm model arguments
parser.add_argument("--llm_version", default="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", help="LLM model version")
parser.add_argument("--max_length", default=8196, type=int, help="Maximum prompt + reponse length for LLM model")

# KeyBert model arguments
parser.add_argument("--keybert_version", default="paraphrase-multilingual-MiniLM-L12-v2", help="KeyBert model version")

# bge model arguments
parser.add_argument("--bge_version", default="bge-base-en-v1.5", help="Embedding model version")
parser.add_argument("--pooling_method",default="cls",help = "Pooling method for text embedding model")
parser.add_argument("--query_instruction_for_retrieval", default="", help="Query instruction for retrieval")
parser.add_argument("--use_fp16", default=True, type=bool, help="Whether to use fp16")
parser.add_argument("--use_cuda", default=True, type=bool, help="Whether to use cuda")
parser.add_argument("--normalize_embeddings", default=True, type=bool, help="Whether to normalize embeddings")

# search backend arguments
parser.add_argument("--threshold", default=0.4, type=float, help="Threshold for search backend")
parser.add_argument("--max_context_length", default=512, type=int, help="Maximum context length for search backend")
parser.add_argument("--timeout", default=60, type=int, help="Timeout for search backend")

# punctuator model arguments
parser.add_argument("--punctuator_version", default="xlm-roberta_punctuation_fullstop_truecase", help="Punctuator model version")

# visual backend arguments
parser.add_argument("--inter_seconds", default=5, type=int, help="The smallest time gap between successive clips, in seconds for static information.")

args = parser.parse_args()
print(args)


@st.cache_resource
def init(_args):

    print('\033[1;32m' + "Initializing models...".center(50, '-') + '\033[0m')
    start_time = time.perf_counter()
    whisper = whisper_model.WhisperModel(_args)
    kerbert = keybert_model.KeyBertModel(_args)

    bge_model = bgemodel.BGEModel(_args)
    llm = llm_model.LLMModel(_args)
    ocr = PaddleOCR(use_angle_cls = True, lang = "ch", show_log = False)

    punctuator = punctuator_model.Punctuator(_args)
    print(f"\033[1;32mModel initialization finished after {time.perf_counter() - start_time}s".center(50, '-') + '\033[0m')

    audio_backend = AudioBackend(whisper, _args)
    visual_backend = VisualBackend(ocr, kerbert, _args)
    llm_backend = LLMBackend(llm, kerbert, bge_model, punctuator, _args)
    
    store_data_emb = StoreDataEmb(bge_model, _args)
    search_single_video = SearchSingleVideo(bge_model, _args)
    answer_single_question = AnswerSingleQuestion(llm, template, _args)
    summary_single_video = SummarySingleVideo(llm, template["summary"], _args)
    search_multiple_videos = SearchMultipleVideos(bge_model, _args)
    
    return audio_backend, visual_backend, llm_backend, search_single_video, answer_single_question, search_multiple_videos


def wrap_text(text, max_width=20):
    """Wrap text to a specified width."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= max_width:
            if current_line:
                current_line += " "
            current_line += word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return "\n".join(lines)

def create_mind_map(node, graph=None, parent_id=None, max_width=40):
    if graph is None:
        graph = Digraph(format='svg')
        graph.attr(rankdir='LR', size='10,10')
        graph.attr('node', shape='box', style='filled', fillcolor='lightgrey', fontname='SimSun', fontsize='12')
        graph.attr('edge', arrowhead='open', arrowsize='1', fontname='SimSun', fontsize='10')
        parent_id = str(id(node))
        title = wrap_text(node.title, max_width)
        description = wrap_text(node.description, max_width)
        graph.node(parent_id, f"{title}\n{description}")
    for child in node.children:
        child_id = str(id(child))
        title = wrap_text(child.title, max_width)
        description = wrap_text(child.description, max_width)
        graph.node(child_id, f"{title}\n{description}")
        graph.edge(parent_id, child_id)
        create_mind_map(child, graph, child_id, max_width)
    return graph

# Page navigation function
def navigate_to_page(page_name):
    st.session_state['page'] = page_name
    
# Home page content
def home_page():
    print("\033[1;34mNavigate to home_page\033[0m")
    
    # Set the page title
    left_co, cent_co,last_co = st.columns(spec=[1, 2, 1])
    with cent_co:
        st.image("./asset/logo.jpg",)
    st.markdown("""
                ## Welcome to VidMentor🦙
                VidMentor is a LLM-based assistant that can help you watch educational videos more efficiently.
                """)
    
    # Create a search box for users to enter their queries
    user_input = st.text_input("", "", placeholder="What do you want to know😊")
    # Create a button to submit user input
    if st.button("Search (Click twice)") and user_input:
        # If the user input is not empty, store the user input in the session state and navigate to the multiple videos page
        st.session_state.user_input = user_input
        navigate_to_page("multiple_videos_page")


def multiple_videos_page():
    st.markdown('## Research Results')
    if st.button("Return to homepage (Click twice)"):
        navigate_to_page("home")
    user_input = st.session_state.user_input
    print(f"\033[1;34mNavigate to multiple_videos_page with user input: {user_input}\033[0m")
    if user_input:
        reference_dict = search_multiple_videos.retrieval([user_input])
        reference = reference_dict[user_input]

        # Create a container to hold all the videos
        cols = st.columns(4)  # Create four columns for the horizontal arrangement of four videos
        index = 0  # Index of the current column

        # traverse all the video references found
        for video_info in reference:
            video_name = video_info['video_name']
            video_path = next((os.path.join('./videos', f) for f in os.listdir('./videos') if video_name.split('.')[0] in f), None)
            if video_path:
                video_file = open(video_path, 'rb')
                video_bytes = video_file.read()
                video_file.close()
                with cols[index]:
                    st.video(video_bytes)
                    video_title = video_info['video_name']
                    
                    if st.button(video_title, key=video_name):  # Use the button to trigger video playback
                        if "video" not in st.session_state:
                            st.session_state.selected_video = video_name
                        navigate_to_page("single_video_page")
                    with st.expander("Click to see more"):
                        st.markdown(f"<p style='font-size:small;'>{video_info['summary']}</p>", unsafe_allow_html=True)
                # Renew the column index to achieve horizontal arrangement of videos
                index = (index + 1) % 4
                
def single_video_page():
    video_name = st.session_state.selected_video
    print(f"\033[1;34mNavigate to single_video_page with video name: {video_name}\033[0m")

    # Retrieve video information from video info
    video_info = retrieve_video_info(f'./videos/{st.session_state.selected_video}.mp4')
    video_path = video_info[-1]
    summary_path = './database/' + st.session_state.selected_video
    video_length = video_info[2]
    
    if video_path:
        tree = parse_markdown_to_tree(open(f'{summary_path}/markdown_result.txt','r',encoding='utf-8').read(),st.session_state.selected_video)
        segments = eval(open(f'{summary_path}/segment_info.json','r',encoding='utf-8').read())
        mind_map = create_mind_map(tree)
        
        st.markdown('## Mind Map')
        st.container(height=300).graphviz_chart(mind_map, use_container_width=True)

        st.markdown('## Video Player')
        select_part = st.selectbox('Choose the video segment you want to watch:', segments.keys(), format_func=lambda x: f'{":->".join(x.split(":->")[-2:])}')

        # Key words of the current frame
        slider_value = st.slider('Move to select the time point to extract keywords', 0, int(video_length), 0)

        # Listen for changes in the slider and adjust the video progress
        if select_part!=st.session_state.last_selected_part:
            st.video(video_path, start_time=segments[select_part][0])
            st.session_state.last_selected_part = select_part
        elif slider_value!=st.session_state.last_slider_value:
            st.video(video_path, start_time=slider_value)
            st.session_state.last_select_second = slider_value
        else:
            pass
        
        st.sidebar.title("VidMentor🦙")
        st.sidebar.subheader('Select a mode')
        query_mode = st.sidebar.selectbox('Query mode', ['Keywords', 'Q&A', 'Generate questions'])

        if query_mode=='Q&A':
            print("\033[1;32mQ&A mode\033[0m")
            user_input = st.sidebar.text_input('Please enter your question:', '')

            # Submit the user input
            if st.sidebar.button('Submit'):
                if user_input:
                    print(f"\033[1;33mQ&A with user input:{user_input}\033[0m")
                    with st.chat_message("user"):
                        st.write(user_input)
                        
                    # Add the user input to the conversation history
                    st.session_state.conversation_history.append(f"User: {user_input}")
                    
                    # Get the model response
                    response =  answer_single_question.answer(question = [user_input])
                    
                    # Add the model response to the conversation history
                    st.session_state.conversation_history.append(f"LLM: {response}")
                    
                    # Empty the user input box
                    user_input = ''
                    
                    with st.chat_message("assistant"):
                        st.write(response)

        elif query_mode=='Keywords':
            st.sidebar.write("Click the keyword to generate information")
            print("\033[1;32mKeywords mode\033[0m")
            kws = visual_backend.get_frame_keywords(filepath=video_path, second = slider_value)
            print(f"\033[1;33mKeywords at {slider_value}s: {kws}\033[0m")
            if len(kws)>0:   
                # Create a row of buttons
                cols = st.columns(len(kws))
                button_clicked = None
                for col, kw in zip(cols, kws):
                    with col:
                        if st.button(kw):
                            button_clicked = kw
                
            elif len(kws) == 0:
                st.write("Didn't find any keywords")

            # Display the answer below the button
            if len(kws) > 0 and button_clicked:
                print(f"\033[1;33mGenerate answer for keyword: {button_clicked}\033[0m")
                # Search for video information related to keywords
                reference_dict = search_single_video.retrieval(
                    [button_clicked], 
                    json_path=f'./database/{st.session_state.selected_video}/audio_info.json',
                    embedding_path=f'./database/{st.session_state.selected_video}/embedding.npz'
                )
                reference = ""
                for term in reference_dict[button_clicked]:
                    reference += term["content"] + "。"

                # Answer questions using this reference information
                answer = answer_single_question.answer(question=[button_clicked], reference=reference)
                with st.chat_message("assistant"):
                    st.write(answer)
                    
        elif query_mode=='Generate questions':
            print("\033[1;32mGenerate questions mode\033[0m")
            questions_num = st.sidebar.slider('Number of questions', 1, 5, 1)
            if st.sidebar.button('Generate'):
                print(f"\033[1;33mGenerate {questions_num} questions for video segment: {select_part}\033[0m")
                
                questions = llm_backend.generate_question_by_clip(video_info, segments[select_part][0], segments[select_part][1], questions_num)
                print(f"\033[1;35mQuestions: \033[0m{questions}")
                
                with st.chat_message("assistant"):
                    for line in questions.split("\n"):
                        st.write(line)
        
        if st.sidebar.button("Return to homepage (Click twice)"):
            navigate_to_page("home")
 


audio_backend,visual_backend,llm_backend,search_single_video,answer_single_question,search_multiple_videos = init(args)

if 'last_slider_value' not in st.session_state:
    st.session_state.last_slider_value = -1 
if 'last_selected_part' not in st.session_state:
    st.session_state.last_selected_part = ''
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
    
# Control which page to display
if st.session_state['page'] == 'home':
    home_page()
elif st.session_state['page'] == 'multiple_videos_page':
    multiple_videos_page()
elif st.session_state['page'] == 'single_video_page':
    single_video_page() 
