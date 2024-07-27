import os
import contextlib
import subprocess
import pandas as pd
import pathvalidate
import json
import numpy as np

@contextlib.contextmanager
def new_cd(x):
    d = os.getcwd()
    # This could raise an exception, but it's probably
    # best to let it propagate and let the caller
    # deal with it, since they requested x
    os.chdir(x)

    try:
        yield

    finally:
        # This could also raise an exception, but you *really*
        # aren't equipped to figure out what went wrong if the
        # old working directory can't be restored.
        os.chdir(d)

def init_database():
    os.makedirs("./database", exist_ok=True)
    pd.DataFrame(columns=["video_name", "file_name", "video_length", "file_path"]).to_csv("./database/video_info.csv", index=False)

def get_video_length(filepath):
    output = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filepath], stdout=subprocess.PIPE)
    video_length = float(output.stdout)
    return video_length

def video_name_is_duplicate(video_name: str):
    video_info_table = pd.read_csv("./database/video_info.csv")
    if video_name in video_info_table["video_name"]:
        return True
    return False

def get_video_info(filepath: str):
    filename = filepath.split('/')[-1]
    
    video_name = filename.split('.')[0]
    
    if not pathvalidate.is_valid_filename(filename):
        raise ValueError(f'\033[1;31mVideo Name: {filename} is not valid...\033[0m')
    if not pathvalidate.is_valid_filepath(f"./database/{video_name}"):
        os.makedirs(f"./database/{video_name}", exist_ok=True)
    video_length = get_video_length(filepath)
    print(f'\033[1;33mVideo Name: {filename}, Video Length: {video_length}...\033[0m')
    row = pd.DataFrame([[video_name, filename, video_length, filepath]])
    
    if not video_name_is_duplicate(video_name):
        row.to_csv("./database/video_info.csv", mode="a", header=False, index=False)
        os.makedirs(f"./database/{video_name}", exist_ok=True)
    else:
        raise ValueError(f'\033[1;31mVideo Name: {filename} is already in the database...\033[0m')
    
    return video_name, filename, video_length, filepath

def retrieve_video_info(filename: str):
    filename = filename.split('/')[-1]
    video_name = filename.split('.')[0]
    video_info_table = pd.read_csv("./database/video_info.csv")
    video_info = video_info_table[video_info_table["video_name"] == video_name]
    if video_info.empty:
        raise ValueError(f'\033[1;31mVideo Name: {filename} is not in the database...\033[0m')
    video_name, filename, video_length, filepath = video_info.values.flatten().tolist()
    return video_name, filename, video_length, filepath
    


def merge_files(data_dir):
    # Create an empty dictionary to store all merged JSON data
    merged_json = {}
    # Create an empty list to store all embeddings
    embeddings_list = []

    # Traverse all folders under data_dir
    for root, dirs, files in os.walk(data_dir):
        for dir in dirs:
            summary_path = os.path.join(root, dir, "summary.json")
            embedding_path = os.path.join(root, dir, "summary_embedding.npz")

            # Check if both files exist
            if os.path.exists(summary_path) and os.path.exists(embedding_path):
                # Read JSON file
                with open(summary_path, 'r') as f:
                    data = json.load(f)
                
                # Merge the read data into the main dictionary
                merged_json.update(data)
                
                # Read and store the embedding vectors in the NPZ file
                embedding_data = np.load(embedding_path)
                
                # Assume the embedding data is stored under the key 'embedding'
                embeddings_list.append(embedding_data['embedding'])

    # Combine all embeddings into one array
    merged_embeddings = np.concatenate(embeddings_list, axis=0)
    
    # Return the merged dictionary and array
    return merged_json, merged_embeddings


