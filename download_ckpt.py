import whisper
from huggingface_hub import snapshot_download

# LLM
snapshot_download(repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                  local_dir="./checkpoints/Meta-Llama-3-8B-Instruct",
                  local_dir_use_symlinks=False)

# medium
whisper.load_model('medium', download_root='./checkpoints/whisper-medium',
                   local_dir_use_symlinks=False)

# Embeddings
snapshot_download(repo_id='BAAI/bge-base-en-v1.5',
                  local_dir="./checkpoints/bge-base-en-v1.5",
                  local_dir_use_symlinks=False)

# KeyBERT
snapshot_download(repo_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 
                  local_dir="./checkpoints/paraphrase-multilingual-MiniLM-L12-v2",
                  local_dir_use_symlinks=False)

# Punctuator
snapshot_download(repo_id='1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase', 
                  local_dir="./checkpoints/xlm-roberta_punctuation_fullstop_truecase",
                  local_dir_use_symlinks=False)