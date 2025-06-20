# qwen_recall
Microsoft Recall imitator based on local Qwen2.5-VL-7B.

## Quick Start
Install dependencies.
```
pip install -r requirements.txt
```
Note you need to install GPU version of pytorch to enable GPU acceleration.

Download Qwen2.5-VL-7B. e.g. From Modelscope.
```
mkdir qwenVL
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir ./qwenVL
```

Download Qwen3-Embedding-0.6B. e.g. From Modelscope.
```
mkdir qwenEmbedding
modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir ./qwenEmbedding
```

Download Qwen3-Reranker-0.6B. e.g. From Modelscope.
```
mkdir qwenReranking
modelscope download --model Qwen/Qwen3-Reranker-0.6B --local_dir ./qwenReranking
```

Download CLIP model and put .safetensors weight file under ./clipModels.

The final file structure should be like:
```
qwen_recall/  
├── clipModels/  
│   └── open_clip_model.safetensors 
├── qwenEmbedding/
│   └── //Omitted
├── qwenReranking/
│   └── //Omitted
├── qwenVL/
│   └── //Omitted
├── .gitignore
...
└── //Omitted
```

Change the paths in utils.py to a valid path to store cached images and descriptions. The file structure should be like:
```
qwen_memory/
├── images/
└── descriptions.txt
```

Finally, run the script with streamlit:
```
streamlit run main.py
```

This project is developed on Windows 11, and tested with Windows 11, amd64 with RTX4060. The frequency is about 25s/item on this setup.