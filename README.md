# LLM-with-tools
Make the smaller model use tools to enhance performance

## Run Training Script

Run following command from root directory:
```
python model/train.py --save_dir model/weights/
```

## Evaluate Model Checkpoints

Change path to model checkpoint in eval.py, model_path (set to None if test out-of-box model); then, run from root directory:
```
python model/eval.py
```

To evaluate LLM with tools (calculator), use the following commands:
```
python llm-with-tools/llm-with-tools.py
```

To change extracted example type (fixed or with RAG) for in-context learning, change the parameter in data.py