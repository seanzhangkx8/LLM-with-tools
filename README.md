# LLM-with-Tools 🔧

Small language models (LLMs), while more efficient and easier to deploy than their larger counterparts, often lack strong reasoning capabilities and frequently make arithmetic mistakes. In this project, we explore how to enhance the reasoning and mathematical abilities of small LLMs. Specifically, we empirically study three popular approaches:

- **Fine-tuning**  
- **In-context learning (ICL)**  
- **Tool usage (e.g., calculator API)**  

Each approach has unique strengths and limitations:

|Method|Advantages|Disadvantages|
|--------------------|-|--|
|Fine-tuning|Tailors the model to specific reasoning tasks; permanent performance gains|Requires training compute; risk of overfitting; not adaptive to unseen tasks|
|In-context learning|Fast iteration; no parameter updates needed; task-flexible|Context length limits; no long-term learning; sensitive to prompt formatting|
|Tool usage |Enables accurate computation via external APIs; easy to integrate|Requires model to "know when" and "how" to invoke tools; |

In this project, we systematically test each approach and combinations of them to better understand their effectiveness—particularly for small models like LLaMA-3.2-1B-Instruct. Our goal is to make small LLMs more capable, efficient, and practically usable.

## 🔢 Dataset

We evaluate models using [GSM8K](https://huggingface.co/datasets/gsm8k), a benchmark dataset consisting of grade-school-level math word problems. We use the standard train/test split provided by the dataset.

## 🏠 Model

We use the `LLaMA-3.2-1B-Instruct` model from Hugging Face as our base model. All training and evaluation scripts are implemented by the authors. Fine-tuning is performed with LoRA on a single GPU.

---

For reproducibility, we provide all our data, training and evaluation scripts, and setup:

## 🛠️ Setup

Start a new conda or virtual environment, and install dependencies from the root directory:

```bash
pip install -r requirements.txt
```

---

## 🚀 Run Training Script

To fine-tune the model, run the following command from the root directory:

```bash
python model/train.py --save_dir model/weights/
```

---

## 📊 Evaluate Model Checkpoints

To evaluate a trained checkpoint, modify the `model_path` in `model/eval.py` (set to None to use the base model out-of-box), then run:

```bash
python model/eval.py
```

---

## 🧮 LLM + Calculator Tool Usage

To evaluate LLM with tools (calculator), use the following commands:

```bash
python llm-with-tools/llm-with-tools.py
```

---

## 🔄 In-Context Learning with Example Types

To switch between fixed and retrieval-augmented (RAG) examples for in-context learning, modify the appropriate parameter in `data.py`.

---

## 📁 Project Structure

```txt
llm-with-tools/
├── model/
│   ├── train.py           # Fine-tuning script
│   ├── eval.py            # Evaluation script
│   ├── data.py            # Dataset preprocessing and ICL control
├── llm-with-tools/
│   ├── llm-with-tools.py  # Tool-augmented inference script
├── requirements.txt
└── README.md              # Project overview
```

## Notes

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this code for both academic and commercial purposes, provided proper attribution is given.

For major re-use (e.g., incorporation into production systems, publications, or commercial applications), please contact the authors (Kai Horstmann and Sean Zhang) for permission and collaboration opportunities.
