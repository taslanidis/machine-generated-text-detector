# Machine Generated Text Detector

We test the generalizable capabilities of RoBERTa to classify machine generated text in a series of different settings; from different LLMs that generated the text, to different dataset domains that the text belongs to, such as QA style questions (followupQG), wikipedia (wikitext) or general knowledge (SQuAD).

## Machine Generated Datasets

We used humanly generated data, to generate the relevant LLM text - answers.

### Question-Answer
We took human questions and human answers, to compare with LLM answers.

### Text Completion
We took human written text, and used it as a prime (20 tokens) for an LLM to continue the generation for another 140 tokens.

## How to Run
1. Download from our drive the data into the `./data` folder and preserve the skeleton.
    a. The way these data are constructed are from the `./data_engineering/download_datasets.ipynb` script to slice the human data.
    b. Then under `./scripts` there are scripts to run inference with all LLMs
    c. All scripts are named as `./scripts/run_pre_processing_*`
2. By the process above we are building our own datasets to run finetuning on RoBERTa.
    a. Run the scripts `./scripts/run_finetuning_*` to finetune RoBERTa on all settings as we describe on our paper.
3. Evaluate by running `./scripts/run_eval.sh`

Link to our drive: https://amsuni-my.sharepoint.com/:f:/g/personal/theofanis_aslanidis_student_uva_nl/EvMkEjWdMHhJjPuzfbQhRGQBswVjd1HuC0K22FXcbfb6pA 
