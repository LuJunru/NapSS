# NapSS
Code and data for our EACL 2023 finding paper "NapSS: Paragraph-level Medical Text Simplification via Narrative Prompting and Sentence-matching Summarization" which can be found: https://arxiv.org/abs/2302.05574. We built our codes upon the repository: ["Paragraph-level Simplification of Medical Texts"](https://github.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts).

## Usage
It takes around 3.5 hours to finetune NapSS from scratch, on 2 Quadro RTX 6000 GPUs.

### Dependency
- We list the packages in our environment in env.yml file for your reference.
- All data we used are provided in `data` folder.

### Prepare datasets (50 mins)
- (30 mins) Extractive narrative key phrases: `python3 scripts/utils/extract_narrative_keys.py data/data_final_1024.json data/data_final_1024-keys.json`. You can mute pls key phrases extraction for saving time.
- (20 mins) Attach intermediate summarization labels for stage one training: `python3 scripts/utils/attach_summary_labels.py data/data_final_1024-keys.json data/data_final_1024-keyswithlabels.json`

### Train from scratch
#### Stage 1: summarization (50 mins)
- Prepare summarization data folder: `mkdir data/data-summarization`
- (12 mins) Prepare summarization train, dev and test sets: `python3 scripts/utils/create_summarization_data.py data/data_final_1024-keyswithlabels.json`
- (24 mins) Finetune classification-based summarization model: `python3 scripts/summarization/train.py`. You can turn on wandb logging by setting `os.environ["WANDB_DISABLED"] = "false"` in the script.
- (Optional) Evaluate finetuned summarization model: `python3 scripts/summarization/evaluate.py`
- Prepare new corpus folder of stage 2: `cp -r data/data-1024 data/data-1024-napss`
- (4 mins) predict over dev and test sets to create new sets of stage 2: `python3 scripts/summarization/predict.py`
- (10 mins) for train set, we don't predict but compose from pre-extracted keys and summaries: `python3 scripts/utils/create_new_trainset.py data/data_final_1024-keyswithlabels.json`

#### Stage 2: simplification (2 hours)
- (20 mins) Train NapSS: `bash scripts/train/bart-napss.sh`
- (100 mins) Generate simplified text via NapSS: `bash scripts/generate/bart-gen-napss.sh`
- (Optional) Run `bash scripts/train/bart-napss-ul.sh` for NapSS + UL joint training. We use "cochrane" settings, you can check detailed instruction for other settings in [Paragraph-level Simplification of Medical Texts](https://github.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts).
- (Optional) Generate simplified text via joint model: `bash scripts/generate/bart-gen-napss-ul.sh`
- (Optional) Use above scripts and finetuned models for zero-shot inference on OOD TICO-19 dataset: `data/data-tico19`

### Finetuned Models
- We provide finetuned models here: https://drive.google.com/drive/folders/1CHBsuI4aEfjM_ds-FzQW_Hp5JeW_ajYy?usp=sharing.

### Calculate metrics
- Calculate evaluation metrics over generated files: `python3 scripts/utils/calculate_evaluation_metrics.py trained_models/bart-no-ul-abskeys/gen_nucleus_test_1_0-500.json`
- Note: Bertscore computing is provided but muted, since it takes quite long time to compute this metric (e.g, 40 mins).

### Human evaluation
- We provide two sets (expert and non_experts) human evaluation results in the `human evaluation` folder.
- The `non_experts` set include results evaluated by 6 annotators on three asepscts: Simplicity, Fluency and Factuality. 
- The `expert` set include results evaluated by 2 annotators only on Factuality.
