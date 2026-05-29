# CIC-NLP at SemEval-2025 Task 11: Predicting Multiclass Emotion Intensity in Text Using Transformer-Based Model

This repository contains the official system description implementation and code framework developed by the **CIC-NLP** team for **Task 11 of the SemEval-2025 Shared Task**.

This research moves beyond standard binary emotion detection to map **fine-grained emotion intensity levels on an ordinal scale of 0 to 3** across five primary human emotions: *anger, fear, joy, sadness, and surprise*.

---

##  Abstract

Emotion intensity prediction in text enhances conversational AI by enabling a deeper understanding of nuanced human emotions, a crucial yet underexplored aspect of natural language processing (NLP). This study employs Transformer-based models to classify emotion intensity levels (0–3) for five emotions: anger, fear, joy, sadness, and surprise. The dataset, sourced from the SemEval shared task, was preprocessed to address class imbalance, and model training was performed using fine-tuned *bert-base-uncased*. Evaluation metrics showed that *sadness* achieved the highest accuracy (0.8017) and F1-macro (0.5916), while *fear* had the lowest accuracy (0.5690) despite a competitive F1-macro (0.5207). The results demonstrate the potential of Transformer-based models in emotion intensity prediction while highlighting the need for further improvements in class balancing and contextual representation.

---

##  Methodology Pipeline

Predicting scalar emotion intensity (0 = Low/None, 3 = High/Extreme) presents deep dataset distribution challenges. The framework handles this via a dedicated training track:

1. **Class Imbalance Mitigation:** Preprocessing routines designed to resample and weigh highly skewed intensity counts across the target partitions.
2. **Transformer Fine-Tuning:** Leveraging a fine-tuned vanilla **BERT** (`bert-base-uncased`) core architecture.
3. **Multi-Head Output Stacks:** Modifying the classification layer to model the specific 4-tier ordinal intensity bounds for each distinct emotion sub-stream.

---

##  Evaluation Metrics per Emotion Track

The framework's predictive performance shows interesting variations in how transformers capture distinct emotional signals:

| Target Emotion | Evaluation Accuracy | Macro F1-Score | Finding / Outcome |
| --- | --- | --- | --- |
| **Sadness** | **0.8017** | **0.5916** | **Top Performer.** Strongest contextual footprint. |
| **Fear** | 0.5690 | 0.5207 | Lowest raw accuracy; high structural variance. |
| **Anger** | *[Insert]* | *[Insert]* | Competitive performance profile. |
| **Joy** | *[Insert]* | *[Insert]* | Sharp detection on peak intensities. |
| **Surprise** | *[Insert]* | *[Insert]* | Prone to class confusion markers. |

---

##  Getting Started

### Prerequisites

* Python 3.9+
* PyTorch
* Transformers (Hugging Face)
* Scikit-Learn
* Pandas, NumPy

### Installation

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/REPO_NAME.git
cd REPO_NAME
pip install -r requirements.txt

```

### Usage

1. **Preprocess and Balance SemEval Data:**
```bash
python src/preprocess_intensity.py --data_path ./data/semeval_task11

```



```
2. **Fine-tune the Intensity Tracking Transformer:**
   ```bash
   python src/train_intensity.py --model bert-base-uncased --epochs 5 --batch_size 32

```

3. **Evaluate Accuracy & F1-Macro Blocks:**
```bash
python src/evaluate_intensity.py --checkpoint ./models/best_intensity_bert.pt

```


##  Citation
If you implement this fine-grained intensity pipeline or reference the CIC-NLP SemEval-2025 benchmarks, please cite our workshop paper:

```bibtex
@inproceedings{abiola-etal-2025-cic-intensity,
    title = "{CIC}-{NLP} at {S}em{E}val-2025 Task 11: Predicting Multiclass Emotion Intensity in Text Using Transformer-Based Model",
    author = "Abiola, Tolulope Olalekan and Ojo, Olumide Ebenezer and Sidorov, Grigori and Kolesnikova, Olga and Calvo, Hiram",
    booktitle = "Proceedings of the 19th International Workshop on Semantic Evaluation (SemEval-2025)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    pages = "1616--1622"
}

```

---
