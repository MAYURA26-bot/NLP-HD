````markdown
# SIT770 2.3HD – Context-Aware Normalisation for Ambiguous Twitter Sentiment Classification

## Overview

This folder contains the code, outputs, and report evidence for my SIT770 2.3HD research task.

The study extends my 2.2D problem formulation task. In the D task, ambiguous/negation-based Twitter text was identified as the weakest case for zero-shot DeBERTa sentiment classification. In this HD task, I extended the experiment by:

1. Comparing three different model types.
2. Checking whether all models struggle on ambiguous/negation tweets.
3. Analysing false positives, false negatives, and real misclassified tweets.
4. Proposing a context-aware normalisation method.
5. Testing whether normalisation improves the ambiguous/negation subset.

## Models Used

| Model | Role |
|---|---|
| `cross-encoder/nli-deberta-v3-large` | Zero-shot DeBERTa baseline |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Twitter-specific fine-tuned sentiment model |
| `Qwen/Qwen3-4B` | Recent open LLM-style zero-shot model |
| `Qwen/Qwen3-4B-Instruct-2507` | Separate model used for proposed normalisation |

## Dataset

Dataset: TweetEval sentiment dataset  
Task type: Binary sentiment classification

The original labels were:

- negative
- neutral
- positive

Neutral examples were removed so the experiment used only:

- negative
- positive

The first 2,000 binary Twitter test samples were used for the main evaluation.

## Special-Case Subsets

Three Twitter text categories were evaluated:

| Case | Description |
|---|---|
| Short | Tweets with fewer than 8 words |
| Slang | Tweets containing informal/slang terms |
| Ambiguous/Negation | Tweets containing negation, uncertainty, mixed sentiment, or sarcasm-like wording |

The original subset sizes were:

| Case | Original Samples |
|---|---:|
| Short | 164 |
| Slang | 169 |
| Ambiguous/Negation | 423 |

To make the comparison fair, all categories were balanced to 164 samples each using random seed 42.

## Main Baseline Results

### Overall 2,000-Sample Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Zero-shot DeBERTa | 0.901 | 0.891 | 0.844 | 0.867 |
| Twitter-RoBERTa-latest | 0.942 | 0.915 | 0.936 | 0.925 |
| Qwen3-4B zero-shot | 0.937 | 0.947 | 0.884 | 0.915 |

### Balanced Special-Case F1 Results

| Model | Short | Slang | Ambiguous/Negation |
|---|---:|---:|---:|
| Zero-shot DeBERTa | 0.903 | 0.889 | 0.816 |
| Twitter-RoBERTa-latest | 0.958 | 0.909 | 0.853 |
| Qwen3-4B zero-shot | 0.948 | 0.929 | 0.877 |

## Main Finding Before Normalisation

The main finding is that ambiguous/negation tweets were the weakest special case across all three model types.

This supports the D-task finding more strongly because the weakness was not only seen in zero-shot DeBERTa. It also appeared in the Twitter-specific model and the Qwen LLM-style model.

## Confusion Matrix Summary

| Model | True Negative | False Positive | False Negative | True Positive |
|---|---:|---:|---:|---:|
| Zero-shot DeBERTa | 1152 | 79 | 120 | 649 |
| Twitter-RoBERTa-latest | 1164 | 67 | 49 | 720 |
| Qwen3-4B zero-shot | 1193 | 38 | 89 | 680 |

## False Positive / False Negative Error Breakdown

| Model | Error Type | Total Errors | Short | Slang | Negation | Mixed Sentiment | Emoji/Symbols |
|---|---|---:|---:|---:|---:|---:|---:|
| Zero-shot DeBERTa | False Negative | 120 | 14 | 10 | 23 | 3 | 24 |
| Zero-shot DeBERTa | False Positive | 79 | 6 | 4 | 11 | 2 | 11 |
| Twitter-RoBERTa-latest | False Negative | 49 | 4 | 6 | 15 | 3 | 9 |
| Twitter-RoBERTa-latest | False Positive | 67 | 5 | 6 | 15 | 0 | 11 |
| Qwen3-4B zero-shot | False Negative | 89 | 7 | 7 | 17 | 4 | 10 |
| Qwen3-4B zero-shot | False Positive | 38 | 4 | 2 | 8 | 0 | 8 |

Note: These category counts do not need to add up to the total error count because one tweet can match multiple patterns, and some tweets may not match any predefined pattern.

## Proposed Method

The proposed method is context-aware sentiment normalisation.

Only the ambiguous/negation subset was normalised because that was the weak case found in the D task and confirmed again in the three-model comparison.

The normalisation model was:

`Qwen/Qwen3-4B-Instruct-2507`

The prompt asked the model to:

- keep the original sentiment
- not add new facts
- not remove important clues such as emojis, slang, hashtags, negation, or sarcasm
- generate a short clarification sentence

The final classifier input used both:

1. the original tweet
2. the generated clarification

## Before and After Normalisation Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| DeBERTa before normalisation | 0.915 | 0.816 | 0.816 | 0.816 |
| DeBERTa + Normalisation | 0.939 | 0.889 | 0.842 | 0.865 |
| Twitter-RoBERTa before normalisation | 0.933 | 0.865 | 0.842 | 0.853 |
| Twitter-RoBERTa + Normalisation | 0.939 | 0.889 | 0.842 | 0.865 |
| Qwen3-4B before normalisation | 0.945 | 0.914 | 0.842 | 0.877 |
| Qwen3-4B + Normalisation | 0.933 | 0.865 | 0.842 | 0.853 |

## Normalisation Effect Summary

| Model | Before Errors | After Errors | Net Change |
|---|---:|---:|---:|
| DeBERTa | 14 | 10 | -4 |
| Twitter-RoBERTa | 11 | 10 | -1 |
| Qwen3-4B | 9 | 11 | +2 |

## Main Finding After Normalisation

Normalisation improved DeBERTa and Twitter-RoBERTa on the ambiguous/negation subset.

However, it did not improve Qwen3-4B. This means the method is useful for some classifier-style models, but it is not a universal solution for all model types.

## Files in This Folder

| File | Description |
|---|---|
| `2.3HD.pdf` | Final ACM-style research paper |
| `SIT770_2_2D_ZeroShot_DeBERTa-1.ipynb - Colab.pdf` | Exported Colab notebook output |
| `.py` file | Python version of the notebook/code |
| CSV output files | Saved result tables, examples, error breakdowns, and normalisation outputs |
| confusion matrix images | Confusion matrix plots for the three models |

## How to Reproduce

The experiment was developed and tested in Google Colab.

Recommended runtime:

- Python 3
- GPU runtime
- A100 GPU used in the final run

Main libraries:

```python
datasets
transformers
accelerate
sentencepiece
scikit-learn
seaborn
matplotlib
pandas
torch
````

Basic run order:

1. Install/import libraries.
2. Load TweetEval sentiment dataset.
3. Remove neutral samples.
4. Select first 2,000 binary Twitter test samples.
5. Create short, slang, and ambiguous/negation subsets.
6. Balance all subsets to 164 samples.
7. Run DeBERTa, Twitter-RoBERTa, and Qwen3-4B.
8. Generate performance tables and confusion matrices.
9. Run error analysis and example-level analysis.
10. Apply Qwen3-4B-Instruct normalisation to ambiguous/negation tweets.
11. Re-evaluate all three classifiers on normalised input.
12. Compare before/after results.

## Important Note

The notebook file could not be uploaded directly to GitHub, so the final Colab notebook export, Python code, and generated output files are shared through this Google Drive folder for reproducibility.

```
```
