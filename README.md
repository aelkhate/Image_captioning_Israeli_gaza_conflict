# Image_captioning_Israeli_gaza_conflict
a pipeline to caption images related to the Israeli/Palestine conflict

173 images are missing from the dataset


extracted images link : 
[text](https://drive.google.com/drive/folders/1XvaDP5Rq5pVsEI9QQZHE03OhfHzi8Hua?usp=sharing)

DataSet link : 
[\[text\](https://docs.google.com/spreadsheets/d/1-dWwXWzP8dvY667nmzYdwK9nTxsdO04IpY_u_0MMB0E/edit?gid=1620162901#gid=1620162901)](https://github.com/MohamedIbrahim1708/Gaza_Israel_War_Dataset)



EDA notes

Captions:
Captions are generally single-sentence but moderately detailed (median 31 words), with a small long-tail of very verbose captions (up to 111 words). This suggests a mostly consistent descriptive annotation style with occasional outliers that may require normalization or truncation policy during training/evaluation.


Unigram and Bigram analysis: 
The raw unigram and bigram frequency analysis shows that the dataset is dominated by caption boilerplate related to image layout and broadcast overlays (e.g., “visible background”, “news logo”, “left corner”). This indicates a systematic annotation style that includes watermark/layout descriptors. Without normalization, such artifacts would bias downstream training toward generating overlay-focused captions rather than semantic descriptions of the scene.
After removing boilerplate tokens, the most frequent terms reflect semantic content relevant to Gaza–Israel imagery: descriptions of people (“man”, “woman”, “wearing”, “dark suit”), destruction (“damaged buildings”, “rubble debris”, “heavily damaged”), and crowd scenes (“crowd people”), with conflict-specific cues appearing directly (e.g., “israeli flag”). This supports using a normalization step for analysis and motivates controlling overlay artifacts during training



| Entity      | Count   | Interpretation                       |
| ----------- | ------- | ------------------------------------ |
| PERSON      | **95**  | High identity leakage                |
| ORG         | **271** | Heavy broadcast / political presence |
| NORP        | **309** | National / group labels              |
| GPE         | **199** | Desired geopolitical grounding       |
| WORK_OF_ART | 51      | Overlay slogans / headlines          |
| EVENT       | 11      | Mostly media-style labels            |
| FAC         | 15      | Hospitals, crossings                 |
| LANGUAGE    | 107     | Not semantically useful              |
| CARDINAL    | 476     | Numeric noise                        |


Rules for NER 

| Entity                    | Rule   |
| ------------------------- | ------ |
| ORG (image-grounded)      | KEEP   |
| ORG (overlay / narration) | REMOVE |
| PERSON                    | REMOVE |
| GPE                       | KEEP   |
| NORP                      | KEEP   |
| EVENT (headline style)    | REMOVE |
| WORK_OF_ART               | REMOVE |


Organizational entities were not removed indiscriminately. Only those appearing as narrative or broadcast overlays rather than image-grounded references were sanitized, preserving visually inferable organizational context.


| Stage                             | What (Beginner-friendly)                | Why (Technical reason)                                 | How (Method)                                                 | Which tools / models            | Practical implementation notes                           |
| --------------------------------- | --------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------- | -------------------------------------------------------- |
| **-1a. Raw EDA**                  | Explore caption length & frequent words | Understand dataset structure and annotation bias       | Length stats + unigram/bigram frequency                      | pandas, matplotlib              | Keep overlay words (BBC, logo, corner) for analysis only |
| **-1b. Deboilerplate EDA**        | Reveal true semantic content            | Overlay artifacts hide scene meaning                   | Remove overlay tokens → redo n-grams                         | regex, pandas                   | Confirms dominance of rubble, people, buildings          |
| **-1c. Language audit**           | Measure Arabic vs English ratio         | Mixed language breaks NLP pipelines                    | Unicode Arabic detection                                     | regex                           | Arabic ≈ 39.4%                                           |
| **-1d. Language normalization**   | Translate Arabic captions to English    | Single-language training & evaluation                  | Neural translation + caching                                 | Helsinki-NLP opus-mt-ar-en, GPU | Keep original + translated columns                       |
| **-1e. NER audit**                | Inspect entity usage in captions        | Defines sanitization rules                             | Run NER on caption_en only                                   | spaCy                           | First pass is BEFORE removing BBC                        |
| **-1f. Qualitative audit**        | Manually inspect sample captions        | Provides thesis justification                          | Review ~15 images/captions                                   | Manual                          | Focus on officials, rubble, crowds                       |
| **0. Dataset sanitization**       | Clean captions for training             | Prevent overlay overfitting, preserve conflict context | Remove overlay ORGs, keep Gaza/Israel, optional name masking | regex + NER                     | Create caption_train column                              |
| **1. Instruction formatting**     | Convert to instruction task             | Aligns model to tone & neutrality                      | Prompt template (image + instruction)                        | HF datasets, Qwen processor     | Keep prompt fixed & versioned                            |
| **2. Domain fine-tuning (QLoRA)** | Adapt model to conflict style           | Improves grounding & tone                              | QLoRA fine-tuning                                            | Qwen2.5-VL / Qwen3-VL, PEFT     | Low LR, early stopping                                   |
| **3. Visual fact extraction**     | Extract only visible facts              | Reduces hallucination                                  | Prompt structured fact list                                  | Fine-tuned Qwen                 | Bullet or JSON output                                    |
| **4. Multi-pass verification**    | Keep stable facts only                  | Filters hallucinations                                 | k-pass consensus filtering                                   | Python + spaCy                  | k=3 initial                                              |
| **5. Constrained captioning**     | Generate from verified facts            | Enforces grounding & neutrality                        | Prompt with fact list                                        | Same Qwen                       | Forbid new facts explicitly                              |
| **6. Multi-objective scoring**    | Choose best caption                     | Mathematical caption selection                         | CLIPScore + coverage + penalties                             | CLIP, spaCy                     | Log all scores                                           |
| **7. Repair loop (optional)**     | Auto-fix weak captions                  | Improves robustness                                    | Regenerate when metrics fail                                 | Same tools                      | Optional advanced stage                                  |


Baseline Model Evaluation

After preparing a sanitized, policy-aligned caption dataset, we evaluated a pretrained vision–language model as a baseline before any task-specific fine-tuning. The goal of this step was to measure how well a general-purpose image captioning model aligns with conflict-aware, empathetic, and unbiased captioning requirements.

The evaluation was conducted on a held-out validation split consisting of 132 image–caption pairs. Ground truth captions were taken from the sanitized training caption column (caption_train), which reflects the official task guidelines emphasizing factual, neutral, and empathetic descriptions.

Three standard automatic metrics were used:

BLEU to measure n-gram lexical overlap,

ROUGE-L to measure longest common subsequence similarity,

BERTScore to measure semantic similarity using contextual embeddings.

Baseline Results
Metric	Score
BLEU	5.31
ROUGE-L	0.293
BERTScore	0.3609
Discussion

The baseline model achieves relatively low BLEU and moderate ROUGE-L scores, indicating limited lexical and structural overlap with the ground truth captions. More importantly, the BERTScore reveals that semantic alignment with the policy-aligned captions remains low.

This confirms that although the pretrained model can generate generally coherent and safe captions, it does not naturally follow the task-specific captioning style, neutrality constraints, and contextual framing required by the Gaza–Israel conflict dataset.

These results justify the need for domain-specific fine-tuning to improve semantic alignment and stylistic consistency with the provided captions.

The baseline results are therefore used as a reference point for all subsequent fine-tuned models.



Fine-tuning with QLoRA substantially improved automatic captioning metrics on the validation split (132 images) using caption_train as ground truth. Compared to the baseline, the fine-tuned model increased BLEU from 5.31 to 13.11 and ROUGE-L from 0.293 to 0.370, indicating improved lexical overlap with the dataset’s sanitized caption style. Semantic similarity also improved markedly, with BERTScore F1 rising from 0.361 to 0.910. While these gains suggest the model better matches the dataset’s content and phrasing, automatic metrics can overestimate caption quality under single-reference evaluation; therefore, qualitative side-by-side comparison is included to assess factual grounding, neutrality, and tone.



| Aspect             | Baseline   | Fine-tuned               |
| ------------------ | ---------- | ------------------------ |
| Prompt leakage     | ❌ Severe   | ✅ Gone                   |
| Grounding to image | ⚠️ Partial | ✅ Stronger               |
| Length & detail    | Medium     | Longer & richer          |
| Neutrality         | Good       | Good                     |
| Overlay obsession  | High       | Lower, but still present |
| Speculation        | Low        | Higher                   |
| Hallucination risk | Medium     | Medium–low               |
