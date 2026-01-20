flowchart TD
  A[dataset.csv<br/>ID, Img URL, Img Name, Desc (S)] --> B[Stage -1 EDA (raw)<br/>length stats + raw n-grams]
  B --> C[Stage -1 EDA (deboilerplate)<br/>remove overlay tokens + semantic n-grams]
  A --> D[Stage -1 Language audit<br/>detect Arabic captions]
  D --> E[Stage -1 Language normalization<br/>translate Arabic→English]
  E --> F[dataset_normalized_en.csv<br/>caption_original + is_arabic + caption_en]

  F --> G[Stage -1e NER audit on caption_en<br/>PERSON/GPE/ORG stats]
  F --> H[Stage -1f Qualitative audit<br/>~15 examples + notes]

  F --> I[Stage 0 Text normalization rules<br/>keep Gaza/Israel terms<br/>suppress overlay boilerplate<br/>redact private names if needed]
  I --> J[Stage 1 Split<br/>train/val/test with fixed seed]

  J --> K[Stage 2 Baseline inference<br/>base VLM captions]
  J --> L[Stage 3 Fine-tuning<br/>Qwen-VL + QLoRA]
  L --> M[Stage 4 Two-pass inference (self-refinement)<br/>Pass1: facts → Pass2: final caption]
  M --> N[Stage 5 Post-gen filter<br/>rules: no names, neutral, non-celebratory]

  K --> O[Stage 6 Evaluation<br/>BERTScore/ROUGE + hallucination checks<br/>human rubric]
  N --> O
  O --> P[Stage 7 Report + Reproducibility<br/>scripts, configs, seeds, model card]
