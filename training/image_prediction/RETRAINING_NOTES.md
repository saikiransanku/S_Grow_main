## Retraining Plan: Feature Fusion + Hard Negative Mining

### 1. Feature Fusion (Architecture Fix)
- Current inference now exposes heuristic signals (for example `heuristic_lesion_count`) in API output.
- For retraining, create a dual-input model:
  - Input A: image tensor (CNN backbone).
  - Input B: engineered heuristic vector:
    - `lesions_detected`
    - `dark_center_presence`
    - `yellow_chlorosis_percent_of_leaf`
    - `leaf_area_ratio_percent`
    - `venation_contrast_score`
- Concatenate `[cnn_embedding, heuristic_vector]` before final dense classifier.
- Add class-weighted loss to penalize false `healthy` predictions when lesions > 0.

### 2. Hard Negative Mining (Training Fix)
- Build a hard-negative split from green leaves with subtle spots or mild chlorosis.
- Label these as diseased classes (not healthy).
- Keep a dedicated validation bucket of hard negatives and track:
  - `healthy_precision`
  - `healthy_recall`
  - `false_healthy_rate`
- Reject model versions where hard-negative `false_healthy_rate` is above threshold.

### 3. Data Augmentation Focus
- For disease classes (especially mild lesions), apply:
  - rotation (`-20` to `+20` degrees),
  - center/leaf-focused crops,
  - brightness/contrast jitter,
  - slight blur/noise.
- Add more healthy hard negatives with shiny, tubular, glare-heavy leaves so the symptom detector learns to ignore reflections and specular highlights.
- Do not over-augment healthy class compared to mild-disease class, but do expand healthy examples that currently cause false lesion alarms.

### 4. Deployment Gate
- Keep the production logic gate:
  - IF predicted class contains `healthy` AND confidence is `>= 95%`:
    - bypass/suppress lesion analysis output.
  - ELSE:
    - compute weighted lesion evidence and only force `manual_review_required_unrecognized_disease` when lesion evidence stays strong after healthy-confidence penalty.
- Remove/relax this gate only after retrained model passes hard-negative validation.
