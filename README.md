# Spoken Grammar Scoring Engine (SHL Hiring Assessment 2026)

Predicts a 0-5 grammar score from 45-60 s speech clips.

## Approach
Audio -> Whisper ASR (word timestamps) -> three feature groups -> stacked regressors
- Fluency, LanguageTool errors, spaCy syntax, GPT-2 perplexity features
- WavLM audio embeddings and MPNet transcript embeddings
- LightGBM, Ridge and SVR models, stratified 5-fold CV, non-negative stacking

## Results
| Metric | Value |
|---|---|
| Training RMSE | 0.2666 |
| CV RMSE | 0.5425 |
| CV Pearson | 0.8995 |
| Leaderboard | 1.1912 |

## Run it
Open `grammar_scoring_engine.ipynb` on Kaggle with GPU and Internet enabled and the
SHL Hiring Assessment 2026 data attached, then run all cells.

## Notes
Audio embeddings were the strongest signal; text-only features were weakest because
ASR partly corrects grammar errors. See the notebook for the full report and plots.
