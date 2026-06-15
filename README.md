# Apple (AAPL) News-Sentiment & Price Movement Study

An end-to-end pipeline that scrapes financial news, scores its sentiment, combines it with market data and technical indicators, and tests that signal against naive baselines to see whether it predicts Apple's next-day stock movement.

**Headline finding:** it can't. The goal of this project is to show that with an honest evaluation rather than a manufactured result. Across two framings (price level and price direction), simple naive baselines beat the LSTM model. This is consistent with weak-form market efficiency, and the project's worth lies in the pipeline itself rather than in any predictive edge.

---

## 

## What this project does

1. **Collects news** — bulk-downloads event records from the [GDELT database](http://data.gdeltproject.org/events/index.html) over 2021–2026.  
     
2. **Filters for Apple** — isolates URLs mentioning Apple as a whole word.  
     
3. **Scores sentiment** — runs FinBERT ([yiyanghkust/finbert-tone](https://huggingface.co/yiyanghkust/finbert-tone)) over the text to produce positive / neutral / negative probabilities and a combined sentiment score.  
     
4. **Pulls prices** — fetches AAPL adjusted close prices via [yfinance](https://pypi.org/project/yfinance/).  
     
5. **Engineers features** — daily average sentiment, rolling sentiment mean/volatility, returns, momentum, RSI, MACD, EMA, and 5-day lags of sentiment and price.  
     
6. **Trains & evaluates** — an LSTM, compared head-to-head against naive baselines.

## Data

- **Range:** 2021 – March 2026  
    
- **Size:** 1,291 rows → **1,187 usable** trading days after dropping NaNs from rolling/lag features  
-   
- **Split:** 1,008 train / 179 test, chronological (no shuffling, preserves time order so the model never sees the future)

Tools

Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, transformers ([FinBERT](https://huggingface.co/yiyanghkust/finbert-tone)), [yfinance](https://pypi.org/project/yfinance/), pandas\_ta, [BeautifulSoup](https://pypi.org/project/beautifulsoup4/).

## Results

### Experiment 1 — predicting next-day price *level* (regression)

| Metric | Naive baseline (tomorrow \= today) | LSTM |
| :---- | :---- | :---- |
| MAE | **2.46** | 5.58 |
| RMSE | **3.53** | 6.49 |
| R² | **0.979** | 0.928 |

The model's R² of 0.93 looks strong in isolation, but a baseline that simply guesses "tomorrow's price \= today's price" scores 0.98. The high R² reflects **price persistence** (prices barely move day-to-day), not predictive skill. The baseline wins on every metric. (Scalers are fit on training data only, so no test information leaks into preprocessing.)

### Experiment 2 — predicting next-day *direction* (up/down classification)

|  | Accuracy |
| :---- | :---- |
| Naive baseline (always predict "up") | **0.531** |
| LSTM | 0.469 |

Direction prediction removes the persistence advantage. Here the LSTM scores 46.9% — below both the majority-class baseline (53.1%) and a coin flip (50%). On 179 test days this sits within the range of random noise: the honest interpretation is that **daily direction is essentially unpredictable from these features**, not that the model is "anti-predictive."

### Takeaway

Both experiments point the same way: simple baselines beat the complex model. This is a textbook illustration of weak-form market efficiency and a reminder that an impressive-looking metric (R² \= 0.93) can be meaningless without a baseline to compare against.

---

## 

## Data audit

Before trusting the data, I spot-checked the dates and sentiment scores rather than assuming it was clean. I found that GDELT coded the event date, not the publication date. I cross-checked against dates embedded in article URLs. Of the \~20% of articles (1,604 / 8,041) whose URL contained a full date, **71.9% matched the assigned date exactly and 92.9% fell within one day**. This is  acceptable for a next-day hypothesis. However, the remaining \~80% had no URL date and could not be verified this way. 

---

## 

## Known limitations

These weaken confidence in the findings and would be the first things to fix in future work

- **Headlines are reconstructed from URLs**, not actual article text, so many are partial, which means FinBERT is working with noisy input.  
- **Apple disambiguation is keyword-based** ("apple" in the URL), which can catch unrelated items (the fruit, people, other entities).  
- **Date alignment** relies on GDELT's event date for the unverifiable majority of rows.  
- **timesteps \= 1** means the LSTM isn't exploiting sequence memory; the lag features carry the temporal information instead.

## What I'd do next

1. Extract real headline text (or article first paragraph) instead of URL slugs.  
     
2. Use GDELT's ingestion timestamp (or the URL date) for publication-accurate alignment.  
     
3. Predict returns rather than price levels, and benchmark every model against a naive baseline by default.  
     
4. Test simpler models (logistic regression, gradient boosting) before deep learning on a dataset this size.  
     
5. Feed the LSTM a true multi-day sequence (e.g. 10–20 timesteps) and remove the manual lag features, so the model learns temporal patterns directly rather than receiving history as static columns.  
   

---

## 

## Why this project exists

The goal was never to "beat the market." It was to build a complete, working data pipeline and to practise evaluating a hypothesis honestly, including reporting a negative result clearly instead of dressing it up. The same discipline applies to assessing any data-backed claim.