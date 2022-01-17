# Esun-AI-Competition-2021Winter

Hi, we are the team of AndyIsMyBoss. In this repo, we share the code of how we achieved 2nd prize in 2021 Winter Esun AI Competition - Credit Card Consumption Tag Recommendation. To learn more information about this competition, please refer to [Competition Website](https://tbrain.trendmicro.com.tw/Competitions/Details/18). 
More details about the method we used in this code, please refer to [Article](https://reurl.cc/nEEkQ2), [Video](https://youtu.be/WiH9avOx9K4), and [Presentation Slide](https://reurl.cc/bkkZMo).

---

## Required Packages

- python 3.8.5
- numpy
- pandas
- csv
- matplotlib
- tqdm
- pickle
- scipy
- torch
- sklearn

## Source Code

- RNN Model.ipynb
  - The method we used to achieve 2nd prize in this competition. In this file, it includes the process of feature engineering, model construction, and training method.
- Ensemble.ipynb
  - The method of soft-ensemble to achieve higher score on leaderboard.
- Ranking Loss.py
  - The PyTorch version of many different ranking loss. For Tensorflow version, please refer to [TF Ranking](https://www.tensorflow.org/ranking).
- Transformer.py
  - The transformer model we implemented.
- DASALC.py
  - We implemented the method published in 2021 ICLR by Google. For more details, please refer to [Are neural rankers still outperformed by gradient boosted decision trees?](https://research.google/pubs/pub50030/)

## Contact Information

   ```
Chieh-Fang Teng:
        + jeff@access.ee.ntu.edu.tw
        + d06943020@ntu.edu.tw
   ```
