# Double-Hard Debias Study

This repository contains the code that was used in support of the paper "Evaluating the Effectiveness of the Double-Hard Debias Technique Against Racial Bias in Word Embeddings".

It includes three notebooks for reproducing the study:

1. Double-Hard Debias generation and embedding bias analysis - [double-hard-debias.ipynb](double-hard-debias.ipynb). This notebook executes both hard an double-hard debias methods and runs a variety of analysis for review and compare.  This notebook also generates required files needed for RSA (see 2)
2. Representation Similarity analysis - [rsa.ipynb](rsa.ipynb). **This notebook requires the w2v generated files created in [double-hard-debias.ipynb](double-hard-debias.ipynb)**.
3. Utility Evaluation - [semantic_eval.ipynb](semantic_eval.ipynb). This notebook executes downstream evaluation methods Concept Categorization and Analogy Analysis.

### Requirements

1. This project was created and tested with python 3.9.  The following libraries are required (and referenced in [requirements.txt](requirements.txt)):
```
gensim==4.1.2
matplotlib==3.4.3
numpy==1.21.3
pandas==1.3.4
scikit_learn==1.0.1
scipy==1.7.1
six==1.16.0
statsmodels==0.13.1
openpyxl==3.0.9
seaborn==0.11.2
```
Jupyter server is also required for execution of notebooks

2. This project also requires that two files be downloaded, saved per instructions and placed within the data folder before executing any notebook:
   1. Pre-trained Word2Vec pt. 0 (w2v_0) embeddings from T. Manzini et al study. File should be saved to **data** folder as **data_vocab_race_pre_trained.w2v**. [download file](https://drive.google.com/file/d/1IJdGfnKNaBLHP9hk0Ns7kReQwo_jR1xx/view)
   2. Pre-processed Hard Debiased embeddings from T. Manzini et al study.  File should be saved to **data** folder as **data_vocab_race_hard_debias.w2v**. [download file](https://drive.google.com/file/d/1at-OZonjKtb-Z1MvvLX3embAbZyfAmwX/view)

3. Proceed with running research notebooks. **NOTE: The RSA study requires that [double-hard-debias.ipynb](double-hard-debias.ipynb) be executed prior to running the notebook for the first time to generate required w2v files.**

### Reference Research & Code Repositories
This code is an adaptation of published code from the following research papers:

[Black is to Criminal as Caucasian is to Police: Detecting and Removing Multiclass Bias in Word Embeddings](https://arxiv.org/abs/1904.04047)
[(project code)](https://github.com/TManzini/DebiasMulticlassWordEmbedding)

[Double-Hard Debias: Tailoring Word Embeddings for Gender Bias Mitigation](https://arxiv.org/abs/2005.00965)
[(project code)](https://github.com/uvavision/Double-Hard-Debias)

[Unequal Representations: Analyzing Intersectional Biases in Word Embeddings Using Representational Similarity Analysis](https://aclanthology.org/2020.coling-main.151.pdf)
[(project code)](https://github.com/mlepori1/Unequal_Representations)

We appreciate the efforts of each of these projects towards helps us move forward this research.
