# Double-Hard Debias Study

This repository contains the code that was used in support of the paper "Evaluating the Effectiveness of the Double-Hard Debias Technique Against Racial Bias in Word Embeddings".
Much of this code is an adaptation of published code from the NAACL 2019 paper "Black is to Criminal as Caucasian is to Police: Detecting and Removing Multiclass Bias in Word Embeddings."

The data used in this study includes data used in the T. Manzini et al research:
1. Race Attributes File (data/race_attributes_optm.json).
2. w2v_0 Pretrained Word2Vec Embeddings as created in T. Manzini et al study (Source https://drive.google.com/file/d/1IJdGfnKNaBLHP9hk0Ns7kReQwo_jR1xx/view?usp=sharing)
3. professions list (data/provisions.json)

The double-hard-debias.ipynb notebook walks through the entire study and includes execution of code which creates and evaluates double-hard debiased embeddings against
the original pretrained and hard debiased embeddings.