'''
@author Michael Lepori
@author Tom McCoy
@date 11/5/19

Utilities for natural language agreement task
'''

import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.descriptivestats import sign_test
import random
import pandas as pd

# Orginally written by Tom McCoy
def create_embedding_dictionary(emb_file, f_dim, filler_to_index, index_to_filler,  unseen_words="zero"):
	embedding_dict = {}
	embed_file = open(emb_file, "r")
	for line in embed_file:
		parts = line.strip().split()
		if len(parts) == f_dim + 1:
			embedding_dict[parts[0]] = list(map(lambda x: float(x), parts[1:]))

	matrix_len = len(filler_to_index.keys())

	weights_matrix = np.zeros((matrix_len, f_dim))

	for i in range(matrix_len):
		word = index_to_filler[i]
		if word in embedding_dict:
			weights_matrix[i] = embedding_dict[word]

		elif word == "***mask***":
			weights_matrix[i] = np.ones((f_dim,))

		else:
			print(word)
			if unseen_words == "random":
				weights_matrix[i] = np.random.normal(scale=0.6, size=(f_dim,))
			elif unseen_words == "zero":
				pass # It was initialized as zero, so don't need to do anything
			else:
				print("Invalid choice for embeddings of unseen words")


	return weights_matrix


def create_word_idx_matrices(sentence_list):

	idx = 0

	word2idx = {}
	idx2word = {}

	for line in sentence_list:
		for word in line:
			if word not in word2idx.keys():
				word2idx[word] = idx
				idx2word[idx] = word
				idx += 1

	return word2idx, idx2word

"""

Script adapted from the below research for word2vec
Unequal Representation: Analyzing Intersectional Biases in Word Embeddings 
Using Representational Similarity Analysis by Michael Lepori, presented at COLING 2020
"""


def preprocess_data(corpus):
    # Preprocess datasets
    sent_list = []
    for sent in corpus:
        sent = sent.lower()
        sent = sent.replace('.', '')
        sent_list.append(sent)
    return sent_list


def get_w2v_embeds(w2v_path, dim, corpus):
    # Get w2v embeddings of all terms in the corpus
    word2idx, idx2word = create_word_idx_matrices([corpus])
    print("word idx matrices created")
    w2v = create_embedding_dictionary(w2v_path, dim, word2idx, idx2word)
    print("w2v matrices created")

    w2v_embeds = []

    for word in corpus:
        w2v_embeds.append(w2v[word2idx[word]])
        
    return np.array(w2v_embeds)

"""
make_concept and code that prints rsa scores as written for Unequal Representation: Analyzing Intersectional Biases in Word Embeddings 
Using Representational Similarity Analysis by Michael Lepori, presented at COLING 2020 with slight change to print scores for three groups instead of two. 
"""

def make_concept3(samps, grp1, grp2, grp3, attr):
    # Make the hypothesis models encoding the hypothesized representational geometry.
    grp1_attr = np.zeros((len(samps), len(samps)))
    grp2_attr = np.zeros((len(samps), len(samps)))
    grp3_attr = np.zeros((len(samps), len(samps)))
    
    for i in range(len(samps)):
        sent1 = samps[i]
        for j in range(len(samps)):
            sent2 = samps[j]
            if ((sent1 in attr or sent1 in grp1) and (sent2 in attr or sent2 in grp1)) or ((sent1 in grp2) and (sent2 in grp2)):
                # Represents the hypothesis that group 1 has the attribute/concept under study, and group 2 does not
                grp1_attr[i][j] = 0
                # Dissimilarity matrices, so 0 means perfect alignment
            else:
                grp1_attr[i][j] = 1

            if ((sent1 in attr or sent1 in grp2) and (sent2 in attr or sent2 in grp2)) or ((sent1 in grp1) and (sent2 in grp1)):
                # Represents the hypothesis that group 2 has the attribute/concept under study, and group 1 does not
                grp2_attr[i][j] = 0
                # Dissimilarity matrices, so 0 means perfect alignment
            else:
                grp2_attr[i][j] = 1
                
            if ((sent1 in attr or sent1 in grp3) and (sent2 in attr or sent2 in grp3)) or ((sent1 in grp3) and (sent2 in grp3)):
                # Represents the hypothesis that group 2 has the attribute/concept under study, and group 1 does not
                grp3_attr[i][j] = 0
                # Dissimilarity matrices, so 0 means perfect alignment
            else:
                grp3_attr[i][j] = 1
    
    
    return grp1_attr, grp2_attr, grp3_attr

def get_RSA3(sheet_path, embedding_path, dims):
    results = {}
    # Set random seed for reproducibility
    np.random.seed(seed=9)
    random.seed(9)

    # Read data
    sheets = pd.ExcelFile(sheet_path)

    # Every datasheet defines 1 test
    for idx, name in enumerate(sheets.sheet_names):
        print(name)
        sheet = sheets.parse(name)
        group1 = list(sheet.iloc[:, 0].dropna())
        grp1_name = sheet.columns[0]
        group2 = list(sheet.iloc[:, 1].dropna())
        grp2_name = sheet.columns[1]
        group3 = list(sheet.iloc[:, 2].dropna())
        grp3_name = sheet.columns[2]
        concept = list(sheet.iloc[:, 3].dropna())
        attr_name = sheet.columns[3]

        # corpus is all words in dataset
        corpus = group1 + group2 + group3 + concept
        w2v_embeds= get_w2v_embeds(embedding_path, dims, preprocess_data(corpus))
        print("embeds generated")

        rsa_grp1 = []
        rsa_grp2 = []
        rsa_grp3 = []

        # Sample 100 different configurations
        samps = []
        while len(samps) < 100:

            # sample 4 elements of group 1, 4 elements of group 2, 4 elements of group 3, 4 attribute elements
            sample = list(np.random.choice(range(0, len(group1)), replace = False, size=4)) + list(np.random.choice(range(len(group1), len(group1) + len(group2)), replace = False, size=4))+ list(np.random.choice(range(len(group1) + len(group2), len(group1) + len(group2) + len(group3)), replace = False, size=4))+ list(np.random.choice(range(len(group1) + len(group2) + len(group3), len(group1) +  len(group2) + len(group3) + len(concept)), size=4))

            if list(sample) in samps:
                continue

            samps.append(sample)

            # Make hypothesis models, as well as reference models
            samp_sentences = np.array(corpus)[sample]
            samp_w2v = w2v_embeds[sample]

            grp1_attr_model, grp2_attr_model, grp3_attr_model  = make_concept3(samp_sentences, group1, group2, group3, concept)

            # 1 - spearman's r similarity matrix to make dissimilarity matrix
            w2v_sim = np.ones(samp_w2v.shape[0]) - spearmanr(samp_w2v, axis=1)[0]

            # Take upper triangle
            w2v_sim = w2v_sim[np.triu_indices(samp_w2v.shape[0], 1)].reshape(-1)
            grp1_attr_model = grp1_attr_model[np.triu_indices(samp_w2v.shape[0], 1)].reshape(-1)
            grp2_attr_model = grp2_attr_model[np.triu_indices(samp_w2v.shape[0], 1)].reshape(-1)
            grp3_attr_model = grp3_attr_model[np.triu_indices(samp_w2v.shape[0], 1)].reshape(-1)

            # Append representational similarity for group 1 and group 2
            rsa_grp1.append(spearmanr([w2v_sim, grp1_attr_model], axis=1)[0])
            rsa_grp2.append(spearmanr([w2v_sim, grp2_attr_model], axis=1)[0])
            rsa_grp3.append(spearmanr([w2v_sim, grp3_attr_model], axis=1)[0])

        print(f'RSA {grp1_name} {attr_name}: {np.mean(rsa_grp1)} STD: {np.std(rsa_grp1)}')
        print(f'RSA {grp2_name} {attr_name}: {np.mean(rsa_grp2)} STD: {np.std(rsa_grp2)}')
        print(f'RSA {grp3_name} {attr_name}: {np.mean(rsa_grp3)} STD: {np.std(rsa_grp3)}')

        # Significance test of differences between group 1 RSA and group 2 RSA
        print(f'Sign Test {grp1_name} vs. {grp2_name}: {sign_test(np.array(rsa_grp1) - np.array(rsa_grp2))[1]}')
        print(f'Sign Test {grp1_name} vs. {grp3_name}: {sign_test(np.array(rsa_grp1) - np.array(rsa_grp3))[1]}')
        print(f'Sign Test {grp3_name} vs. {grp2_name}: {sign_test(np.array(rsa_grp3) - np.array(rsa_grp2))[1]}\n')
        
        results[grp1_name+"-"+attr_name] =  (np.mean(rsa_grp1), grp1_name, attr_name)
        results[grp2_name+"-"+attr_name] =  (np.mean(rsa_grp2), grp2_name, attr_name)
        results[grp3_name+"-"+attr_name] =  (np.mean(rsa_grp3), grp3_name, attr_name)
    return (results)

        

