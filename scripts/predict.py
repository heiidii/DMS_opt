#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 07 2021

@author: sai pooja mahajan
"""

# Import libraries
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Import machine learning models
from utils import *

# ----------------------
# Run classifiers on in
# silico generated data
# ----------------------

# Create model directory
model_dir = 'classification'
os.makedirs(model_dir, exist_ok=True)

# Use tuned model parameters for CNN (performed in separate script)
params = [['CONV', 400, 5, 1], ['DROP', 0.2], ['POOL', 2, 1], ['FLAT'],
          ['DENSE', 300]]

# input shape -> L=10 (H3 loop), 20 -> one hot enc
CNN_classifier = create_cnn(params, (10, 20), 'relu', None)

#test
seqs = ['WGGDGFYAMD']
pbinder_wt = run_classification(seqs, CNN_classifier, False)
print("Wildtype score: ", pbinder_wt)

infile = sys.argv[1]
file_lines = open(infile, 'r').readlines()
sequences = [
    t.split()[-1].rstrip() for t in file_lines
    if (t.find('>') == -1 and t != '\n')
]
ddg = [
    t.split()[-2].rstrip() for t in file_lines
    if (t.find('>') == -1 and t != '\n')
]
print(sequences)
if len(sequences[0]) < 10:
    sequences = [t[0:6] + 'Y' + t[6:] for t in sequences]

prob_binding = run_classification(sequences, CNN_classifier, False)
prob_binding = prob_binding.flatten()
outpath = os.path.dirname(infile)
outfile = os.path.join(outpath, 'dms_herceptin_pbinding.txt')
with open(outfile, 'w') as f:
    file_lines = [
        t.split('\n')[0] for t in file_lines
        if (t.find('>') == -1 and t != '\n')
    ]
    for line, prob in zip(file_lines, list(prob_binding)):
        f.write(line + '\t' + str(prob) + '\n')

outplt = os.path.join(outpath, 'dms_herceptin_pbinding_vs_ddg.png')
plt.scatter(ddg, prob_binding)
plt.xlabel('ddG (REU)')
plt.ylabel('p(binder) Mason et al.')
plt.savefig(outplt, transparent=True, dpi=600)
plt.close()

outplt = os.path.join(outpath, 'dms_herceptin_pbinding_dist.png')

sns.distplot(prob_binding, kde=True)
plt.axvline(pbinder_wt, color='black')
plt.xlabel('p(binder) Mason et al.')
plt.ylabel('Normalized Density')
plt.tight_layout()
plt.savefig(outplt, transparent=True, dpi=600)
plt.close()