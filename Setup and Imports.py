#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install transformers --use-feature=2020-resolver')
get_ipython().system('git clone https://github.com/aub-mind/arabert')
get_ipython().system('pip install pyarabic --use-feature=2020-resolver')
get_ipython().system('pip install farasapy --use-feature=2020-resolver')
get_ipython().system('pip install flask-ngrok')
import pandas as pd
import numpy as np
import json
import requests
import re
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import torch
import pickle
from flask_ngrok import run_with_ngrok
from flask import Flask, request, render_template

url = 'https://recruitment.aimtechnologies.co/ai-tasks'

