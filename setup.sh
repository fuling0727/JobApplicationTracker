#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt
pip uninstall -y numpy spacy torch transformers
pip install "numpy<2" spacy torch transformers
python -m spacy download en_core_web_sm
