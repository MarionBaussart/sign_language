# Word Translator for American Sign Language 

## Dataset
[WLASL (World Level American Sign Language) Video](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed/data?select=WLASL_v0.3.json): 
2,000 common different words in ASL

File: [preprocess.py](/home/marion/HOLBERTON/sign_language/WLASL/dataset/preprocess.py) 
- Create a Dataframe from WLASL

- Keep only word that have more than 12 videos
    - 17 words_keeped: ['drink', 'computer', 'before', 'go', 'who', 'candy', 'cousin', 'help', 'thin', 'cool', 'thanksgiving', 'bed', 'bowling', 'tall', 'accident', 'short', 'trade']

- Split into training, validation and tests sets (75%, 10%, 15%)
    - train: 186 videos, test: 52 videos

## Build input pipeline
Create frames from videos files

## Packages installed
keras 2.13.1
keras-rl2 1.0.4
matplotlib 3.8.3
numpy 1.23.5
opencv-python 4.9.0.80
pandas 2.2.1
tensorflow 2.13.0
