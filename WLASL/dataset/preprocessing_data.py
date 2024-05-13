#!/usr/bin/env python3
"""
Preprocess the WLASL (World Level American Sign Language) dataset
"""
import cv2
import imageio
from imutils import paths
from IPython.display import Image
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# Define hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
MAX_SEQ_LENGTH = 30
NUM_FEATURES = 512

# Read the data
file_path = './WLASL_v0.3.json'
df_wlasl = pd.read_json(file_path)

# Get the word, video id and video url corresponding
def get_video_infos(df_wlasl):
    """
    Get the video gloss(glossary), id and url if video exists
    Argument: 
        df_wlasl: json_dataframe
    Return: 
        list of videos_glossary, videos_ids and videos_urls
    """
    videos_ids = []
    videos_urls = []
    videos_glossary = []

    for i, instance in enumerate(df_wlasl['instances']):
        video_glossary = df_wlasl['gloss'][i]

        for item in instance:
            video_id = item['video_id']
            video_url = item['url']
            if os.path.exists('../videos/' + video_id + '.mp4'):
                videos_ids.append(video_id)
                videos_urls.append(video_url)
                videos_glossary.append(video_glossary)

    return videos_glossary, videos_ids, videos_urls


videos_glossary, videos_ids, videos_urls = get_video_infos(df_wlasl)

# Create dataframe
wlasl_dict = {'video_id': videos_ids, 'label': videos_glossary,}
df_wlasl_preprocessed = pd.DataFrame(wlasl_dict)

# Keep only word that have more than 12 videos
df_wlasl_preprocessed = df_wlasl_preprocessed.groupby('label').filter(lambda x : len(x) > 14)
df_wlasl_preprocessed['nb_videos'] = df_wlasl_preprocessed.groupby(['label'])['video_id'].transform('count')
df_wlasl_preprocessed.reset_index(inplace=True)
df_wlasl_preprocessed = df_wlasl_preprocessed.drop(['index'], axis=1)

# Split date into training and test set
def split_data(df):
    """
    Split data into training and test set
    Arg: dataframe
    Return: 
        list of training videos,
        list of testing videos,
        list of words keeped
    """
    words_list = []
    train_video_list = []
    test_video_list = []
    counter = 0
    last_word = df['label'][0]
    if words_list == []:
        words_list.append(df['label'][0])
    for index, word in enumerate(df['label']):
        if word == last_word:
            if counter >= 10:
                test_video_list.append(df['video_id'][index])
            else:
                train_video_list.append(df['video_id'][index])
            counter += 1
            last_word = word
        else:
            words_list.append(word)
            train_video_list.append(df['video_id'][index])
            counter = 0
            last_word = word
    return train_video_list, test_video_list, words_list


train_video_list, test_video_list, words_list = split_data(df_wlasl_preprocessed)

# Create train and test dataframe
df_wlasl_preprocessed['set'] = ['train' if video_id in train_video_list else 'test' for video_id in df_wlasl_preprocessed['video_id']]

df_wlasl_preprocessed = df_wlasl_preprocessed.drop(columns=['nb_videos'])

df_wlasl_preprocessed['video_id'] = df_wlasl_preprocessed['video_id'] + '.mp4'
df_wlasl_preprocessed = df_wlasl_preprocessed.rename(columns={'video_id': 'video_name'})

df_train = df_wlasl_preprocessed[df_wlasl_preprocessed['set'] == 'train']
df_test = df_wlasl_preprocessed[df_wlasl_preprocessed['set'] == 'test']

print('\ndf train:\n', df_train,
      '\ndf test:\n', df_test)
print('Total videos for training: {}'.format(len(df_train)))
print('Total videos for testing: {}'.format(len(df_test)))

def crop_center_square(frame):
    """
    Define center of the frame for padding if too small
    """
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    """
    Load video and split into frame
    Return: numpy array containing the frames
    """
    video = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = crop_center_square(frame)
        frame = cv2.resize(frame, resize)
        frame = frame[:, :, [2, 1, 0]]
        frames.append(frame)

        if len(frames) == max_frames:
            break
    video.release()
    return np.array(frames)


def build_feature_extractor():
    """
    Extract meaningful features from the extracted frames
    """
    feature_extractor = keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        pooling="max",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.vgg16.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()

# Convert class label into integers
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(df_train["label"])
)
print(label_processor.get_vocabulary())


def prepare_all_videos(df, root_dir):
    """
    Split all video in path root_dir into frames and extract features
    """
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["label"].values
    labels = keras.ops.convert_to_numpy(label_processor(labels[..., None]))

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(root_dir + path)
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(
            shape=(1, MAX_SEQ_LENGTH,),
            dtype="bool"
        )
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :], verbose=0,
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


train_data, train_labels = prepare_all_videos(df_train, "./train/")
test_data, test_labels = prepare_all_videos(df_test, "./test/")

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")


# Utility for our sequence model.
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model

# Utility for running experiments.
def run_experiment():
    filepath = "./video_classifier/ckpt.weights.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


_, sequence_model = run_experiment()