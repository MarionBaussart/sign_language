#!/usr/bin/env python3
"""
Preprocess the WLASL (World Level American Sign Language) dataset
"""
import cv2
import keras as K
from moviepy.editor import ImageSequenceClip
import numpy as np
import os
import pandas as pd
import tensorflow as tf


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

# Keep only word that have more than 14 videos
df_wlasl_preprocessed = df_wlasl_preprocessed.groupby('label').filter(lambda x : len(x) > 12)
df_wlasl_preprocessed['nb_videos'] = df_wlasl_preprocessed.groupby(['label'])['video_id'].transform('count')
df_wlasl_preprocessed.reset_index(inplace=True)
df_wlasl_preprocessed = df_wlasl_preprocessed.drop(['index'], axis=1)

# Split date into training and test set
def split_data(df):
    """
    Split data into training and test set
    Arg: dataframe
    Return: 
        list of training videos ids,
        list of testing videos ids,
        list of words keeped
    """
    words_list = []
    train_videos_ids_list = []
    test_videos_ids_list = []
    train_labels_list = []
    test_labels_list = []
    counter = 0
    last_word = df['label'][0]
    if words_list == []:
        words_list.append(df['label'][0])
    for index, word in enumerate(df['label']):
        if word == last_word:
            if counter >= 10:
                test_videos_ids_list.append(df['video_id'][index])
                test_labels_list.append(df['label'][index])
            else:
                train_videos_ids_list.append(df['video_id'][index])
                train_labels_list.append(df['label'][index])
            counter += 1
            last_word = word
        else:
            words_list.append(word)
            train_videos_ids_list.append(df['video_id'][index])
            train_labels_list.append(df['label'][index])
            counter = 0
            last_word = word
    return train_videos_ids_list, test_videos_ids_list, words_list, train_labels_list, test_labels_list


train_videos_ids_list, test_videos_ids_list, words_list, train_labels_list, test_labels_list = split_data(df_wlasl_preprocessed)

# Create train and test dataframe
df_wlasl_preprocessed['set'] = ['train' if video_id in train_videos_ids_list else 'test' for video_id in df_wlasl_preprocessed['video_id']]

df_wlasl_preprocessed = df_wlasl_preprocessed.drop(columns=['nb_videos'])

df_wlasl_preprocessed['video_id'] = df_wlasl_preprocessed['video_id'] + '.mp4'
df_wlasl_preprocessed = df_wlasl_preprocessed.rename(columns={'video_id': 'video_name'})

df_train = df_wlasl_preprocessed[df_wlasl_preprocessed['set'] == 'train']
df_test = df_wlasl_preprocessed[df_wlasl_preprocessed['set'] == 'test']

print('\ndf train:\n', df_train,
      '\ndf test:\n', df_test)
print('Total videos for training: {}'.format(len(df_train)))
print('Total videos for testing: {}'.format(len(df_test)))


# Get number of frames by video
def nb_frames_by_video(train_videos_ids_list, test_videos_ids_list):
    """
    Get the list of number of frame by video
    """
    nb_frames_list = []
    for video_id in train_videos_ids_list:
        video_path = '../videos/' + video_id + '.mp4'
        video = cv2.VideoCapture(video_path)
        # Get the number or frames
        nb_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        nb_frames_list.append(nb_frames)

    for video_id in test_videos_ids_list:
        video_path = '../videos/' + video_id + '.mp4'
        video = cv2.VideoCapture(video_path)
        # Get the number or frames
        nb_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        nb_frames_list.append(nb_frames)
    
    return nb_frames_list
    

nb_frames_list = nb_frames_by_video(train_videos_ids_list, test_videos_ids_list)
print('nb_frames_list:', nb_frames_list,
      '\nnb mean frames:', sum(nb_frames_list) / len(nb_frames_list),
      '\nmin_frame_count_list:', min(nb_frames_list),
      '\nmax_frame_count_list:', max(nb_frames_list))
index_max = nb_frames_list.index(195)
print('\nindex:', index_max)

# Split video into frames, resize and convert into numpy array
def np_array_from_video(video_path, output_size=(224,224), nb_max_frames=20):
    """
    Generate frame from video, resize and convert them into a numpy array
    Return:
        An NumPy array of frames in the shape of (n_frames, height, width)
    """
    list_frames = []
    # Load the video
    video = cv2.VideoCapture(video_path)
    # Get number of frames
    nb_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # Calculate the frequency to have nb_max_frames by video
    frequency = nb_frames // nb_max_frames
    # Extract the frame every 'frequency' frames
    for i_frame in range(nb_frames):
        ret, frame = video.read()
        if i_frame % frequency == 0 and (i_frame < frequency * nb_max_frames):
            # Check if the frame was successfully read
            if ret:
                # Resize
                casted_frame = tf.image.convert_image_dtype(frame, tf.float32)
                resized_frame = tf.image.resize_with_pad(
                    image=casted_frame,                   target_height=output_size[0],
                    target_width=output_size[1])
                # Grayscale
                np_array_frame = np.array(resized_frame)
                gray_frame = cv2.cvtColor(np_array_frame, cv2.COLOR_BGR2GRAY)
                
                list_frames.append(gray_frame)
            else:
                list_frames.append(np.zeros(shape=output_size))

    # Release the video
    video.release()
    np_array_frames = np.array(list_frames)
    
    return np_array_frames

# # test numpy array
# video_path = '../videos/' + train_videos_ids_list[114] + '.mp4'
# np_array_114_frames = np_array_from_video(video_path)
# print('\narray:\n', np_array_114_frames,'\nshape:\n', np_array_114_frames.shape)

# Visualize with a gif
def numpy_to_gif(filename, np_array, fps=20, scale=1.0):
    """
    Convert a numpy array (gray image) to gif
    """
    np_array *= 255
    np_array = np_array[..., np.newaxis]
    gif = ImageSequenceClip(list(np_array), fps=fps).resize(scale)
    gif.write_gif(filename, fps=fps)

    return gif


# numpy_to_gif('test_min_frames.gif', np_array_frames)

print('\ntrain_videos_ids_list:\n', train_videos_ids_list,
      '\ntest_videos_ids_list:\n', test_videos_ids_list,
      '\nwords_list:\n', words_list,
      '\ntrain_labels_list:\n', train_labels_list,
      '\ntest_labels_list:\n', test_labels_list)

# X_train, X_test
X_train_list = []
X_test_list = []

for video_id in train_videos_ids_list:
    video_path = '../videos/' + video_id + '.mp4'
    np_array_video = np_array_from_video(video_path)
    X_train_list.append(np_array_video[np.newaxis])
for video_id in test_videos_ids_list:
    video_path = '../videos/' + video_id + '.mp4'
    np_array_video = np_array_from_video(video_path)
    X_test_list.append(np_array_video[np.newaxis])

X_train = np.concatenate(X_train_list, axis=0)
X_test = np.concatenate(X_test_list, axis=0)

# Y_train, Y_train, convert labels into integer
Y_train_list = np.array(train_labels_list)
Y_test_list = np.array(test_labels_list)

labels = {}
list_labels = np.unique(Y_train_list)

for i, label in enumerate(list_labels):
    labels[label] = i

Y_train = np.vectorize(labels.get)(Y_train_list)
Y_test = np.vectorize(labels.get)(Y_test_list)

Y_train_one_hot = K.utils.to_categorical(
    Y_train)
Y_test_one_hot = K.utils.to_categorical(
    Y_test)

# print('\nX_train:\n', X_train,
#       '\nX_test:\n', X_test,
#       '\nY_train:\n', Y_train,
#       '\nY_test:\n', Y_test)
print('\nX_train.shape:\n', X_train.shape,
      '\nX_test.shape:\n', X_test.shape,
      '\nY_train.shape:\n', Y_train.shape,
      '\nY_test.shape:\n', Y_test.shape,
      '\nlabels:\n', labels,
      '\nY_train_one_hot.shape:\n', Y_train_one_hot.shape,
      '\nY_test_one_hot.shape:\n', Y_test_one_hot.shape)

nb_frames, height, width = X_train.shape[1:]
# input_shape = (nb_frames, height, width)


# Build the model
def build_model(nb_frames, height, width):
    """
    Build the model
    """
    inputs = K.Input(shape=(nb_frames, height, width))

    initializer_kernel_weights = K.initializers.HeNormal(seed=None)

    conv_7x7 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        activation='relu',
        kernel_initializer=initializer_kernel_weights
    )(inputs)
    batch_normalization = K.layers.BatchNormalization()(conv_7x7)
    activated_output = K.layers.Activation('relu')(batch_normalization)
    max_pooling = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(activated_output)
    dropout = K.layers.Dropout(
        rate=0.1
    )(max_pooling)

    average_pooling = K.layers.GlobalAveragePooling2D()(dropout)

    flatten = K.layers.Flatten()(average_pooling)
    output = K.layers.Dense(
        units=17,
        activation='softmax',
        kernel_initializer=initializer_kernel_weights
    )(flatten)

    model = K.Model(inputs=inputs, outputs=output)

    return model

model = build_model(nb_frames, height, width)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model
filepath = "./video_classifier/ckpt.cnn.weights.h5"
checkpoint = K.callbacks.ModelCheckpoint(
    filepath, save_weights_only=True, save_best_only=True, verbose=1
)
batch_size = 100
epochs = 50

history = model.fit(
    x=X_train,
    y=Y_train_one_hot,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[checkpoint],
    verbose=True
    #validation_data=(X_test, Y_test_one_hot)
)

model.load_weights(filepath)

model.evaluate(x=X_test,
               y=Y_test_one_hot)

# Plotting
# print(history.history.keys())

# plt.plot(history.history['accuracy'], label='Training accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation accuracy')
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(loc='upper left')
# plt.show()

# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label= 'Validation loss')
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(loc='upper right')
# plt.show()


# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

