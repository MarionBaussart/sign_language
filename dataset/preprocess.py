#!/usr/bin/env python3
"""
Preprocess the American Language Sign dataset
"""
import pandas as pd

# Read the data
file_path = './sign_mnist_train.csv'
df_asl_train = pd.read_csv(file_path)
file_path = './sign_mnist_test.csv'
df_asl_test = pd.read_csv(file_path)

print("df_asl_train:\n", df_asl_train.head())
print("df_asl_test:\n", df_asl_test.head())

# Count NaN values
nb_nan_asl_train = df_asl_train.isna().sum()
nb_nan_asl_test = df_asl_test.isna().sum()

print("number of NaN asl_train:\n", nb_nan_asl_train, "\n")
print("number of NaN asl_test:\n", nb_nan_asl_test, "\n")

# check types
print("\ntypes:")
print(df_asl_test.dtypes)
print(df_asl_train.dtypes)

# Describe data
print("train data description:\n", df_asl_train.describe())
print("test data description:\n", df_asl_test.describe())

# export the preprocessed data
df_asl_train.to_csv('./asl_train_preprocessed.csv', index=False)
df_asl_test.to_csv('./asl_test_preprocessed.csv', index=False)
