# ============================================================================
# 🚀 AMD ROCm (GPU) - Setup de Aceleração
# ============================================================================

import sys
sys.path.append('../../')  # Aponta para a pasta Tensorflow onde está o tf_startup.py
import tf_startup

import tensorflow as tf

# The following line disables JIT compilation. This is a workaround for a
# potential issue on some GPU setups (like ROCm) where certain operations
# can cause JIT compilation errors.
tf.config.optimizer.set_jit(False)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gráficos
%matplotlib inline
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = [14, 6]

# Import series of helper functions for the notebook
from helper_functions import unzip_data, create_tensorboard_callback, plot_loss_curves, compare_historys
import zipfile
import urllib.request

# Baixar o arquivo ZIP
url = "https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip"
zip_path = "nlp_getting_started.zip"
urllib.request.urlretrieve(url, zip_path)

# # Descompactar o arquivo ZIP
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall()
import pandas as pd
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(train_df.head())
# Shuffle training dataframe
train_df_shuffled = train_df.sample(frac=1, random_state=42) # random_state is set to 42 for reproducibility
train_df_shuffled.head()
# What does the test dataframe look like?
test_df.head()
# How many examples of each class are in the training set?
train_df.target.value_counts()
# How many total samples?
len(train_df), len(test_df)
# Let's visualize some random training examples
import random
random_index = random.randint(0, len(train_df)-5)
for row in train_df_shuffled[["text", "target"]][random_index:random_index+5].itertuples():
    _, text, target = row
    print(f"Target: {target}", "(real threat)" if target > 0 else "(not a threat)")
    print(f"Text:\n{text}\n")
    print("---\n")
# split the data into training and testing sets
from sklearn.model_selection import train_test_split
# Use train_test_split to split the data into training and testing sets.
train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
                                                                            train_df_shuffled["target"].to_numpy(),
                                                                            test_size=0.1,  # 10% of the data
                                                                            random_state=42) # random seed for reproducibility
# Check the length of the text.
len(train_sentences), len(train_labels), len(val_sentences), len(val_labels)
# Check  the first 10 samples
train_sentences[:10], train_labels[:10]
# convert text to numbers.
from tensorflow.keras.layers import TextVectorization

text_vectorizer = TextVectorization(max_tokens=10000, # how many words in the vocabulary (automatically add <OOV>)
                                    standardize="lower_and_strip_punctuation", # convert text to lowercase and remove punctuation
                                    split="whitespace", # split text into words or characters
                                    ngrams=None, # create groups of words or characters
                                    output_mode="int", # how to convert tokens to numbers
                                    output_sequence_length=None, # how long is the output sequence
                                    pad_to_max_tokens=True) # pad the sequence to the max length 
len(train_sentences[0].split()) # how many words in the first tweet
# Finde the average numbers of tokens (words) in the training tweets

round(sum([len(tweet.split()) for tweet in train_sentences])/len(train_sentences))
# Setup text Vectorization variables
max_vocab_length = 10000 # how many unique words in the vocabulary
max_length = 15 # max length of a text to consider

text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)
# Fit the text vectorizer on the training tweets
text_vectorizer.adapt(train_sentences)

# Create a sample sentence and tokenize it
sample_sentence = " There's a flood in my street!"
text_vectorizer([sample_sentence])
# Choose a random sentence from the training dataset and tokenize it
random_sentence = random.choice(train_sentences)   
print(f"Original tweet: \n {random_sentence}\
      \n\nVectorized Version:")
text_vectorizer([random_sentence])
# get the unique words in the vocabulary
words_in_vocab = text_vectorizer.get_vocabulary() # get all the unique words in the vocabulary
top_5_words = words_in_vocab[:5] # get the top 5 words in the vocabulary
bottom_5_words = words_in_vocab[-5:] # get the bottom 5 words in the vocabulary
print(f"Numbers of words in vocab: {len(words_in_vocab)}")
print(f"5 most commom words: {top_5_words}")
print(f"5 least commom words: {bottom_5_words}")
from tensorflow.keras import layers

embedding = layers.Embedding(input_dim=max_vocab_length,
                            output_dim=128,
                            input_length=max_length,
                            )
# get a random sentence from the training set
random_sentence = random.choice(train_sentences)
print(f"Original tweet: \n {random_sentence}\
            \n\nEmbedded Version:")

# Embed the radon sentence (turn it into a dense vector of fixed sizes)
sample_embed = embedding(text_vectorizer([random_sentence]))
sample_embed 
# Check out a single token's embedding
sample_embed[0][0], sample_embed[0][0].shape, random_sentence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
# Create the pipeline

model_0 = Pipeline([
    ('tfidf', TfidfVectorizer()), # convert text to numbers
    ('clf', MultinomialNB()) # model the text
])

# Fit the pipeline to the training data
model_0.fit(train_sentences, train_labels)
# Evaluate our baseline model
baseline_score = model_0.score(train_sentences, train_labels)
print(f"Our baseline model achieves an accuracy of: {baseline_score:.3f}%")
# Make predictions
baseline_preds = model_0.predict(val_sentences)
baseline_preds[:20]
# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
  -----
  y_true = true labels in the form of a 1D array
  y_pred = predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted" average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results
# get baseline results
baseline_results = calculate_results(y_true=val_labels, y_pred=baseline_preds)
baseline_results
# Create a tensorboard callback (need to create a new one for each model)
from helper_functions import create_tensorboard_callback

# create a directory to save TensorBoard logs
SAVE_DIR = "model_logs"
from tensorflow.keras import layers

inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)       # (None, 1) → (None, sequence_length)
x = embedding(x)                  # (None, sequence_length) → (None, sequence_length, embed_dim)
x = layers.GlobalAveragePooling1D()(x)  # (None, sequence_length, embed_dim) → (None, embed_dim)
outputs = layers.Dense(1, activation="sigmoid")(x)  # (None, embed_dim) → (None, 1)

model_1 = tf.keras.Model(inputs, outputs, name="model_1_dense")
model_1.summary()
# Compile the model
model_1.compile(
    loss="binary_crossentropy",  # NOT sparse_categorical or from_logits=True
    optimizer="adam",
    metrics=["accuracy"]
)
model_1.summary()
# Check intermediate shapes
print("After vectorizer:", text_vectorizer(tf.constant(["test"])).shape)
print("Model summary:")
model_1.summary()
# Fit the model
model_1_history = model_1.fit(x=train_sentences, # input sentences can be a list of strings due to text preprocessing layer built-in model
                              y=train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR, 
                                                                     experiment_name="simple_dense_model")])
# Check the results
model_1.evaluate(val_sentences, val_labels)
embedding.weights
# Make some predictions with our new model and evaluate those predictions
model_1_pred_probs = model_1.predict(val_sentences)
model_1_pred_probs.shape
model_1_pred_probs[0]
model_1_pred_probs[:10]
# Convert prediction probabilities to labels
model_1_preds = tf.round(model_1_pred_probs)  # rounds 0.5+ to 1, below 0.5 to 0
model_1_preds.shape  # (762, 1) — still fine

# Flatten if needed for evaluation
model_1_preds = tf.squeeze(model_1_preds)  # (762,) — matches val_labels shape
model_1_preds.shape  # (762,)
from sklearn.metrics import classification_report

print(classification_report(val_labels, model_1_preds))
# Convert model prediction probabilities to label format
model_1_preds = tf.squeeze(tf.round(model_1_pred_probs))
model_1_preds[:20]
# Calculate our model_1 results
from pyexpat import model


model_1_results = calculate_results(y_true=val_labels, y_pred=model_1_preds)
print(model_1_results)
baseline_results
import numpy as np
np.array(list(model_1_results.values()) > np.array(list(baseline_results.values())))

# get the vocabulary from the text vectorization
words_in_vocab = text_vectorizer.get_vocabulary()
len(words_in_vocab), words_in_vocab[:20]
# Model1 summary
model_1.summary()
# Get the weight matrix of embedding layer
# (these are the numerical representations of each token in our training data, wich have been learned for 5 epochs)
embed_weights = model_1.get_layer('embedding').get_weights()[0]
print(embed_weights.shape) # same size as vocab size and embedding dim (output dim of our embedding layer)
# Create embedding files (we got this from TensorFlo's word embedding documentation )

import io
out_v = io.open('vectors.tsv', 'w', encoding='utf-8') # write vectors.tsv
out_m = io.open('metadata.tsv', 'w', encoding='utf-8') # write metadata.tsv

for index, word in enumerate(words_in_vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = embed_weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()

# Create an LSTM model
from tensorflow.keras import layers
inputs = layers.Input(shape=(1,), dtype='string')
x = text_vectorizer(inputs)
x = embedding(x)
# print(x.shape)
# x = layers.LSTM(64, return_sequences=True)(x) # when you're stacking RNN cells together, you will need to return_sequences=True 
# print(x.shape)
x = layers.LSTM(64)(x)
# print(x.shape)
# x = layers.Dense(64, activation='relu')(x)
# print(x.shape)
outputs = layers.Dense(1, activation='sigmoid')(x)
model_2 = tf.keras.Model(inputs, outputs, name='model_2_LSTM')
# get a summary of the model
model_2.summary()
# Compile the model
model_2.compile(loss='binary_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
# Fit the model
model_2_history = model_2.fit(train_sentences,
                                train_labels, 
                                epochs=5, 
                                validation_data=(val_sentences, val_labels),
                                callbacks=[create_tensorboard_callback(SAVE_DIR,
                                                                    'model_2_LSTM')])
# make predictions with LSTM model
model_2_pred_probs = model_2.predict(val_sentences)
model_2_pred_probs[:10]
# Convert model 2 pred probs to labels
model_2_preds = tf.squeeze(tf.round(model_2_pred_probs))
model_2_preds[:10]
# Calculate model 2 results
model_2_results = calculate_results(y_true=val_labels, 
                                    y_pred=model_2_preds)
model_2_results
# Build an RNN using the GRU cell
from tensorflow.keras import layers
inputs = layers.Input(shape=(1, ), dtype=tf.string)
x = text_vectorizer(inputs)
x = embedding(x)
# x = layers.GRU(64, return_sequences=True)(x) # if you want to stack recurrent layers on top of each other, you need to set return_sequences=True
# x = layers.LSTM(64, return_sequences=True)(x)
x = layers.GRU(64)(x)
# x = layers.Dense(64, activation="relu")(x)
# x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model_3 = tf.keras.Model(inputs, outputs, name="model_3_GRU")
model_3.summary()
# Compile the model
model_3.compile(loss="binary_crossentropy", 
                optimizer="adam", 
                metrics=["accuracy"])
# Fit the model
model_3_history = model_3.fit(train_sentences, 
                              train_labels, 
                              epochs=5, 
                              validation_data=(val_sentences, val_labels), 
                              callbacks=[create_tensorboard_callback(SAVE_DIR, 
                                                                     "model_3_GRU")])

# make some predicitions with our GRU model
model_3_pred_probs = model_3.predict(val_sentences)
model_1_preds[:10]
# Convert model 3 pred probs to labels
model_3_preds = tf.squeeze(tf.round(model_3_pred_probs))
model_3_preds[:10]
# Calculate model 3 results
model_3_results = calculate_results(y_true=val_labels, y_pred=model_3_preds)
model_3_results
# Build a bidirectional RNN in TensorFlow
from tensorflow.keras import layers
inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = embedding(x)
#x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model_4 = tf.keras.Model(inputs, outputs, name="model_4_bidirectional")
model_4.summary()
# Compile model
model_4.compile(loss="binary_crossentropy", 
                optimizer=tf.keras.optimizers.Adam(), 
                metrics=["accuracy"])
# Fit the model
model_4_history = model_4.fit(train_sentences,
                              train_labels,
                              validation_data=(val_sentences, val_labels),
                              epochs=5,
                              callbacks=[create_tensorboard_callback("SAVE_DIR", "model_4_bidirectional")])
# make predictions with our bidirectional model
model_4_pred_probs = model_4.predict(val_sentences)
model_4_pred_probs[:10]
# Convert pred probs to pred labels
model_4_preds = tf.squeeze(tf.round(model_4_pred_probs))
model_4_preds[:10]
# Calculate the results of our bidirectional model
model_4_results = calculate_results(y_true=val_labels, y_pred=model_4_preds)
model_4_results
# test out our embedding layer, Conv1D layer and max_pooling layer
embedding_test = embedding(text_vectorizer(["Hello world"])) # turn target sequence into embedding
conv_1D = layers.Conv1D(filters=64, # 
                        kernel_size=5, # kernel_size is the size of the convolutional window (5 words at a time)
                        strides=1,
                        activation="relu",
                        padding="same") # padding="valid" means that the output will be smaller than the input, "same" is the same of the input between layers
conv_1D_output = conv_1D(embedding_test) # apply the convolutional layer to the embedding layer output
max_pool = layers.GlobalMaxPool1D() # max_pooling layer
max_pool_output = max_pool(conv_1D_output) # apply the max_pooling layer to the convolutional layer output the most important features from the sequence.


embedding_test.shape, conv_1D_output.shape, max_pool_output.shape
# embedding_test


# conv_1D_output
# max_pool_output
# Create 1-dimensional convolutional layer to model sequences
from tensorflow.keras import layers
inputs = layers.Input(shape=(1,), dtype= tf.string)
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.Conv1D(filters=64, kernel_size=5, strides=1, activation='relu', padding="valid")(x)
x = layers.GlobalMaxPool1D()(x)
# x = layers.Dense(64, activation='relu')
outputs = layers.Dense(1, activation='sigmoid')(x)
model_5 = tf.keras.Model(inputs, outputs, name="model_5_Conv1D")

# Compile Conv1D
model_5.compile(loss='binary_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])

#Get a summary of our model
model_5.summary()

# Fit the model
from gc import callbacks


model_5_history = model_5.fit(train_sentences, 
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[create_tensorboard_callback(SAVE_DIR, "Conv1D")])
# make some predictions with our Conv1D model
model_5_pred_probs = model_5.predict(val_sentences)
model_5_pred_probs[:10]
# Convert model 5 pred probs to labels
model_5_preds = tf.squeeze(tf.round(model_5_pred_probs))
model_5_preds[:10]
# Evaluate model 5 predictions
model_5_results = calculate_results(y_true=val_labels, 
                                    y_pred=model_5_preds)
model_5_results
baseline_results
sample_sentence
import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embed_samples = embed([sample_sentence, "When you can the universal sentence encoder on a sentence, it turns it into numbers."])
print(embed_samples[0][:50])
embed_samples[0].shape
# Create a keras layer using the USE (Universal Sentence Encoder) pretrained layer from tensorflow hub
sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", 
                                        input_shape=[], 
                                        dtype=tf.string, 
                                        trainable=False,
                                        name="USE")
# Create model using the Functional API (solves TF Hub Sequential compatibility in Keras 3)
inputs = layers.Input(shape=(), dtype=tf.string)
x = sentence_encoder_layer(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid', name='output_layer')(x)
model_6 = tf.keras.Model(inputs, outputs, name="model_6_USE")

# Compile model
model_6.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model_6.summary()
# Train a classifier on top of USE pretrained embeddings
model_6_history = model_6.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[create_tensorboard_callback(SAVE_DIR, "tf_hub_sentence_encoder")])
# Make predictions with USE TF Hub model
model_6_pred_probs = model_6.predict(val_sentences)
model_6_pred_probs[:10]
# Convert prediction probabilities to labels
model_6_preds = tf.squeeze(tf.round(model_6_pred_probs))
model_6_preds[:10]
# calculate model 6 performance metrics
model_6_results = calculate_results(y_true=val_labels,
                                    y_pred=model_6_preds)
model_6_results


baseline_results
len(train_sentences)
## Note: make data splits like below leads to data leakage (model_7 trained on 10% data, outperforms model_6 trained on 100% data)
## DO NOT MAKE DATA SPLITS WHICH LEAK DATA FROM VALIDATION/TEST SETS INTO TRAINING SETS

# create subsets of 10% of the training data
# train_10_percent = train_df_shuffled[["text","target"]].sample(frac=0.1, random_state=42)
# # train_10_percent.head(), len(train_10_percent)
# train_sentences_10_percent = train_10_percent["text"].to_list()
# train_labels_10_percent = train_10_percent["target"].to_list()
# len(train_sentences_10_percent), len(train_labels_10_percent)
# MAKING A BETTER DATASE SPLIT (NO DATA LEAKAGE)
train_10_percent_split = int(len(train_sentences)*0.1)
train_sentences_10_percent = train_sentences[:train_10_percent_split]
train_labels_10_percent = train_labels[:train_10_percent_split]
# Check the number of each label in the updated training data subset
pd.Series(np.array(train_labels_10_percent)).value_counts()
# Check the number of targets in our subset of data
#train_labels["target"].value_counts()
train_df_shuffled["target"].value_counts()
import tensorflow as tf
# let's build a model the same as model_6
model_7 = tf.keras.models.clone_model(model_6)

# Compile the model
model_7.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#Get a summary (will be the same as model_6)
model_7.summary()


# --- Adicionado conforme solicitado ---
# 1. Identificando os dois melhores modelos:
best_models = all_model_results.sort_values(by='f1', ascending=False).head(2)
print('Os dois melhores modelos são:')
print(best_models)

# 2. Setando o TensorBoard para a pasta de logs dos modelos
%load_ext tensorboard
%tensorboard --logdir "model_logs" --port 6010

# Fi the model to 10% training data subsets
model_7_history = model_7.fit(train_sentences_10_percent,
                              train_labels_10_percent,
                              epochs=5,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[create_tensorboard_callback(SAVE_DIR, "tf_hub_sentence_encoder_10_percent_correct_split")])
# make some predictions with the model trained on 10% of the data
model_7_pred_probs = model_7.predict(val_sentences)
model_7_pred_probs[:10]
# Turn pred probs into labels
model_7_preds = tf.squeeze(tf.round(model_1_pred_probs))
model_7_preds[:10]
# Evaluate model 7 predictions
model_7_results = calculate_results(y_true=val_labels,
                                    y_pred=model_7_preds)

model_7_results
model_6_results
# Combine model results into a Dataframe
all_model_results = pd.DataFrame({"baseline": baseline_results,
                                   "1_simple_dense": model_1_results,
                                   "2_lstm":  model_2_results,
                                    "3_gru": model_3_results,
                                    "4_bidirectional": model_4_results,
                                    "5_conv1d": model_5_results,
                                    "6_tf_hub_use_encoder": model_6_results,
                                    "7_tf_hub_use_encoder_10_percent": model_7_results})
all_model_results = all_model_results.transpose()
all_model_results
# Reduce the accuracy to same scale as other metrics
all_model_results["accuracy"] = all_model_results["accuracy"]/100
import matplotlib.pyplot as plt
import seaborn as sns

# Apply a clean and modern seaborn style
sns.set_theme(style="whitegrid")

# Create the plot using pandas, but with the 'mako' colormap from seaborn
ax = all_model_results.plot(kind="bar", figsize=(12, 7), cmap="mako")

# Customize title and axes
plt.title("Model Performance Comparison", fontsize=16, fontweight="bold", pad=15)
plt.xlabel("Models", fontsize=12)
plt.ylabel("Metrics", fontsize=12)

# Rotate x-axis labels to make them easier to read
plt.xticks(rotation=45, ha="right")

# Adjust the legend to sit outside the plot
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Metrics")

# Remove the top and right spines (makes the plot look cleaner)
sns.despine()

# Ensure nothing (like the legend or x-axis labels) gets cut off
plt.tight_layout()

# Show the plot
plt.show()
# Sort model results by f1-score
all_model_results.sort_values(by=["f1"], ascending=False)["f1"].plot(kind="bar", figsize=(10,7));

%load_ext tensorboard

# Testando abrir a pasta de um único modelo
%tensorboard --logdir "C:\Project_1\model_logs" --port 6010
# If didn't work, try this:
!tensorboard --logdir "C:\Project_1\model_logs" --port 6010
 # Save TF Hub Sentence Encoder model to HDF5 format
model_6.save("model_6_h5")
# Load model with custom Hub layer (required HDF5 format)
loaded_model_6 = tf.keras.models.load_model("model_6_h5",
                                            custom_objects={"KerasLayer": hub.KerasLayer})
# How does our model perform?
loaded_model_6.evaluate(val_sentences, val_labels)
# Save TF hub sentence Encoder model to Saved_model format (default)
model_6.save("model_6_SavedModel_format")
# Load in amodel from the savemodel format
loaded_model_6_SavedModel_format = tf.keras.models.load_model("model_6_SavedModel_format")
# Evaluate model in SavedModel format
loaded_model_6_SavedModel_format.evaluate(val_sentences, val_labels)
# import zipfile
# import urllib.request

# # Baixar o arquivo ZIP
# url = "https://storage.googleapis.com/ztm_tf_course/08_model_6_USE_feature_extractor.zip"
# zip_path = "08_model_6_USE_feature_extractor.zip"
# urllib.request.urlretrieve(url, zip_path)

# # # Descompactar o arquivo ZIP
# with zipfile.ZipFile(zip_path, "r") as zip_ref:
#     zip_ref.extractall()
# Import previously trained model from Google storage
model_6_pretrained = tf.keras.models.load_model("08_model_6_USE_feature_extractor")
model_6_pretrained.evaluate(val_sentences, val_labels)
# Make predi
# Create a dataframe with validation sentences and best performing model predictions
model_6_pretrained_pred_probs = model_6_pretrained.predict(val_sentences)
model_6_pretrained_preds = tf.squeeze(tf.round(model_6_pretrained_pred_probs))
model_6_pretrained_preds[:10] # these should be in label format
# Create Dataframe with validation sentences, validation labels and best performing model predictions labels + probabilities
val_df = pd.DataFrame({"text": val_sentences,
                       "target": val_labels,
                       "pred": model_6_pretrained_preds,
                       "pred_prob": tf.squeeze(model_6_pretrained_pred_probs)})
val_df.head()

# Find the wrong predictions and sort by prediction probabilities
most_wrong = val_df[val_df["target"]!= val_df["pred"]].sort_values("pred_prob", ascending=False)
most_wrong[:10]
most_wrong.tail()
# Check the false negatives (model predicted 0 when should have been 1)
for row in most_wrong[-10:].itertuples():
    _, text, target, pred, pred_prob = row
    print(f"Target: {target}, Pred: {pred}, Prob: {pred_prob}")
    print(f"Text: \n{text}\n")
    print("-----\n")

test_df
test_sentences = test_df['text'].tolist()
test_samples = random.sample(test_sentences, 10)
for test_sample in test_samples:
    pred_prob = tf.squeeze(model_6_pretrained.predict([test_sample])) # get probability of prediction
    pred = tf.round(pred_prob)
    print(f"Pred: {int(pred)}, Prob: {pred_prob}") # print predicted class and"
    print(f"Text: {test_sample}") # print original
    print("---------\n")
 
    

# --- Avaliação dos Melhores Modelos e TensorBoard ---
# 1. Identificando os dois melhores modelos:
best_models = all_model_results.sort_values(by='f1', ascending=False).head(2)
print('Os dois melhores modelos são:')
print(best_models)

# 2. Setando o TensorBoard para a pasta de logs dos modelos
%load_ext tensorboard
%tensorboard --logdir "model_logs" --port 6010
