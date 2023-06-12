import keras.backend
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tokenizers
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
from ast import literal_eval
#from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.preprocessing.text import Tokenizer

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow.keras.backend as K

if __name__ == "__main__":
    def main():
        # Load the dataset
        df = pd.read_csv('preprocessed_data.csv', converters={"message": literal_eval})
        df.head(10)
        df.sentiment.replace(-1, 0, inplace=True)
        #%%

        def print_accuracy(clf, X_train, y_train, X_test, y_test):
            y_train_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)
            y_train_pred = tf.cast(tf.round(y_train_pred), tf.int32)
            y_test_pred = tf.cast(tf.round(y_test_pred), tf.int32)
            print('Train accuracy is:', accuracy_score(y_train_pred, y_train))
            print('Test accuracy is:', accuracy_score(y_test_pred, y_test))
            print(classification_report(y_test, y_test_pred))

            ax = plt.subplot()
            sns.heatmap(confusion_matrix(y_test, y_test_pred, normalize='true'), cmap='RdBu_r',annot=True, ax=ax)

            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            plt.show()

        #%%

        # Downloading the word embedding
        #word_emb = api.load('word2vec-google-news-300')

        print('loading...')

        # Or load it from your local folder if you have it available on your drive
        #word_emb = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

        print('loaded word_emb')
        #%%

        def return_embedding_data(df_in):
            df = df_in

            # Dividing the dataset into training and testing
            X_train, X_test, y_train, y_test = train_test_split(df['preprocessed_text'], df["sentiment"], test_size=0.1, stratify=df["sentiment"])
            y_train = y_train.values
            y_test = y_test.values
            X_test = np.stack(X_test)

            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

            X_val = np.stack(X_val)
            X_train = np.stack(X_train)

            tokenizer = Tokenizer(num_words=20000, oov_token='<unk>')
            tokenizer.fit_on_texts(X_train)

            X_train = tokenizer.texts_to_sequences(X_train)
            X_test = tokenizer.texts_to_sequences(X_test)
            X_val = tokenizer.texts_to_sequences(X_val)

            # Get max training sequence length
            maxlen = max([len(x) for x in X_train])

            # Pad the training sequences
            X_train = pad_sequences(X_train, padding='post', truncating='post', maxlen=maxlen)
            X_val = pad_sequences(X_val, padding='post', truncating='post', maxlen=maxlen)
            X_test = pad_sequences(X_test, padding='post', truncating='post', maxlen=maxlen)


            # let us try to do downsampling
            ind_pos = np.where(y_train == 1)[0]
            ind_neg = np.where(y_train == 0)[0]

            len_neg = len(ind_neg)

            ind_pos_downsampled = np.random.choice(ind_pos, size=int(len_neg), replace=False)
            #y_train = np.hstack((y_train[ind_neg], y_train[ind_pos_downsampled]))
            #X_train = np.vstack((X_train[ind_neg], X_train[ind_pos_downsampled]))

            # Upsampling the class with less data by using SMOTE, as done also before
            #sm = SMOTE(sampling_strategy='minority',random_state=42, k_neighbors=1)
            #oversampled_trainX, oversampled_trainY = sm.fit_resample(features_train, y_train)

            return df, X_train, y_train, X_test, y_test, X_val, y_val

        #%%

        df_word2vec, X_train, y_train, X_test, y_test, X_val, y_val = return_embedding_data(df)

        #%% md

        ## Neural Networks
        print('Neural Networks')

        def f1(y_true, y_pred):
            def recall(y_true, y_pred):
                """Recall metric.

                Only computes a batch-wise average of recall.

                Computes the recall, a metric for multi-label classification of
                how many relevant items are selected.
                """
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
                recall = true_positives / (possible_positives + K.epsilon())
                return recall

            def precision(y_true, y_pred):
                """Precision metric.

                Only computes a batch-wise average of precision.

                Computes the precision, a metric for multi-label classification of
                how many selected items are relevant.
                """
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                precision = true_positives / (predicted_positives + K.epsilon())
                return precision

            pos = tf.where(y_true == 1)
            y_true_pos = tf.gather(y_true, pos)
            y_pred_pos = tf.gather(y_pred, pos)
            precision_pos = precision(y_true_pos, y_pred_pos)
            recall_pos = recall(y_true_pos, y_pred_pos)

            prc_pos = 2 * ((precision_pos * recall_pos) / (precision_pos + recall_pos + K.epsilon()))

            neg = tf.where(y_true == 0)
            y_true_neg = tf.gather(y_true, neg)
            y_pred_neg = tf.gather(y_pred, neg)
            precision_neg = precision(y_true_neg, y_pred_neg)
            recall_neg = recall(y_true_neg, y_pred_neg)

            prc_neg = 2 * ((precision_neg * recall_neg) / (precision_neg + recall_neg + K.epsilon()))
            return (prc_neg + prc_pos) / 2


        filepath = "dropout_0.5_2_best_weights.{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_f1', verbose=1, save_best_only=True, mode='max')
        early_stop = EarlyStopping(monitor='val_f1', patience=5, mode='max')
        callbacks_list = [checkpoint, early_stop]

        neg = np.sum(y_train == 0)
        pos = np.sum(y_train == 1)
        total = neg + pos

        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)

        class_weight = {0: weight_for_0, 1: weight_for_1}

        print('Weight for class 0: {:.2f}'.format(weight_for_0))
        print('Weight for class 1: {:.2f}'.format(weight_for_1))

        class Model(tf.keras.Model):
            def __init__(self):
                inputs = tf.keras.layers.Input(shape=[None], ragged=True)
                layer = inputs

                ones_like = tf.ones_like(layer, dtype=tf.float32)
                dropout = tf.keras.layers.Dropout(0.2)(ones_like)
                layer = layer * tf.cast(dropout != 0, tf.float32)

                layer = tf.keras.layers.Embedding(20000, 256)(layer)
                l = tf.keras.layers.LSTM(64)
                layer = tf.keras.layers.Bidirectional(l, merge_mode='sum')(layer)
                layer = tf.keras.layers.Dense(units=256, activation='relu')(layer)
                """
                layer = tf.keras.layers.LayerNormalization()(layer_to_add)
                layer = tf.keras.layers.Dense(units=32, activation=tf.nn.relu)(layer)
                layer = tf.keras.layers.Dense(units=16)(layer)
                layer = tf.keras.layers.Add()([layer, layer_to_add])
                """

                #layer = tf.keras.layers.GlobalAveragePooling1D()(layer)

                layer = tf.keras.layers.Dropout(0.5)(layer)

                outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(layer)

                super().__init__(inputs=inputs, outputs=outputs)

                self.compile(optimizer=tf.keras.optimizers.Adam(jit_compile=False),
                             loss=tf.losses.BinaryFocalCrossentropy(),
                             metrics=[keras.metrics.BinaryAccuracy(name='accuracy'),
                                      keras.metrics.Precision(name='precision'),
                                      keras.metrics.Recall(name='recall'),
                                      keras.metrics.AUC(name='auc'),
                                      keras.metrics.AUC(name='prc', curve='PR'),
                                      f1])

        model = Model()

        model.fit(X_train, y_train, batch_size=128, epochs=150, validation_data=(X_val, y_val), callbacks=callbacks_list)

        print_accuracy(model, X_train, y_train, X_test, y_test)


main()