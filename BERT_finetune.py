import tensorflow as tf
import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow.keras.backend as K
import sklearn.model_selection

import transformers
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


BATCH_SIZE = 256
EPOCHS = 15

class Model(tf.keras.Model):
    def __init__(self, bert, trainlen):
        inputs = {
            'input_ids': tf.keras.layers.Input(shape=[None]),
            'attention_mask': tf.keras.layers.Input(shape=[None])
        }

        dense_inputs = tf.cast(inputs['input_ids'], tf.int32)
        electra_output = bert(dense_inputs, attention_mask=inputs['attention_mask']).last_hidden_state
        layer = tf.RaggedTensor.from_tensor(electra_output)
        layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(layer)

        outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(layer)

        super().__init__(inputs=inputs, outputs=outputs)

        steps_per_epoch = trainlen // BATCH_SIZE
        num_train_steps = steps_per_epoch * EPOCHS
        warmup_steps = int(0.01 * num_train_steps)
        initial_la = 5e-5

        class LearningScheduleWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, initial_learning_rate, warmup_steps):
                self.multiplicative_factor = tf.constant(1 / tf.math.sqrt(tf.cast(initial_learning_rate, tf.float64)))
                self.warmup_steps = warmup_steps

            def __call__(self, steps):
                min = tf.math.minimum(1 / tf.math.sqrt(tf.cast(steps, tf.float64)),
                                      steps / self.warmup_steps * 1 / tf.math.sqrt(
                                          tf.cast(self.warmup_steps, tf.float64)))
                return self.multiplicative_factor * min

        # cosine_decay = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=initial_la, decay_steps=num_train_steps, alpha=alpha)

        optimizer = tf.keras.optimizers.Adam(jit_compile=False,
                                                           learning_rate=LearningScheduleWithWarmup(
                                                               initial_learning_rate=initial_la,
                                                               warmup_steps=warmup_steps),
                                                           weight_decay=0.001)

        self.compile(optimizer=optimizer,
                     loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                     metrics=[tf.keras.metrics.CategoricalAccuracy('accuracy')])


def main():
    df = pd.read_csv('preprocessed_data.csv', converters={"message": literal_eval})
    df.head(10)
    df.sentiment.replace(-1, 0, inplace=True)

    def print_accuracy(clf, X_train, y_train, X_test, y_test):
        predictions = clf.predict(X_train)
        predictions = tf.nn.softmax(predictions.logits)
        y_train_pred = []
        for sentence in predictions:
            y_train_pred.append(tf.argmax(sentence))
        y_train_pred = np.array(y_train_pred)
        #print(y_train_pred)
        #print(y_train)

        predictions = clf.predict(X_test)
        predictions = tf.nn.softmax(predictions.logits)
        y_test_pred = []
        for sentence in predictions:
            y_test_pred.append(tf.argmax(sentence))
        y_test_pred = np.array(y_test_pred)
        #print(y_test_pred)

        print('Train accuracy is:', accuracy_score(y_train_pred, y_train))
        print('Test accuracy is:', accuracy_score(y_test_pred, y_test))
        print(classification_report(y_test, y_test_pred))

        ax = plt.subplot()
        sns.heatmap(confusion_matrix(y_test, y_test_pred, normalize='true'), cmap='RdBu_r', annot=True, ax=ax)

        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        plt.show()


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

    tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = transformers.TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df['original_message'], df["sentiment"], test_size=0.1,
                                                        stratify=df["sentiment"])
    y_train = y_train.values
    y_test = y_test.values
    X_test = np.stack(X_test)

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)
    X_val = np.stack(X_val)
    X_train = np.stack(X_train)

    def create_dataset(X, y, name):
        """if name != 'test':
            # Creating one hot vectors for labels so that we can use label smoothing
            labels = tf.keras.utils.to_categorical(y, num_classes=2)
        else:
            labels = None"""
        labels = tf.keras.utils.to_categorical(y, num_classes=2)

        batch = [tokenizer.encode(x) for x in X]
        max_length = max(len(x) for x in X)

        batch_ids = np.zeros([len(batch), max_length], dtype=np.int32)
        batch_masks = np.zeros([len(batch), max_length], dtype=np.int32)
        for i in range(len(batch)):
            batch_ids[i, :len(batch[i])] = batch[i]
            batch_masks[i, :len(batch[i])] = 1

        print(batch_ids)

        dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': batch_ids, 'attention_mask': batch_masks}, labels))
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return dataset

    def create_dataset_2(X, y, name):
        """if name != 'test':
            # Creating one hot vectors for labels so that we can use label smoothing
            labels = tf.keras.utils.to_categorical(y, num_classes=2)
        else:
            labels = None"""

        X = [tokenizer(x, return_tensors='tf', padding=True) for x in X]
        print(X)
        dataset = tf.data.Dataset.from_tensor_slices((X, labels))
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return dataset

    train = create_dataset(X_train, y_train, 'train')
    val = create_dataset(X_val, y_val, 'validation')
    test = create_dataset(X_test, y_test, 'test')

    steps_per_epoch = len(y_train) // BATCH_SIZE
    num_train_steps = steps_per_epoch * EPOCHS
    initial_la = 5e-5
    final_la = 5e-6
    alpha = final_la / initial_la

    cosine_decay = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=initial_la,
                                                             decay_steps=num_train_steps, alpha=alpha)

    bert_model.compile(optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=cosine_decay),
                       loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                       metrics=[f1])

    #model = Model(bert_model, len(X_train))

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_f1', patience=5, mode='max')
    callbacks_list = [early_stop]

    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    total = neg + pos

    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

    bert_model.fit(train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=val,
              validation_batch_size=BATCH_SIZE, callbacks=callbacks_list, class_weight=class_weight)

    print_accuracy(bert_model, train, y_train, test, y_test)


if __name__ == "__main__":
    main()