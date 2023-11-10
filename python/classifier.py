import argparse
import json

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import os
import numpy as np

from tensorflow import keras
from keras import layers
from keras import losses
from keras import preprocessing
from keras import initializers
from keras.layers.experimental.preprocessing import TextVectorization

from functools import partial

from official.nlp import optimization

AUTOTUNE = tf.data.AUTOTUNE


def load_training_dataset(dataset_name, type, batch_size):
    print(f"Loading {type} dataset...")

    dataset_file_path = (
        f"../data/stage3/{dataset_name}/{type}_document_word_dataset_text.txt"
    )
    documents = []
    labels = []
    unique_labels = set()

    with open(dataset_file_path, "rt") as file:
        content = file.read()

        for line in content.split("\n"):
            if line == "":
                continue

            [_, label, document] = line.split(" ", 2)
            documents.append(document)
            labels.append(int(label))
            unique_labels.add(int(label))

    document_ds = tf.data.Dataset.from_tensor_slices(documents)

    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    label_ds = label_ds.map(
        lambda x: tf.one_hot(x, len(unique_labels)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    combined_ds = tf.data.Dataset.zip((document_ds, label_ds))
    combined_ds = combined_ds.shuffle(buffer_size=batch_size * 8)
    combined_ds = combined_ds.batch(batch_size)
    combined_ds.class_names = list(unique_labels)
    return combined_ds


def load_testing_dataset(dataset_name, type):
    print(f"Loading {type} dataset...")

    dataset_file_path = (
        f"../data/stage3/{dataset_name}/{type}_document_word_dataset_text.txt"
    )
    ids = []
    documents = []
    id_label_map = {}
    labels = set()

    with open(dataset_file_path, "rt") as file:
        content = file.read()

        for line in content.split("\n"):
            if line == "":
                continue

            [id, label, document] = line.split(" ", 2)
            ids.append(id)
            documents.append(document)
            id_label_map[id] = int(label)
            labels.add(int(label))

    print(f"Found {len(documents)} documents belonging to {len(labels)} labels")
    return {"ids": ids, "documents": documents, "id_label_map": id_label_map}


def vectorize_document(vectorize_layer, document, label):
    document = tf.expand_dims(document, -1)
    return vectorize_layer(document), label


def configure_dataset(dataset):
    return dataset.cache().prefetch(buffer_size=AUTOTUNE)


def bag_of_words_model(key_count, vectorize_layer):
    model = tf.keras.Sequential([layers.Dense(key_count)])
    model.compile(
        loss=losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=["accuracy"],
    )
    return model


def bag_of_words(
    output_directory, raw_training, raw_validation, raw_testing, max_vocabulary_size
):
    vectorize_layer = TextVectorization(
        max_tokens=max_vocabulary_size, output_mode="binary"
    )
    training_documents = raw_training.map(lambda document, _: document)

    vectorize_layer.adapt(training_documents)

    training = raw_training.map(partial(vectorize_document, vectorize_layer))
    validation = raw_validation.map(partial(vectorize_document, vectorize_layer))

    training = configure_dataset(training)
    validation = configure_dataset(validation)
    label_map = {}

    for i, label in enumerate(raw_training.class_names):
        label_map[i] = label

    key_count = len(label_map.keys())
    print(f"Label count: {key_count}")

    print(f"Vocabulary Size: {len(vectorize_layer.get_vocabulary())}")

    print("Training...")

    model = bag_of_words_model(key_count, vectorize_layer)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=1, restore_best_weights=True
    )
    history = model.fit(
        training, validation_data=validation, callbacks=[early_stopping], epochs=50
    )

    epochs = get_optimal_epoch(history)

    print(f"Optimal number of epochs: {epochs}")

    print("Predicting...")

    export_model = tf.keras.Sequential(
        [vectorize_layer, model, layers.Activation("softmax")]
    )
    export_model.compile(
        loss=losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=["accuracy"],
    )

    scores = export_model.predict(raw_testing["documents"])

    compute_all_label_metrics(
        output_directory,
        raw_testing["ids"],
        raw_testing["id_label_map"],
        scores,
        label_map,
        epochs,
    )

    save_scores(
        output_directory,
        raw_testing["ids"],
        raw_testing["id_label_map"],
        label_map,
        scores,
    )

    print("DONE")


def tf_idf_model(key_count, vectorize_layer):
    model = tf.keras.Sequential([layers.Dense(key_count)])
    model.compile(
        loss=losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=["accuracy"],
    )
    return model


def tf_idf(
    output_directory, raw_training, raw_validation, raw_testing, max_vocabulary_size
):
    vectorize_layer = TextVectorization(
        max_tokens=max_vocabulary_size, output_mode="tf-idf"
    )
    training_documents = raw_training.map(lambda document, _: document)

    vectorize_layer.adapt(training_documents)

    training = raw_training.map(partial(vectorize_document, vectorize_layer))
    validation = raw_validation.map(partial(vectorize_document, vectorize_layer))

    training = configure_dataset(training)
    validation = configure_dataset(validation)
    label_map = {}

    for i, label in enumerate(raw_training.class_names):
        label_map[i] = label

    key_count = len(label_map.keys())
    print(f"Label count: {key_count}")

    print(f"Vocabulary Size: {len(vectorize_layer.get_vocabulary())}")

    print("Training...")

    model = tf_idf_model(key_count, vectorize_layer)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=1, restore_best_weights=True
    )
    history = model.fit(
        training, validation_data=validation, callbacks=[early_stopping], epochs=50
    )

    epochs = get_optimal_epoch(history)

    print(f"Optimal number of epochs: {epochs}")

    print("Predicting...")

    export_model = tf.keras.Sequential(
        [vectorize_layer, model, layers.Activation("softmax")]
    )
    export_model.compile(
        loss=losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=["accuracy"],
    )

    scores = export_model.predict(raw_testing["documents"])

    compute_all_label_metrics(
        output_directory,
        raw_testing["ids"],
        raw_testing["id_label_map"],
        scores,
        label_map,
        epochs,
    )

    save_scores(
        output_directory,
        raw_testing["ids"],
        raw_testing["id_label_map"],
        label_map,
        scores,
    )

    print("DONE")


def rnn_model(key_count, vectorize_layer):
    embedding_matrix = load_glove_embeddings(vectorize_layer.get_vocabulary())

    model = tf.keras.Sequential(
        [
            layers.Embedding(
                input_dim=len(vectorize_layer.get_vocabulary()),
                output_dim=100,
                embeddings_initializer=initializers.Constant(embedding_matrix),
                mask_zero=True,
            ),
            layers.Bidirectional(layers.LSTM(50)),
            layers.Dense(50, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(key_count),
        ]
    )
    model.compile(
        loss=losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=["accuracy"],
    )
    return model


def rnn(
    output_directory, raw_training, raw_validation, raw_testing, max_vocabulary_size
):
    vectorize_layer = TextVectorization(max_tokens=max_vocabulary_size)
    training_documents = raw_training.map(lambda document, label: document)

    vectorize_layer.adapt(training_documents)

    training = raw_training.map(partial(vectorize_document, vectorize_layer))
    validation = raw_validation.map(partial(vectorize_document, vectorize_layer))

    training = configure_dataset(training)
    validation = configure_dataset(validation)
    label_map = {}

    for i, label in enumerate(raw_training.class_names):
        label_map[i] = label

    key_count = len(label_map.keys())
    print(f"Label count: {key_count}")

    print(f"Vocabulary Size: {len(vectorize_layer.get_vocabulary())}")

    print("Training...")

    model = rnn_model(key_count, vectorize_layer)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=1, restore_best_weights=True
    )
    history = model.fit(
        training, validation_data=validation, callbacks=[early_stopping], epochs=50
    )

    epochs = get_optimal_epoch(history)

    print(f"Optimal number of epochs: {epochs}")

    print("Predicting...")

    export_model = tf.keras.Sequential(
        [vectorize_layer, model, layers.Activation("softmax")]
    )
    export_model.compile(
        loss=losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=["accuracy"],
    )

    scores = export_model.predict(raw_testing["documents"])

    compute_all_label_metrics(
        output_directory,
        raw_testing["ids"],
        raw_testing["id_label_map"],
        scores,
        label_map,
        epochs,
    )

    save_scores(
        output_directory,
        raw_testing["ids"],
        raw_testing["id_label_map"],
        label_map,
        scores,
    )

    print("DONE")


def cnn_model(key_count, vectorize_layer):
    embedding_matrix = load_glove_embeddings(vectorize_layer.get_vocabulary())

    model = tf.keras.Sequential(
        [
            layers.Embedding(
                input_dim=len(vectorize_layer.get_vocabulary()),
                output_dim=100,
                embeddings_initializer=initializers.Constant(embedding_matrix),
            ),
            layers.Conv1D(filters=200, kernel_size=5, activation="relu"),
            layers.GlobalMaxPooling1D(),
            layers.Dense(50, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(key_count),
        ]
    )
    model.compile(
        loss=losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=["accuracy"],
    )
    return model


def cnn(
    output_directory, raw_training, raw_validation, raw_testing, max_vocabulary_size
):
    vectorize_layer = TextVectorization(max_tokens=max_vocabulary_size)
    training_documents = raw_training.map(lambda document, label: document)

    vectorize_layer.adapt(training_documents)

    training = raw_training.map(partial(vectorize_document, vectorize_layer))
    validation = raw_validation.map(partial(vectorize_document, vectorize_layer))

    training = configure_dataset(training)
    validation = configure_dataset(validation)
    label_map = {}

    for i, label in enumerate(raw_training.class_names):
        label_map[i] = label

    key_count = len(label_map.keys())
    print(f"Label count: {key_count}")

    print(f"Vocabulary Size: {len(vectorize_layer.get_vocabulary())}")

    print("Training...")

    model = cnn_model(key_count, vectorize_layer)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=1, restore_best_weights=True
    )
    history = model.fit(
        training, validation_data=validation, callbacks=[early_stopping], epochs=50
    )

    epochs = get_optimal_epoch(history)

    print(f"Optimal number of epochs: {epochs}")

    print("Predicting...")

    print("Predicting...")

    export_model = tf.keras.Sequential(
        [vectorize_layer, model, layers.Activation("softmax")]
    )
    export_model.compile(
        loss=losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=["accuracy"],
    )

    scores = export_model.predict(raw_testing["documents"])

    compute_all_label_metrics(
        output_directory,
        raw_testing["ids"],
        raw_testing["id_label_map"],
        scores,
        label_map,
        epochs,
    )

    save_scores(
        output_directory,
        raw_testing["ids"],
        raw_testing["id_label_map"],
        label_map,
        scores,
    )

    print("DONE")


def bert_model(key_count, epochs, training):
    tfhub_handle_encoder = (
        "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1"
    )
    tfhub_handle_preprocess = (
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    )
    bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name="preprocessing")
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name="BERT_encoder")
    outputs = encoder(encoder_inputs)

    net = outputs["pooled_output"]
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(key_count, activation=None, name="classifier")(net)
    model = tf.keras.Model(text_input, net)

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = tf.metrics.CategoricalCrossentropy()

    steps_per_epoch = tf.data.experimental.cardinality(training).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(
        init_lr=init_lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type="adamw",
    )

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def bert(
    output_directory, raw_training, raw_validation, raw_testing, max_vocabulary_size
):
    training = raw_training.cache().prefetch(buffer_size=AUTOTUNE)
    validation = raw_validation.cache().prefetch(buffer_size=AUTOTUNE)

    label_map = {}

    for i, label in enumerate(raw_training.class_names):
        label_map[i] = label

    key_count = len(label_map.keys())
    print(f"Label count: {key_count}")

    epochs = 15

    model = bert_model(key_count, epochs, training)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=1, restore_best_weights=True
    )
    history = model.fit(
        training, validation_data=validation, callbacks=[early_stopping], epochs=epochs
    )

    epochs = get_optimal_epoch(history)

    print(f"Optimal number of epochs: {epochs}")

    print("Predicting...")

    export_model = tf.keras.Sequential([model, layers.Activation("softmax")])
    export_model.compile(
        loss=losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=["accuracy"],
    )

    scores = export_model.predict(raw_testing["documents"])

    compute_all_label_metrics(
        output_directory,
        raw_testing["ids"],
        raw_testing["id_label_map"],
        scores,
        label_map,
        epochs,
    )

    save_scores(
        output_directory,
        raw_testing["ids"],
        raw_testing["id_label_map"],
        label_map,
        scores,
    )

    print("DONE")


def compute_all_label_metrics(
    output_directory, ids, id_label_map, scores, label_map, optimal_epoch_count
):
    precisions = {}
    recalls = {}
    fscores = {}

    for label in label_map.values():
        precision, recall, fscore = compute_single_label_metrics(
            ids, id_label_map, label_map, scores, label
        )
        precisions[label] = precision
        recalls[label] = recall
        fscores[label] = fscore

    accuracy = compute_accuracy(ids, id_label_map, label_map, scores)

    metrics = {
        "epochs": optimal_epoch_count,
        "accuracy": accuracy,
        "precisions": precisions,
        "recalls": recalls,
        "fscores": fscores,
    }

    print(metrics)

    with open(f"{output_directory}/testing_metrics.json", "wt") as file:
        json.dump(metrics, file, indent=2)


def compute_accuracy(ids, id_label_map, label_map, scores):
    correct = 0.0
    incorrect = 0.0

    for i in range(0, len(scores)):
        max_score = None
        max_label = None

        for label_id, label in label_map.items():
            if max_score is None or max_score < scores[i][label_id]:
                max_score = scores[i][label_id]
                max_label = label

        if id_label_map[ids[i]] == max_label:
            correct += 1
        else:
            incorrect += 1

    return correct / (correct + incorrect + 1e-15)


def compute_single_label_metrics(ids, id_label_map, label_map, scores, positive_label):
    true_positive = 0.0
    false_positive = 0.0
    true_negative = 0.0
    false_negative = 0.0

    for i in range(0, len(scores)):
        max_score = None
        max_label = None

        for label_id, label in label_map.items():
            if max_score is None or max_score < scores[i][label_id]:
                max_score = scores[i][label_id]
                max_label = label

        if id_label_map[ids[i]] == positive_label:
            if max_label == positive_label:
                true_positive += 1
            else:
                false_negative += 1
        else:
            if max_label == positive_label:
                false_positive += 1
            else:
                true_negative += 1

    precision = true_positive / (true_positive + false_positive + 1e-15)
    recall = true_positive / (true_positive + false_negative + 1e-15)
    fscore = 2 * precision * recall / (precision + recall + 1e-15)

    return precision, recall, fscore


def get_optimal_epoch(history):
    validation_loss = history.history["val_loss"]
    optimal_index = 0
    minimum_loss = None

    for i in range(0, len(validation_loss)):
        if minimum_loss is None or validation_loss[i] < minimum_loss:
            minimum_loss = validation_loss[i]
            optimal_index = i

    return optimal_index + 1


def save_scores(output_directory, ids, id_label_map, label_map, scores):
    reverse_label_map = {}

    for label_id, label in label_map.items():
        reverse_label_map[label] = label_id

    with open(f"{output_directory}/testing_results.txt", "wt") as file:
        for i in range(0, len(ids)):
            max_score = None
            max_label = None

            for label_id, label in label_map.items():
                if max_score is None or max_score < scores[i][label_id]:
                    max_score = scores[i][label_id]
                    max_label = label

            id = ids[i]
            actual_label = id_label_map[id]
            predicted_label = max_label
            file.write(f"{id},{actual_label},{predicted_label}")

            labels = list(label_map.values())
            labels.sort()

            for label in labels:
                label_id = reverse_label_map[label]
                file.write(f",{scores[i][label_id]}")

            file.write("\n")


def load_glove_embeddings(vocabulary):
    vocabulary_map = dict(zip(vocabulary, range(len(vocabulary))))
    embedding_map = {}
    found_count = 0

    with open("../data/embeddings/glove.6B.100d.txt", "rt") as file:
        for line in file:
            word, embedding_string = line.split(maxsplit=1)

            if word not in vocabulary_map:
                continue

            found_count += 1

            embedding = np.fromstring(embedding_string, "f", sep=" ")
            embedding_map[word] = embedding

    print(
        f"Found Glove embeddings for {found_count} out of {len(vocabulary_map)} words."
    )

    embedding_matrix = np.zeros((len(vocabulary_map), 100))

    for word, index in vocabulary_map.items():
        if word not in embedding_map:
            continue

        embedding = embedding_map[word]
        embedding_matrix[index] = embedding

    return embedding_matrix


def main():
    parser = argparse.ArgumentParser(description="Executes the classifier")
    parser.add_argument(
        "-d", "--dataset_name", required=True, help="Specifies the dataset name"
    )
    parser.add_argument(
        "-v",
        "--validation_dataset_type",
        default="validation",
        help="Specifies the validation dataset type",
    )
    parser.add_argument(
        "-i",
        "--testing_dataset_type",
        default="testing",
        help="Specifies the inference dataset type",
    )
    parser.add_argument(
        "-o", "--output_directory", required=True, help="Specifies the output directory"
    )
    parser.add_argument(
        "-c", "--classifier", required=True, help="Specifies the classifier"
    )
    parser.add_argument(
        "-m",
        "--max_vocabulary_size",
        required=True,
        type=int,
        help="Specifies the maximum vocabulary size",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=32, help="Specifies the batch size"
    )

    args = parser.parse_args()

    print(f"Avaliable GPUs: {tf.config.list_physical_devices('GPU')}")
    print(f"GPU being used: {tf.test.is_gpu_available(cuda_only=True)}")
    print(f"Selected GPU: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    print(f"Dataset Name: {args.dataset_name}")
    print(f"Validation Dataset Type: {args.validation_dataset_type}")
    print(f"Testing Dataset Type: {args.testing_dataset_type}")
    print(f"Output Directory: {args.output_directory}")
    print(f"Classifier: {args.classifier}")
    print(f"Max Vocabulary Size: {args.max_vocabulary_size}")
    print(f"Batch Size: {args.batch_size}")

    raw_training = load_training_dataset(args.dataset_name, "training", args.batch_size)
    raw_validation = load_training_dataset(
        args.dataset_name, args.validation_dataset_type, args.batch_size
    )
    raw_testing = load_testing_dataset(args.dataset_name, args.testing_dataset_type)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    if args.classifier == "bow":
        bag_of_words(
            args.output_directory,
            raw_training,
            raw_validation,
            raw_testing,
            args.max_vocabulary_size,
        )
    elif args.classifier == "tf-idf":
        tf_idf(
            args.output_directory,
            raw_training,
            raw_validation,
            raw_testing,
            args.max_vocabulary_size,
        )
    elif args.classifier == "rnn":
        rnn(
            args.output_directory,
            raw_training,
            raw_validation,
            raw_testing,
            args.max_vocabulary_size,
        )
    elif args.classifier == "cnn":
        cnn(
            args.output_directory,
            raw_training,
            raw_validation,
            raw_testing,
            args.max_vocabulary_size,
        )
    elif args.classifier == "bert":
        bert(
            args.output_directory,
            raw_training,
            raw_validation,
            raw_testing,
            args.max_vocabulary_size,
        )
    else:
        print(f"Unknown model: {args.classifier}")


if __name__ == "__main__":
    main()
