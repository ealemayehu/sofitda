import scipy
import numpy as np
import re
import os.path
import random


DATASET_DIRECTORY_FORMAT = "../data/stage3/%s"


def class_count(class_id, dataset_directory):
    response_count_file_path = "%s/training_response_count_dataset.txt" % (
        dataset_directory
    )

    with open(response_count_file_path, "rt") as file:
        text = file.read()

    for line in text.split("\n"):
        columns = line.split(" ")

        if len(columns) == 1:
            continue

        if int(columns[0]) == class_id:
            return int(columns[1])

    raise RuntimeError(
        "Class ID %d not found in dataset: %s" % (class_id, dataset_directory)
    )


def number_of_classes(dataset_directory):
    response_count_file_path = "%s/training_response_count_dataset.txt" % (
        dataset_directory
    )

    with open(response_count_file_path, "rt") as file:
        text = file.read()

    return len(text.strip().split("\n"))


def max_count(dataset_directory):
    response_count_file_path = "%s/training_response_count_dataset.txt" % (
        dataset_directory
    )

    with open(response_count_file_path, "rt") as file:
        text = file.read()

    max_class_count = 0

    for line in text.split("\n"):
        columns = line.split(" ")

        if len(columns) < 2:
            continue

        class_count = int(columns[1])

        if class_count > max_class_count:
            max_class_count = class_count

    return max_class_count


def remove_delimiters(line):
    if not line.startswith("[BEGIN]") or not line.endswith("[END]"):
        return None

    begin_index = len("[BEGIN]")
    end_index = len(line) - len("[END]")
    return re.sub("\\s\\s+", " ", line[begin_index:end_index].strip())


def read_embeddings_root_ids(output_directory, cluster_count, class_id):
    file_path = f"{output_directory}/testing_output_embeddings_indexes_k{cluster_count}_c{class_id}.txt"

    if not os.path.exists(file_path):
        return []

    with open(file_path, "rt") as file:
        lines = file.read().split("\n")
        root_ids = []

        for line in lines:
            if len(line) > 0:
                root_ids.append(line)

    return root_ids


def has_clusters(output_directory, cluster_count, class_id):
    file_path = f"{output_directory}/testing_output_embeddings_clusters_k{cluster_count}_c{class_id}.txt"
    return os.path.exists(file_path)


def load_samples(file_path, has_delimiters):
    # samples = set()
    samples = []

    with open(file_path, "rt") as file:
        text = file.read()

        for line in text.split("\n"):
            if has_delimiters:
                sample = remove_delimiters(line)

                if sample is not None and len(sample) > 0:
                    # samples.add(sample)
                    samples.append(sample)
            elif len(line) > 0:
                # samples.add(line)
                samples.append(line)

    return samples


def read_embeddings(output_directory, cluster_count, class_id):
    file_path = f"{output_directory}/testing_output_embeddings_values_k{cluster_count}_c{class_id}.npz"

    if not os.path.exists(file_path):
        return None

    embeddings = scipy.sparse.load_npz(file_path)

    print("Embeddings shape: (%d, %d)" % (embeddings.shape[0], embeddings.shape[1]))
    return embeddings


def read_relevance(root_ids, output_directory, class_id):
    file_path = f"{output_directory}/testing_results.txt"

    document_scores_map = {}

    with open(file_path, "rt") as file:
        lines = file.read()

        for line in lines.split("\n"):
            if len(line) == 0:
                continue

            components = line.split(",")
            document_id = components[0]
            actual_class_id = int(components[1])

            if actual_class_id != class_id:
                continue

            score = float(components[3 + class_id])
            document_scores_map[document_id] = score

    in_order_document_scores = []

    for root_id in root_ids:
        in_order_document_scores.append([document_scores_map[root_id]])

    scores = np.array(in_order_document_scores, dtype=np.float32)

    print(
        f"Relevance score stats: max = {np.max(scores)}, min = {np.min(scores)}, avg = {np.mean(scores)}"
    )

    return scores


def get_output_directory(generator, dataset_name, classifier=None):
    if classifier is not None:
        return f"../output/genrank_{generator}_{dataset_name}_{classifier}"
    else:
        return f"../output/genrank_{generator}_{dataset_name}"


def read_selected_root_ids(
    output_directory, cluster_count, alpha, class_count, baseline
):
    root_ids = []

    for i in range(0, class_count):
        if baseline == "none":
            file_path = f"{output_directory}/testing_sofitda_k{cluster_count}_a{alpha}_c{i}_root_ids.txt"
        else:
            file_path = f"{output_directory}/testing_{baseline}_c{i}_root_ids.txt"

        if not os.path.exists(file_path):
            continue

        with open(file_path, "rt") as file:
            lines = file.read().split("\n")

            for line in lines:
                if len(line) > 0:
                    root_ids.append(line)

    return root_ids


def read_random_root_ids(output_directory, dataset_name, number_of_classes):
    file_path = f"{output_directory}/testing_results.txt"

    print(f"Loading file: {file_path}")

    with open(file_path, "rt") as file:
        content = file.read()

    selected_root_ids = []

    for i in range(0, number_of_classes):
        root_ids = []
        class_id = i

        for line in content.split("\n"):
            if line == "":
                continue

            components = line.split(",")
            root_id = components[0]
            actual_class_id = int(components[1])

            if actual_class_id != class_id:
                continue

            root_ids.append(root_id)

        random.Random(20).shuffle(root_ids)
        dataset_directory = DATASET_DIRECTORY_FORMAT % dataset_name

        selection_count = max_count(dataset_directory) - class_count(
            class_id, dataset_directory
        )

        selected_root_ids.extend(root_ids[0:selection_count])

    return selected_root_ids


def selection_count(dataset_name, class_id):
    dataset_directory = DATASET_DIRECTORY_FORMAT % dataset_name

    selection_count = max_count(dataset_directory) - class_count(
        class_id, dataset_directory
    )

    return selection_count
