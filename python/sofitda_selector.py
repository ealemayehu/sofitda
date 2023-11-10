from datetime import datetime

import argparse
import numpy as np

import common

DATASET_DIRECTORY_FORMAT = "../data/stage3/%s"


def read_clusters(embedding_directory, output_directory, cluster_count, class_id):
    root_ids = common.read_embeddings_root_ids(
        embedding_directory, cluster_count, class_id
    )
    relevance_scores = compute_scores(root_ids, output_directory, class_id)

    file_path = f"{embedding_directory}/testing_output_embeddings_clusters_k{cluster_count}_c{class_id}.txt"

    with open(file_path, "rt") as file:
        clusters = {}
        index = 0

        for line in file.read().split("\n"):
            if len(line) == 0:
                continue

            columns = line.split(",")
            root_id = int(columns[0])
            cluster_id = int(columns[1])

            if cluster_id not in clusters:
                clusters[cluster_id] = []

            clusters[cluster_id].append(
                {"root_id": root_id, "score": relevance_scores[index]}
            )

            index += 1

        for cluster_id, cluster in clusters.items():
            cluster.sort(reverse=True, key=(lambda item: item["score"]))

    return clusters


def write_selected_ids(root_ids, output_directory, cluster_count, alpha, class_id):
    file_path = f"{output_directory}/testing_sofitda_k{cluster_count}_a{alpha}_c{class_id}_root_ids.txt"

    with open(file_path, "wt") as file:
        for root_id in root_ids:
            file.write("%d\n" % root_id)


def write_selected_documents(
    root_ids, output_directory, cluster_count, alpha, class_id, document_textmap
):
    print(f"Selected Root ID count: {len(root_ids)}")

    file_path = f"{output_directory}/testing_sofitda_k{cluster_count}_a{alpha}_c{class_id}_documents.txt"

    with open(file_path, "wt") as file:
        for root_id in root_ids:
            file.write(f"{document_textmap[root_id]}\n")


def write_duration(
    output_directory,
    cluster_count,
    alpha,
    class_id,
    duration,
):
    file_path = f"{output_directory}/testing_sofitda_k{cluster_count}_a{alpha}_c{class_id}_duration.txt"

    with open(file_path, "wt") as file:
        file.write(f"{duration}\n")


def write_selection_stats(
    cluster_position,
    clusters,
    output_directory,
    cluster_count,
    alpha,
    class_id,
    total_count,
):
    file_path = f"{output_directory}/testing_sofitda_k{cluster_count}_a{alpha}_c{class_id}_selection_stats.txt"

    with open(file_path, "wt") as file:
        cluster_ids = list(clusters.keys())
        cluster_ids.sort()

        for cluster_id in cluster_ids:
            selected_count = cluster_position[cluster_id]
            cluster_count = len(clusters[cluster_id])
            selected_percentage = "%.2f" % (
                float(selected_count) / float(cluster_count) * 100.0
            )
            selected_share_percentage = "%.2f" % (
                float(selected_count) / float(total_count) * 100.0
            )

            file.write(
                f"{cluster_id}, {selected_count}, {cluster_count}, {selected_percentage}, {selected_share_percentage}\n"
            )


def get_document_text_map(generator, dataset_name, class_id):
    map = {}

    file_path = f"../data/stage3/genrank_{generator}_{dataset_name}/all_document_word_dataset_text.txt"

    with open(file_path, "rt") as file:
        content = file.read()

        for line in content.split("\n"):
            if line == "":
                continue

            first_space_index = line.index(" ")
            second_space_index = line.index(" ", first_space_index + 1)

            document_id = int(line[0:first_space_index])
            document_class_id = int(line[first_space_index + 1 : second_space_index])
            document_text = line[second_space_index + 1 :]

            if document_class_id == class_id:
                map[document_id] = document_text

    return map


def compute_scores(root_ids, output_directory, class_id):
    relevance_scores = common.read_relevance(root_ids, output_directory, class_id)
    print("relevance_scores shape: %s" % str(relevance_scores.shape))
    return np.squeeze(relevance_scores)


def get_select_count(dataset_name, class_id):
    dataset_directory = DATASET_DIRECTORY_FORMAT % dataset_name
    select_count = common.max_count(dataset_directory) - common.class_count(
        class_id, dataset_directory
    )
    return select_count


def select(
    output_directory,
    clusters,
    dataset_name,
    classifier,
    cluster_count,
    alpha,
    class_id,
    document_text_map,
):
    select_count = get_select_count(dataset_name, class_id)

    cluster_sum = np.zeros(cluster_count)
    cluster_positions = {}
    selected_root_ids = []

    start = datetime.now()

    for i in range(0, select_count):
        max_cluster_id = None
        max_score = None
        max_root_id = None
        max_diff = None

        for cluster_id, cluster in clusters.items():
            if cluster_id not in cluster_positions:
                cluster_positions[cluster_id] = 0

            cluster_position = cluster_positions[cluster_id]

            if cluster_position >= len(cluster):
                continue

            score = cluster[cluster_position]["score"]

            cluster_sum[cluster_id] += score
            value = np.sum(np.power(cluster_sum, alpha))
            cluster_sum[cluster_id] -= score
            diff = value - np.sum(np.power(cluster_sum, alpha))

            if max_cluster_id is None or max_diff < diff:
                max_cluster_id = cluster_id
                max_score = score
                max_root_id = cluster[cluster_position]["root_id"]
                max_diff = diff

        cluster_sum[max_cluster_id] += max_score
        cluster_positions[max_cluster_id] += 1
        selected_root_ids.append(max_root_id)

    duration = (datetime.now() - start).total_seconds()
    print(f"Duration: {duration}")

    write_duration(output_directory, cluster_count, alpha, class_id, duration)

    write_selected_ids(
        selected_root_ids, output_directory, cluster_count, alpha, class_id
    )
    write_selected_documents(
        selected_root_ids,
        output_directory,
        cluster_count,
        alpha,
        class_id,
        document_text_map,
    )
    write_selection_stats(
        cluster_positions,
        clusters,
        output_directory,
        cluster_count,
        alpha,
        class_id,
        select_count,
    )


def main():
    parser = argparse.ArgumentParser(description="Executes the selection algorithm")
    parser.add_argument(
        "-d", "--dataset_name", required=True, help="Specifies the dataset name"
    )
    parser.add_argument(
        "-m", "--classifier", required=True, help="Specifies the classifier"
    )
    parser.add_argument(
        "-c", "--class_id", required=True, help="Specifies the class ID"
    )
    parser.add_argument(
        "-g", "--generator", required=True, help="Specifies the generator"
    )
    parser.add_argument(
        "-a", "--alpha", required=True, type=float, help="Specifies the alpha value"
    )
    parser.add_argument(
        "-k", "--cluster_count", required=True, help="Specifies the cluster counts"
    )

    args = parser.parse_args()
    classifier = args.classifier.lower()

    embedding_directory = common.get_output_directory(args.generator, args.dataset_name)
    output_directory = common.get_output_directory(
        args.generator, args.dataset_name, classifier
    )

    print("--------------------------------------------")
    print(f"Classifier: {classifier}")
    print(f"Alpha: {args.alpha}")
    print(f"Cluster count: {args.cluster_count}")
    print(f"Class Is: {args.class_id}")
    print(f"Embedding Directory: {embedding_directory}")
    print(f"Output Directory: {output_directory}")
    print(f"Generator: {args.generator}")
    print("--------------------------------------------")

    processes = []

    class_id = int(args.class_id)

    document_text_map = get_document_text_map(
        args.generator, args.dataset_name, class_id
    )

    print(f"Document text map length: {len(document_text_map)} for class ID {class_id}")

    if not common.has_clusters(embedding_directory, args.cluster_count, class_id):
        print(
            f"No clusters for cluster count {args.cluster_count} and class ID {class_id}"
        )
        return

    clusters = read_clusters(
        embedding_directory, output_directory, args.cluster_count, class_id
    )
    select(
        output_directory,
        clusters,
        args.dataset_name,
        classifier,
        int(args.cluster_count),
        args.alpha,
        class_id,
        document_text_map,
    )

    print("Done selecting items")


if __name__ == "__main__":
    main()
