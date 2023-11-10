import argparse
import scipy
import common

from scipy import cluster
from scipy.cluster.vq import kmeans, whiten, vq
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(generator, dataset_name, output_directory, cluster_count, class_id):
    with open(
        f"../data/stage3/genrank_{generator}_{dataset_name}/testing_document_word_dataset_text.txt",
        "rt",
    ) as file:
        lines = file.read().split("\n")
        root_ids = []
        documents = []

        for line in lines:
            if len(line) == 0:
                continue

            first_index = line.index(" ")
            second_index = line.index(" ", first_index + 1)
            root_id = line[0:first_index]
            label = line[first_index + 1 : second_index]
            document = line[second_index + 1 :]

            if label == class_id:
                root_ids.append(root_id)
                documents.append(document)

    if len(documents) > 0:
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(documents)

        write_embeddings_root_ids(root_ids, output_directory, cluster_count, class_id)
        write_embeddings(embeddings, output_directory, cluster_count, class_id)
    else:
        embeddings = []

    return root_ids, embeddings


def write_embeddings_root_ids(root_ids, output_directory, cluster_count, class_id):
    output_file_path = f"{output_directory}/testing_output_embeddings_indexes_k{cluster_count}_c{class_id}.txt"

    with open(output_file_path, "wt") as file:
        for root_id in root_ids:
            file.write(f"{root_id}\n")


def write_embeddings(embeddings, output_directory, cluster_count, class_id):
    print(f"Writing embeddings {embeddings.shape}...")

    output_file_path = f"{output_directory}/testing_output_embeddings_values_k{cluster_count}_c{class_id}.npz"

    scipy.sparse.save_npz(output_file_path, embeddings)


def main():
    parser = argparse.ArgumentParser(description="Computes K-Means")
    parser.add_argument(
        "-d", "--dataset_name", required=True, help="Specifies the dataset name"
    )
    parser.add_argument("-c", "--class_id", required=True, help="Specifies the class")
    parser.add_argument(
        "-g", "--generator", required=True, help="Specifies the generator"
    )
    parser.add_argument(
        "-k",
        "--cluster_count",
        required=True,
        type=int,
        help="Specifies the number of clusters",
    )

    args = parser.parse_args()

    print(f"Generator: {args.generator}")
    print(f"Cluster Count: {args.cluster_count}")
    print(f"Class ID: {args.class_id}")

    output_directory = common.get_output_directory(args.generator, args.dataset_name)

    print("Getting embeddings...")

    root_ids, embeddings = tf_idf(
        args.generator,
        args.dataset_name,
        output_directory,
        args.cluster_count,
        args.class_id,
    )

    if len(root_ids) == 0:
        print(f"No data found for Class ID {args.class_id}")
        return

    mini_batch_kmeans = MiniBatchKMeans(n_clusters=args.cluster_count, random_state=20)
    mini_batch_kmeans.fit(embeddings)
    indexes = mini_batch_kmeans.predict(embeddings)

    print(f"Computed centroid indexes of documents")

    output_file_path = f"{output_directory}/testing_output_embeddings_clusters_k{args.cluster_count}_c{args.class_id}.txt"

    with open(output_file_path, "wt") as file:
        if len(indexes) != len(root_ids):
            print(
                "WARNING: length of cluster assignments does not match length of root_ids"
            )

        stats = {}

        for i in range(0, len(indexes)):
            index = indexes[i]
            file.write(f"{root_ids[i]},{index}\n")

            if index not in stats:
                stats[index] = 0

            stats[index] += 1

    print("Cluster statistics:")

    for index in stats:
        print(f"{index}: {stats[index]}")

    print("Done with K-Means evaluation")


if __name__ == "__main__":
    main()
