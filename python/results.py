import json
import os
import argparse

import scipy.stats
import sklearn.metrics
import numpy as np

from sklearn.metrics import roc_auc_score

OUTPUT_DIRECTORY = "../output"
DATA_DIRECTORY = "../data/stage3"

import common


def get_fpr(model_directory, class_id):
    file_path = f"{OUTPUT_DIRECTORY}/{model_directory}/testing_metrics.json"

    with open(file_path, "rt") as file:
        data = json.load(file)

        return {
            "epochs": data["epochs"],
            "accuracy": data["accuracy"],
            f"precision_{class_id}": data["precisions"][str(class_id)],
            f"recall_{class_id}": data["recalls"][str(class_id)],
            f"fscore_{class_id}": data["fscores"][str(class_id)],
        }


def get_class_distribution(model_directory, class_id):
    file_path = (
        f"{DATA_DIRECTORY}/{model_directory}/training_response_count_dataset.txt"
    )

    with open(file_path, "rt") as file:
        class_map = {}
        total_count = 0.0

        for line in file.read().split("\n"):
            if line == "":
                continue

            components = line.split(" ")
            class_map[int(components[0])] = int(components[1])
            total_count += int(components[1])

        return {
            f"class_count_{class_id}": class_map[class_id],
            f"class_percentage_{class_id}": class_map[class_id] / total_count,
        }


def get_class_ids(model_directory):
    file_path = (
        f"{DATA_DIRECTORY}/{model_directory}/training_response_count_dataset.txt"
    )

    with open(file_path, "rt") as file:
        classes = []

        for line in file.read().split("\n"):
            if line == "":
                continue

            components = line.split(" ")
            classes.append(int(components[0]))

    classes.sort()
    return classes


def populate_overlap_statistics(info):
    info["random_overlap"] = 0
    info["bottom_ranked_overlap"] = 0
    info["top_ranked_overlap"] = 0

    if info["model"] != "sofitda":
        return

    dataset_name = info["category"][3:]  # For example, augquora

    class_count = common.number_of_classes(f"../data/stage3/{dataset_name}")
    output_directory = common.get_output_directory(
        info["generator"], dataset_name, info["classifier"]
    )

    sofitda_root_ids = common.read_selected_root_ids(
        output_directory, info["cluster_count"], info["alpha"], class_count, "none"
    )

    for baseline in ["random", "bottom_ranked", "top_ranked"]:
        if baseline == "random":
            baseline_root_ids = common.read_random_root_ids(
                output_directory, dataset_name, class_count
            )
        else:
            baseline_root_ids = common.read_selected_root_ids(
                output_directory,
                info["cluster_count"],
                info["alpha"],
                class_count,
                baseline,
            )
        overlap_count = 0

        baseline_root_ids = set(baseline_root_ids)

        for root_id in sofitda_root_ids:
            if root_id in baseline_root_ids:
                overlap_count += 1

        info[f"{baseline}_overlap"] += overlap_count


def get_weighted_scores(model_directory):
    path = f"{OUTPUT_DIRECTORY}/{model_directory}/testing_results.txt"
    actuals = []
    predictions = []

    with open(path, "rt") as output_file:
        content = output_file.read()

        for line in content.split("\n"):
            if line == "":
                continue

            cells = line.split(",")

            actual_class = int(cells[1])
            predicted_class = int(cells[2])

            actuals.append(actual_class)  # Actual
            predictions.append(predicted_class)  # Predicted

    return {
        "macro_fscore": sklearn.metrics.f1_score(actuals, predictions, average="macro"),
        "macro_precision": sklearn.metrics.precision_score(
            actuals, predictions, average="macro"
        ),
        "macro_recall": sklearn.metrics.recall_score(
            actuals, predictions, average="macro"
        ),
    }


def get_pvalue(model_directory, info, baseline):
    model_partition_fscores = get_partiton_fscores(model_directory)
    baseline_partition_fscores = get_partiton_fscores(
        f"{info['category']}_{baseline}_m{info['classifier']}_g{info['generator']}_ptesting"
    )

    _, p_value = scipy.stats.ttest_ind(
        model_partition_fscores, baseline_partition_fscores
    )

    # print(f"P-Value: {p_value}")
    return p_value


def get_partiton_fscores(model_directory, vector_size=10):
    path = "%s/%s/testing_results.txt" % (OUTPUT_DIRECTORY, model_directory)
    actuals = []
    predictions = []

    with open(path, "rt") as output_file:
        content = output_file.read()

        for line in content.split("\n"):
            if line == "":
                continue

            cells = line.split(",")

            actual_class = int(cells[1])
            predicted_class = int(cells[2])

            actuals.append(actual_class)  # Actual
            predictions.append(predicted_class)  # Predicted

    partition_size = int(len(actuals) / vector_size)
    partition_fscores = []

    for i in range(0, vector_size):
        begin_index = i * partition_size

        if i == vector_size - 1:
            end_index = -1
        else:
            end_index = begin_index + partition_size

        actual_partition = actuals[begin_index:end_index]
        prediction_partition = predictions[begin_index:end_index]
        partition_fscore = sklearn.metrics.f1_score(
            actual_partition, prediction_partition, average="macro"
        )
        partition_fscores.append(partition_fscore)

    # fscore = sklearn.metrics.f1_score(actuals, predictions, average="macro")
    # print(f"F-Score: {fscore}, Partition F-Scores: {partition_fscores}")
    return partition_fscores


def get_mean_precision_rank(model_directory, class_id):
    path = f"{OUTPUT_DIRECTORY}/{model_directory}/testing_results.txt"
    results = []
    count = 0

    with open(path, "rt") as output_file:
        content = output_file.read()

        for line in content.split("\n"):
            if line == "":
                continue

            cells = line.split(",")
            result = []

            document_id = int(cells[0])
            actual_class = int(cells[1])
            predicted_class = int(cells[2])

            result.append(document_id)  # ID
            result.append(actual_class)  # Actual
            result.append(predicted_class)  # Predicted

            for i in range(3, len(cells)):
                result.append(float(cells[i]))  # class score

            results.append(result)
            count += 1

    class_map = {}
    mprs = {}

    for i in range(0, len(result) - 3):
        class_map[i] = {}

    for result in results:
        for j in range(0, len(result) - 3):
            score_index = 3 + j
            score = result[score_index]

            if score not in class_map[j]:
                class_map[j][score] = []

            class_map[j][score].append(result)

    for i in range(0, len(result) - 3):
        scores = list(class_map[i].keys())
        scores.sort(reverse=True)
        rank = 0
        percentiles = []

        for score in scores:
            percentile = float(rank) / count * 100.0

            for result in class_map[i][score]:
                actual_class = result[1]

                if actual_class == i:
                    percentiles.append(percentile)

                rank += 1

        mprs[i] = sum(percentiles) / len(percentiles)

    return mprs[class_id]


def main():
    parser = argparse.ArgumentParser(description="Executes the ranker")
    parser.add_argument(
        "-t",
        "--dataset_type",
        required=False,
        default="all",
        help="Specifies the dataset type",
    )
    parser.add_argument(
        "-m",
        "--classifier",
        required=False,
        default="all",
        help="Specifies the classifier",
    )
    parser.add_argument(
        "-g",
        "--generator",
        required=False,
        default="all",
        help="Specifies the generator",
    )
    parser.add_argument(
        "-d", "--dataset_name", required=True, help="Specifies the dataset name"
    )

    args = parser.parse_args()

    print(f"Dataset Type: {args.dataset_type}")
    print(f"Classifier: {args.classifier}")
    print(f"Generator: {args.generator}")
    print(f"Dataset Name: {args.dataset_name}")

    infos = []

    for name in os.listdir(OUTPUT_DIRECTORY):
        path = f"{OUTPUT_DIRECTORY}/{name}"

        if not os.path.isdir(path):
            continue

        components = name.split("_")
        category = "aug%s" % args.dataset_name

        if components[0] == category:
            print(f"Processing {path}...")

            info = {"cluster_count": "", "alpha": "", "path": path, "name": name}

            info["category"] = category

            # Examples: augquora_sofitda_mbert_c20_a0.1_geda_ptesting
            #           augquora_random_mbert_geda_ptesting
            if components[1] == "sofitda":
                info["model"] = components[1]
                info["classifier"] = components[2][1:]
                info["cluster_count"] = int(components[3][1:])
                info["alpha"] = float(components[4][1:])
                info["generator"] = components[5][1:]
                info["dataset"] = components[6][1:]
                infos.append(info)
            elif (
                components[1] == "bottom"
                or components[1] == "top"
                or components[1] == "no"
            ):
                info["model"] = f"{components[1]}_{components[2]}"
                info["classifier"] = components[3][1:]
                info["generator"] = components[4][1:]
                info["dataset"] = components[5][1:]
                infos.append(info)
            else:
                info["model"] = components[1]
                info["classifier"] = components[2][1:]
                info["generator"] = components[3][1:]
                info["dataset"] = components[4][1:]
                infos.append(info)

            if args.classifier != "all" and args.classifier != info["classifier"]:
                continue

            if args.dataset_type != "all" and args.dataset_type != info["dataset"]:
                continue

            if args.generator != "all" and args.generator != info["generator"]:
                continue

            info["p_value_top_ranked"] = get_pvalue(name, info, "top_ranked")
            info["p_value_random"] = get_pvalue(name, info, "random")
            info["p_value_bottom_ranked"] = get_pvalue(name, info, "bottom_ranked")
            info["p_value_no_aug"] = get_pvalue(name, info, "no_aug")
            info.update(get_weighted_scores(name))

            class_ids = get_class_ids(name)

            for class_id in class_ids:
                info.update(get_fpr(name, class_id))
                info.update(get_class_distribution(name, class_id))

            populate_overlap_statistics(info)

    if (
        args.dataset_type == "all"
        and args.classifier == "all"
        and args.generator == "all"
    ):
        file_path = f"{OUTPUT_DIRECTORY}/results_{args.dataset_name}.csv"
    else:
        file_path = f"{OUTPUT_DIRECTORY}/results_{args.dataset_name}_t{args.dataset_type}_m{args.classifier}_g{args.generator}.csv"

    with open(file_path, "wt") as file:
        columns = [
            "category",
            "model",
            "classifier",
            "dataset",
            "generator",
            "cluster_count",
            "alpha",
            "epochs",
            "accuracy",
            "p_value_top_ranked",
            "p_value_random",
            "p_value_bottom_ranked",
            "p_value_no_aug",
            "macro_fscore",
            "macro_precision",
            "macro_recall",
            "random_overlap",
            "bottom_ranked_overlap",
            "top_ranked_overlap",
        ]

        file.write(",".join(columns))

        if len(infos) > 0:
            class_ids = get_class_ids(infos[0]["name"])

            for class_id in class_ids:
                file.write(
                    f",class_count_{class_id},class_perecentage_{class_id},precision_{class_id},recall_{class_id},fscore_{class_id}"
                )

            file.write("\n")

        for info in infos:
            if args.classifier != "all" and args.classifier != info["classifier"]:
                continue

            if args.dataset_type != "all" and args.dataset_type != info["dataset"]:
                continue

            if args.generator != "all" and args.generator != info["generator"]:
                continue

            file.write(info["category"])
            file.write(",")
            file.write(info["model"])
            file.write(",")
            file.write(info["classifier"])
            file.write(",")
            file.write(info["dataset"])
            file.write(",")
            file.write(info["generator"])
            file.write(",")
            file.write(str(info["cluster_count"]))
            file.write(",")
            file.write(str(info["alpha"]))
            file.write(",")
            file.write(str(info["epochs"]))
            file.write(",")
            file.write(str(info["accuracy"]))
            file.write(",")
            file.write(str(info["p_value_top_ranked"]))
            file.write(",")
            file.write(str(info["p_value_random"]))
            file.write(",")
            file.write(str(info["p_value_bottom_ranked"]))
            file.write(",")
            file.write(str(info["p_value_no_aug"]))
            file.write(",")
            file.write(str(info["macro_fscore"]))
            file.write(",")
            file.write(str(info["macro_precision"]))
            file.write(",")
            file.write(str(info["macro_recall"]))
            file.write(",")
            file.write(str(info["random_overlap"]))
            file.write(",")
            file.write(str(info["bottom_ranked_overlap"]))
            file.write(",")
            file.write(str(info["top_ranked_overlap"]))

            for class_id in class_ids:
                file.write(",")
                file.write(str(info[f"class_count_{class_id}"]))
                file.write(",")
                file.write(str(info[f"class_percentage_{class_id}"]))
                file.write(",")
                file.write(str(info[f"precision_{class_id}"]))
                file.write(",")
                file.write(str(info[f"recall_{class_id}"]))
                file.write(",")
                file.write(str(info[f"fscore_{class_id}"]))

            file.write("\n")


if __name__ == "__main__":
    main()
