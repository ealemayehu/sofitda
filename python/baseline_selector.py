import argparse
import random

import common

DATASET_DIRECTORY_FORMAT = "../data/stage3/%s"


def select_sorted(
    dataset_directory,
    output_directory,
    class_id,
    document_text_map,
    content,
    is_reversed,
):
    document_scores = {}
    document_count = 0

    for line in content.split("\n"):
        if line == "":
            continue

        components = line.split(",")
        document_id = int(components[0])
        actual_class_id = int(components[1])

        if actual_class_id != class_id:
            continue

        score = float(components[3 + class_id])

        if score not in document_scores:
            document_scores[score] = []

        document_scores[score].append(document_id)
        document_count += 1

    print(f"Document count for class ID {class_id}: {document_count}")

    top_count = common.max_count(dataset_directory) - common.class_count(
        class_id, dataset_directory
    )

    sort_direction = "top" if is_reversed else "bottom"

    print(f"{sort_direction.capitalize()} count: {top_count}")

    sorted_scores = sorted(document_scores.keys(), reverse=is_reversed)
    selected_count = 0

    id_file_path = (
        f"{output_directory}/testing_{sort_direction}_ranked_c{class_id}_root_ids.txt"
    )
    document_file_path = (
        f"{output_directory}/testing_{sort_direction}_ranked_c{class_id}_documents.txt"
    )

    print(f"ID file for {sort_direction}: {id_file_path}")
    print(f"Document file path for {sort_direction}: {document_file_path}")

    with open(id_file_path, "wt") as id_file:
        with open(document_file_path, "wt") as document_file:
            for score in sorted_scores:
                for document_id in document_scores[score]:
                    if selected_count == top_count:
                        return
                    else:
                        selected_count += 1

                    id_file.write(f"{document_id}\n")
                    document_file.write(f"{document_text_map[document_id]}\n")


def select_random(
    dataset_directory, output_directory, class_id, document_text_map, content
):
    document_ids = []

    for line in content.split("\n"):
        if line == "":
            continue

        components = line.split(",")
        document_id = int(components[0])
        actual_class_id = int(components[1])

        if actual_class_id != class_id:
            continue

        score = float(components[4])
        document_ids.append(document_id)

    print(f"Document count for class ID {class_id}: {len(document_ids)}")

    random.Random(20).shuffle(document_ids)

    top_count = common.max_count(dataset_directory) - common.class_count(
        class_id, dataset_directory
    )

    selected_document_ids = document_ids[0:top_count]

    print(f"Random count: {top_count}")

    id_file_path = f"{output_directory}/testing_random_c{class_id}_root_ids.txt"
    document_file_path = f"{output_directory}/testing_random_c{class_id}_documents.txt"

    print(f"ID file path for random: {id_file_path}")
    print(f"Document file path for random: {document_file_path}")

    with open(id_file_path, "wt") as id_file:
        with open(document_file_path, "wt") as document_file:
            for document_id in selected_document_ids:
                id_file.write(f"{document_id}\n")
                document_file.write(f"{document_text_map[document_id]}\n")


def get_document_text_map(generator, dataset_name):
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

            map[document_id] = document_text

    return map


def main():
    parser = argparse.ArgumentParser(description="Executes the ranker")
    parser.add_argument(
        "-d", "--dataset_name", required=True, help="Specifies the dataset name"
    )
    parser.add_argument(
        "-m", "--classifier", required=True, help="Specifies the classifier"
    )
    parser.add_argument(
        "-g", "--generator", required=True, help="Specifies the generator"
    )
    parser.add_argument(
        "-c", "--class_ids", required=True, help="Specifies the class ID"
    )

    args = parser.parse_args()
    classifier = args.classifier.lower()

    output_directory = common.get_output_directory(
        args.generator, args.dataset_name, classifier
    )

    file_path = f"{output_directory}/testing_results.txt"

    print(f"Loading file: {file_path}")

    with open(file_path, "rt") as file:
        content = file.read()

    dataset_directory = DATASET_DIRECTORY_FORMAT % args.dataset_name

    document_text_map = get_document_text_map(args.generator, args.dataset_name)

    for class_id in args.class_ids.split(" "):
        class_id = int(class_id)

        print(f"Creating top ranked data for class {class_id}...")
        select_sorted(
            dataset_directory,
            output_directory,
            class_id,
            document_text_map,
            content,
            True,
        )

        print(f"Creating top bottom data for class {class_id}...")
        select_sorted(
            dataset_directory,
            output_directory,
            class_id,
            document_text_map,
            content,
            False,
        )

        print(f"Creating random data for class {class_id}...")
        select_random(
            dataset_directory, output_directory, class_id, document_text_map, content
        )


if __name__ == "__main__":
    main()
