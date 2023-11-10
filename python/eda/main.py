from eda import *

import common
import math
import argparse
import random

DATASET_DIRECTORY_FORMAT = "../../data/stage3/%s"


def main():
    parser = argparse.ArgumentParser(description="Executes the EDA text generator")
    parser.add_argument(
        "-d", "--dataset-name", required=True, help="Specifies the dataset name"
    )
    parser.add_argument(
        "-i", "--input-file", required=True, help="Specifies the input file"
    )
    parser.add_argument(
        "-c", "--class-id", type=int, required=True, help="Specifies the class ID"
    )
    parser.add_argument(
        "-f", "--factor", required=True, type=int, help="Specifies the factor"
    )
    parser.add_argument(
        "-o", "--output-file", required=True, help="Specifies the output file"
    )

    args = parser.parse_args()

    existing_samples = common.load_samples(args.input_file, True)
    dataset_directory = DATASET_DIRECTORY_FORMAT % args.dataset_name

    print(f"Dataset: {dataset_directory}")
    print(f"Class ID: {args.class_id}")
    print(f"Training sample count: {len(existing_samples)}")

    ratio = (
        float(common.max_count(dataset_directory))
        / float(common.class_count(args.class_id, dataset_directory))
        * float(args.factor)
    )
    factor = math.ceil(ratio) - 1

    print(f"Ratio: {ratio}, Factor: {factor}")

    if factor == 0:
        print("Not generating")
        return

    error_count = 0

    generated_items = []

    for sample in existing_samples:
        if len(sample) == 0:
            continue

        try:
            generated_samples = eda(sample, num_aug=factor)

            for generated_sample in generated_samples:
                generated_items.append(generated_sample)
        except:
            error_count += 1

    random.shuffle(generated_items)

    with open(args.output_file, "wt") as file:
        count = (
            common.max_count(dataset_directory)
            - common.class_count(args.class_id, dataset_directory)
        ) * args.factor

        for i in range(0, count):
            file.write(f"{generated_items[i]}\n")

    print(f"Error Count: {error_count}")
    print(f"Selected count: {count}")


if __name__ == "__main__":
    main()
