import argparse
import common

DATASET_DIRECTORY_FORMAT = "../data/stage3/%s"


def main():
    parser = argparse.ArgumentParser(description="Class count")
    parser.add_argument(
        "-d", "--dataset_name", required=True, help="Specifies the dataset name"
    )

    args = parser.parse_args()

    dataset_directory = DATASET_DIRECTORY_FORMAT % args.dataset_name
    class_count = common.number_of_classes(dataset_directory)
    class_ids = [str(i) for i in range(0, class_count)]
    print(" ".join(class_ids), end="")


if __name__ == "__main__":
    main()
