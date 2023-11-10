from shared import log
import argparse
import gpt_2_simple as gpt2
import os
import shutil
import sys
import common

DATASET_DIRECTORY_FORMAT = "../../data/stage3/%s"


def main():
    parser = argparse.ArgumentParser(description="Executes the GPT-2 text generator")
    parser.add_argument(
        "-d", "--dataset-name", required=True, help="Specifies the dataset name"
    )
    parser.add_argument(
        "-i", "--input-file", required=True, help="Specifies the input file"
    )
    parser.add_argument(
        "-f", "--factor", required=True, type=int, help="Specifies the factor"
    )
    parser.add_argument(
        "-c", "--class-id", required=True, type=int, help="Specifies the class ID"
    )
    parser.add_argument(
        "-o", "--output-file", required=True, help="Specifies the output file"
    )

    args = parser.parse_args()

    existing_samples = common.load_samples(args.input_file, True)

    print("Training sample count: %d" % len(existing_samples))

    if os.path.isfile(args.output_file):
        generated_samples = common.load_samples(args.output_file, False)
        generated_sample_count = len(generated_samples)
        existing_samples.extend(generated_samples)
    else:
        generated_sample_count = 0

    dataset_directory = DATASET_DIRECTORY_FORMAT % args.dataset_name
    sample_count = (
        common.max_count(dataset_directory)
        - common.class_count(args.class_id, dataset_directory)
    ) * args.factor

    print(f"Sample count: {sample_count}")
    print(f"Initial generated sample count: {generated_sample_count}")
    print(f"Class ID: {args.class_id}")

    if generated_sample_count >= sample_count:
        log(f"Done generated {generated_sample_count} samples")
        return

    shutil.rmtree("checkpoint", ignore_errors=True)
    shutil.rmtree("samples", ignore_errors=True)

    model_name = "124M"

    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)

    sess = gpt2.start_tf_sess()
    gpt2.finetune(
        sess, args.input_file, model_name=model_name, steps=100
    )  # steps is max number of training steps

    print("Done with fine tunning")

    with open(args.output_file, "at") as output_file:
        while True:
            try:
                generated_text_list = gpt2.generate(
                    sess,
                    return_as_list=True,
                    length=(50 + len("[BEGIN] ") + len(" [END]\n")),
                    temperature=0.7,
                    prefix="[BEGIN]",
                    nsamples=200,
                    batch_size=200,
                )
                for generated_text in generated_text_list:
                    lines = generated_text.split("\n")

                    for line in lines:
                        sample = common.remove_delimiters(line)

                        if sample is None:
                            continue

                        if len(sample) == 0:
                            continue

                        generated_sample_count += 1
                        existing_samples.append(sample)

                        output_file.write(f"{sample}\n")

                        if generated_sample_count >= sample_count:
                            log(f"Done generated {generated_sample_count} samples")
                            return

                log(f"Generated {generated_sample_count} samples")

            except:
                print(sys.exc_info()[0])


if __name__ == "__main__":
    main()
