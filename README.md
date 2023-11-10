# SOFITDA

## Introduction

This project hosts the source code and dataset of the experiments specified in the paper "A Submodular Optimization Framework for Imbalanced Text Classification With Data Augmentation" (SOFITDA).

## Prerequisites

### Install Tools

To execute the experiments, you will need to install the following tools:

- Bash shell
- Java 1.8 or later
- Python 3.10 or later

### Install Python Libraries

Install the Python libraries required by the experiments by executing the following from the root directory of this project:

```
pip install -r python/requirements.txt
```

### Download Glove Embeddings

The CNN and RNN classifiers require Glove Embeddings. Do the following to download and deploy the Glove Embeddings:

- Download the zip file of Glove 6B from: https://nlp.stanford.edu/data/glove.6B.zip
- Unzip the zip file, and copy the `glove.6B.100d.txt` file to the `data/embeddings` folder in this project.

## Run Experiments

For a given dataset, generator and classifier, the experiments can be run executing the following from the root directory of this project:

```
cd python
./runner.sh -d <dataset> -c <classifier> -g <generator>
```

The following are the valid values for the dataset:
- amazon
- quora
- sst5
- trec
- tripadvisor
- yelp

The following are the valid values for the classifier:
- bow
- rnn
- cnn
- bert

The following are the valid values for the generator:
- eda
- gpt2

The output of running the experiments is a CSV file that is stored in a directory called `output`
that is created in the root directory of this project. The name of the CSV file has the following
format:

```
results_<dataset>_tall_m<classifier>_g<generator>.csv
```

The CSV file shows the performance metrics of the baselines and the various configurations of the
SOFITDA models.

## Help
If you have any issues about running the experiments, please send an email to eyor.alemayehu@gmail.com



