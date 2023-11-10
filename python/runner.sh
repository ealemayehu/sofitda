#!/bin/sh

set -e

CURRENT_DIR=$(pwd)
GPU=0
BASELINES='random top_ranked bottom_ranked no_aug'
ALPHA_VALUES='0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9'
CLUSTER_COUNTS='2 3 4 5 6 7 8 9 10 11 12'
MAX_VOCABULARY=100000

while getopts "d:c:g:a:k:" o; do
  case "${o}" in
    d)
      DATASET_NAME=$OPTARG
      ;;
    c)
      CLASSIFIER=$OPTARG
      ;;
    g)
      GENERATOR=$OPTARG
      ;;
    a)
      ALPHA_VALUES=$OPTARG
      ;;
    k)
      CLUSTER_COUNTS=$OPTARG
      ;;
    *)
      if [ "$OPTERR" = 1 ] && [ "${optspec:0:1}" != ":" ]; then
        echo "Unknown option --${OPTARG}" >&2
      fi
      ;;
  esac
done

if [ -z "${DATASET_NAME}" ]; then
    echo "Missing required DATASET_NAME option (-d)"
    exit 1
fi

if [ -z "${CLASSIFIER}" ]; then
    echo "Missing required CLASSIFIER option (-c)"
    exit 1
fi

if [ -z "${GENERATOR}" ]; then
    echo "Missing required GENERATOR option (-g)"
    exit 1
fi

CLASS_IDS=`python class_ids.py -d ${DATASET_NAME}`

echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
echo 'Configurations'
echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
echo "Generator: ${GENERATOR}"
echo "Dataset: ${DATASET_NAME}"
echo "Classifier: ${CLASSIFIER}"
echo "Class count: ${CLASS_IDS}"
echo "Alpha values: ${ALPHA_VALUES}"
echo "Cluster counts: ${CLUSTER_COUNTS}"

GENRANK=genrank_${GENERATOR}_${DATASET_NAME}

function generate {
    if [ "${GENERATOR}" == "eda" ]; then
        generate_eda_text
    else
        generate_gpt2_text
    fi
}

function generate_gpt2_text {
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    echo "Generating GPT-2 text"
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

    cd $CURRENT_DIR

    mkdir -p ../data/stage1/${GENRANK}/

    for class_id in ${CLASS_IDS}
    do
        cp ../data/stage3/${DATASET_NAME}/response_${class_id}_text_generation_dataset.txt \
           ../data/stage1/${GENRANK}/response_${class_id}_text_generation_dataset.txt


        cd $CURRENT_DIR/generator

        export PYTHONPATH='../'

        python main.py -i ../../data/stage1/${GENRANK}/response_${class_id}_text_generation_dataset.txt \
                    -o ../../data/stage1/${GENRANK}/generated_${class_id}.txt \
                    -f 10 \
                    -c ${class_id} \
                    -d ${DATASET_NAME}

        cd ../
    done
}

function generate_eda_text {
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    echo "Generating EDA text"
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

    cd $CURRENT_DIR

    mkdir -p ../data/stage1/${GENRANK}/

    for class_id in ${CLASS_IDS}
    do
        cp ../data/stage3/${DATASET_NAME}/response_${class_id}_text_generation_dataset.txt \
        ../data/stage1/${GENRANK}/response_${class_id}_text_generation_dataset.txt

        cd $CURRENT_DIR/eda

        export PYTHONPATH='../'
        
        python main.py -i ../../data/stage1/${GENRANK}/response_${class_id}_text_generation_dataset.txt \
                    -o ../../data/stage1/${GENRANK}/generated_${class_id}.txt \
                    -f 10 \
                    -c ${class_id} \
                    -d ${DATASET_NAME}

        cd ../
    done
}

function prepare_selection_dataset {
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    echo 'Preparing selection dataset'
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'


    cd $CURRENT_DIR
    cd ../java
    java -jar extractor.jar ${GENRANK}
}

function compute_kmeans {
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    echo 'Compute K-Means'
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

    cd $CURRENT_DIR

    mkdir -p ../output/${GENRANK}

    for class_id in ${CLASS_IDS}
    do
        for cluster_count in ${CLUSTER_COUNTS}
        do
            python kmeans.py --dataset_name ${DATASET_NAME} \
                             --generator ${GENERATOR} \
                             --cluster_count ${cluster_count} \
                             --class_id ${class_id}
        done
    done
}

function compute_scores {
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    echo 'Compute scores for generated items'
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

    cd $CURRENT_DIR
    rm -rf ../output/${GENRANK}_${CLASSIFIER}
    python classifier.py --dataset_name ${GENRANK} \
                         --output_directory ../output/${GENRANK}_${CLASSIFIER} \
                         --validation_dataset_type validation \
                         --testing_dataset_type testing \
                         --classifier ${CLASSIFIER} \
                         --max_vocabulary_size 100000
}

function execute_sofitda_selectors {
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    echo 'Execute SOFITDA selectors'
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

    cd $CURRENT_DIR

    for alpha in ${ALPHA_VALUES}
    do
        for class_id in ${CLASS_IDS}
        do
            for cluster_count in ${CLUSTER_COUNTS}
            do
                python sofitda_selector.py --dataset_name ${DATASET_NAME} \
                                            --classifier ${CLASSIFIER} \
                                            --class_id "${class_id}" \
                                            --cluster_count "${cluster_count}" \
                                            --alpha "${alpha}" \
                                            --generator "${GENERATOR}"
            done
        done          
    done
}

function execute_baseline_selectors {
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    echo 'Execute baseline selectors'
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

    cd $CURRENT_DIR
    python baseline_selector.py --dataset_name $DATASET_NAME \
                                --class_ids "${CLASS_IDS}" \
                                --classifier "${CLASSIFIER}" \
                                --generator "${GENERATOR}"
}

function evaluate_sofitda_selections {
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    echo 'Evaluate SOFITDA selections'
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

    cd $CURRENT_DIR

    for alpha in ${ALPHA_VALUES}
    do
        for cluster_count in ${CLUSTER_COUNTS}
        do
            ./evaluate_sofitda.sh -c "${CLASS_IDS}" \
                                  -m ${CLASSIFIER} \
                                  -k ${cluster_count} \
                                  -a ${alpha} \
                                  -d ${DATASET_NAME} \
                                  -g "${GENERATOR}"
        done
    done
}

function evaluate_baseline_selections {
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    echo 'Evaluate baseline selections'
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

    cd $CURRENT_DIR

    ./evaluate_baseline.sh -c "${CLASS_IDS}" \
                           -m ${CLASSIFIER} \
                           -b "${BASELINES}" \
                           -d ${DATASET_NAME} \
                           -g "${GENERATOR}"
}

function results {
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    echo 'Create results'
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

    python results.py -d ${DATASET_NAME} -m ${CLASSIFIER} -g ${GENERATOR}
}

function clean_up {
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    echo 'Clean up intermediate files'
    echo '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

    rm -rf ../data/stage1/aug*
    rm -rf ../data/stage1/genrank*

    rm -rf ../data/stage3/aug*
    rm -rf ../data/stage3/genrank*

    rm -rf ../output/aug*
    rm -rf ../output/genrank*
}

generate
prepare_selection_dataset
compute_kmeans
compute_scores
execute_sofitda_selectors
execute_baseline_selectors
evaluate_sofitda_selections
evaluate_baseline_selections
results
clean_up

echo "All Done !!!"
