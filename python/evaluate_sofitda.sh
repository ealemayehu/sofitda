set -e

MAX_EPOCH=5
TESTING_DATASET_TYPE=testing

while test $# -gt 0; do
    case "$1" in
        -c)
            CLASS_IDS=$2
            shift
            shift
            ;;

        -a)
            ALPHA=$2
            shift
            shift
            ;;

        -m)
            MODEL=$2
            shift
            shift
            ;;

        -k)
            CLUSTER_COUNT=$2
            shift
            shift
            ;;


        -d)
            DATASET_NAME=$2
            shift
            shift
            ;;

        -g)
            GENERATOR=$2
            shift
            shift
            ;;

        *)
            echo "Invalid option: $1 $2"
            exit 1
    esac
done

if [ -z "${CLASS_IDS}" ]; then
    echo "Error: required option -c that specifies the class IDs is not specified"
    exit 1
fi

if [ -z "${MODEL}" ]; then
    echo "Error: required option -m that specifies the model is not specified"
    exit 1
fi

if [ -z "${CLUSTER_COUNT}" ]; then
    echo "Error: required option -k that specifies the cluster count is not specified"
    exit 1
fi

if [ -z "${ALPHA}" ]; then
    echo "Error: required option -a that specifies the alpha is not specified"
    exit 1
fi

if [ -z "${DATASET_NAME}" ]; then
    echo "Error: required option -d that specifies the dataset is not specified"
    exit 1
fi

if [ -z "${GENERATOR}" ]; then
    echo "Error: required option -g that specifies the generator is not specified"
    exit 1
fi

CLASSIFIER=`echo ${MODEL} | awk '{print tolower($0)}'`
CURRENT_DIR=$(pwd)

directory=aug${DATASET_NAME}_sofitda_m${CLASSIFIER}_c${CLUSTER_COUNT}_a${ALPHA}_g${GENERATOR}_p${TESTING_DATASET_TYPE}

echo '*************************************************************************'
echo "Creating ${directory}"
echo '*************************************************************************'

cd $CURRENT_DIR

echo "Class IDs: ${CLASS_IDS}"
echo "Model: ${MODEL}"
echo "Cluster Count: ${CLUSTER_COUNT}"
echo "Alpha: ${ALPHA}"
echo "Dataset Name: ${DATASET_NAME}"
echo "Generator: ${GENERATOR}"
echo "Dataset type: ${DATASET_TYPE}"

mkdir -p ../data/stage1/${directory}
genrank_directory=../output/genrank_${GENERATOR}_${DATASET_NAME}_${CLASSIFIER}

for class_id in ${CLASS_IDS}
do
    sofitda_file=${genrank_directory}/testing_sofitda_k${CLUSTER_COUNT}_a${ALPHA}_c${class_id}_documents.txt

    if [ -f "${sofitda_file}" ]; then
        target_file=../data/stage1/${directory}/generated_${class_id}.txt
        echo "Adding generated file ${sofitda_file} to ${target_file}..."
        cp ${sofitda_file} ${target_file}
    else
        echo "Did not find file: ${sofitda_file}"
    fi
done

cd ../java
java -jar extractor.jar ${directory}

cd $CURRENT_DIR

output_directory=../output/${directory}

command_line="python classifier.py --dataset_name ${directory} --output_directory ${output_directory} --classifier ${CLASSIFIER} --testing_dataset_type ${TESTING_DATASET_TYPE} --max_vocabulary_size 100000"

echo "****************************************"
echo ${command_line}
echo "****************************************"

${command_line}