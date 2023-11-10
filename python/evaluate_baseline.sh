set -e

while test $# -gt 0; do
    case "$1" in
        -c)
            CLASS_IDS=$2
            shift
            shift
            ;;

        -m)
            MODEL=$2
            shift
            shift
            ;;

        -d)
            DATASET_NAME=$2
            shift
            shift
            ;;

        -b) 
            BASELINES=$2
            shift
            shift
            ;;

        -g)
            GENERATOR=$2
            shift
            shift
            ;;

        *)
            echo "Invalid option: $1 $2 (arguments $# remaining)"
            exit 1
    esac
done

if [ -z "${CLASS_IDS}" ]; then
    echo "Error: required option -c that specifies the class IDs is not specified"
    exit 1
fi

if [ -z "${BASELINES}" ]; then
    echo "Error: required option -b that specifies the baselines is not specified"
    exit 1
fi

if [ -z "${MODEL}" ]; then
    echo "Error: required option -m that specifies the model is not specified"
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

for baseline in ${BASELINES}
do
    directory=aug${DATASET_NAME}_${baseline}_m${CLASSIFIER}_g${GENERATOR}_ptesting

    echo '*************************************************************************'
    echo "Creating ${directory}"
    echo '*************************************************************************'

    cd $CURRENT_DIR

    mkdir -p ../data/stage1/${directory}
    genrank_directory=../output/genrank_${GENERATOR}_${DATASET_NAME}_${CLASSIFIER}

    for class_id in ${CLASS_IDS}
    do
        baseline_file=''

        if [ "${baseline}" != "no_aug" ]; then
            input="testing_${baseline}_c${class_id}_documents.txt"
            baseline_file=${genrank_directory}/${input}
        fi

        if [ -f "${baseline_file}" ]; then
            echo "Adding generated file ${baseline_file} to training dataset..."
            cp ${baseline_file} ../data/stage1/${directory}/generated_${class_id}.txt
        fi
    done

    cd ../java
    java -jar extractor.jar ${directory}

    cd $CURRENT_DIR
    cd ../python

    output_directory=../output/${directory}

    command_line="python classifier.py --dataset_name ${directory} --output_directory ${output_directory} --classifier ${CLASSIFIER} --max_vocabulary_size 100000"

    echo "****************************************"
    echo ${command_line}
    echo "****************************************"

    ${command_line}  
done

