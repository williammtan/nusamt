OUTPUT_DIR=${1}
TEST_PAIRS=${2:-"ban-en,en-ban"}
DATA_DIR=${3}

## Evaluation
for pair in ${TEST_PAIRS//,/ }; do
    src=$(echo ${pair} | cut -d "-" -f 1)
    tgt=$(echo ${pair} | cut -d "-" -f 2)
    TOK="flores200"

    tgt_path=${DATA_DIR}/${src}${tgt}/test.${src}-${tgt}.${tgt}
    output_path=${OUTPUT_DIR}/test-${src}-${tgt}
    echo "${tgt_path} ${output_path}"

    sacrebleu -tok ${TOK} -w 2 ${tgt_path} < ${output_path} > ${output_path}.bleu
    cat ${output_path}.bleu
done

for pair in ${TEST_PAIRS//,/ }; do
    src=$(echo ${pair} | cut -d "-" -f 1)
    tgt=$(echo ${pair} | cut -d "-" -f 2)
    echo "---------------------------${src}-${tgt}-------------------------------"
    output_path=${OUTPUT_DIR}/test-${src}-${tgt}
    cat ${output_path}.bleu
done