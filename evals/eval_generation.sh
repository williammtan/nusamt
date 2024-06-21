OUTPUT_DIR=${1}
TEST_PAIRS=${2:-"ban-en,en-ban"}
DATA_DIR=${3}

## Evaluation
for pair in ${TEST_PAIRS//,/ }; do
    src=$(echo ${pair} | cut -d "-" -f 1)
    tgt=$(echo ${pair} | cut -d "-" -f 2)
    TOK="13a"
    tgt_path=flores-eval/${src}${tgt}/test.${src}-${tgt}.${tgt}
    echo "--------------------Results for ${pair}-------------------------------------"
    
    output_path=${OUTPUT_DIR}/test-${src}-${tgt}
    sacrebleu -tok ${TOK} -w 2 ${tgt_path} < ${output_path} > ${output_path}.bleu
    cat ${output_path}.bleu
    # comet-score -s ${src_path} -t ${output_path} -r ${tgt_path} --batch_size 256 --model Unbabel/wmt22-comet-da --gpus 1 > ${output_path}.comet
    # comet-score -s ${src_path} -t ${output_path} --batch_size 256 --model Unbabel/wmt22-cometkiwi-da --gpus 1 > ${output_path}.cometkiwi
    # comet-score -s ${src_path} -t ${output_path} --batch_size 8 --model Unbabel/wmt23-cometkiwi-da-xxl --gpus 1 > ${output_path}.cometkiwi_10b
    # comet-score -s ${src_path} -t ${output_path} --batch_size 8 --model Unbabel/XCOMET-XXL --gpus 1 --to_json ${output_path}.xcomet.output.json > ${output_path}.xcomet_10b    
    # tail -n 1 ${output_path}.comet
done

for pair in ${TEST_PAIRS//,/ }; do
    src=$(echo ${pair} | cut -d "-" -f 1)
    tgt=$(echo ${pair} | cut -d "-" -f 2)
    echo "---------------------------${src}-${tgt}-------------------------------"
    output_path=${OUTPUT_DIR}/test-${src}-${tgt}
    cat ${output_path}.bleu
    # tail -n 1 ${output_path}.comet
    # tail -n 1 ${output_path}.cometkiwi
    # tail -n 1 ${output_path}.cometkiwi_10b
    # tail -n 2 ${output_path}.xcomet_10b
done
