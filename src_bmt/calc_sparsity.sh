set -e

source get_ckpt_list.sh
last_ckpt=(${ckpt_list[@]: -5})
checkpoint_list=""
for ckpt in "${last_ckpt[@]}"; do
    checkpoint_list+="${save_path}/${model}/${ckpt},"
done
# remove the last comma
checkpoint_list=${checkpoint_list:0:-1}


MASTER_PORT=8894
if [[ -z $GPUS_PER_NODE ]] ; then GPUS_PER_NODE=1; fi
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --master_port $MASTER_PORT"

OPT=""
OPT+=" --tokenizer-path ${tokenizer_path}"
OPT+=" --model-name ${model}"
OPT+=" --dataset-name ${dataset}"
OPT+=" --checkpoint-list ${checkpoint_list}"
OPT+=" --target-ppl-ratio ${target_ppl_ratio}"
result_file=/tmp/${BASHPID}.res
OPT+=" --result-file-name ${result_file}"

CMD="torchrun ${DISTRIBUTED_ARGS} calc_cett.py ${OPT}"
echo ${CMD}
${CMD}

# fetch the result cett
cett=$(cat ${result_file})

echo "The results while cett is ${cett}"
export prune_strategy='cett'
export cett_upper_bound=$cett
for checkpoint in ${ckpt_list[@]} ; do
    export load_path="${save_path}/${model}/${checkpoint}"
    echo "The results by the checkpoint of ${checkpoint} steps"
    bash run_inspect.sh
done