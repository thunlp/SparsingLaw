set -e

# parameters
if [[ -z $dataset ]] ; then dataset="valid"; fi
# end

MASTER_PORT=8894
if [[ -z $GPUS_PER_NODE ]] ; then GPUS_PER_NODE=1; fi
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --master_port $MASTER_PORT"

OPT=""
OPT+=" --tokenizer-path ${tokenizer_path}"
OPT+=" --model-name ${model}"
OPT+=" --dataset-name ${dataset}"
OPT+=" --target-ppl-ratio ${target_ppl_ratio}"
OPT+=" --load-path ${load_path}"

CMD="torchrun ${DISTRIBUTED_ARGS} calc_threshold_for_each_layer.py ${OPT}"
echo ${CMD}
${CMD}
