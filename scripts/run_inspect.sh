MASTER_PORT=8894
if [[ -z $GPUS_PER_NODE ]] ; then GPUS_PER_NODE=1; fi
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --master_port $MASTER_PORT"


if [[ -z $dataset ]] ; then echo 'no dataset'; exit; fi
if [[ -z $model ]] ; then echo 'no model'; exit; fi
if [[ -z $tokenizer_path ]] ; then echo 'no tokenizer path'; exit; fi
if [[ -z $prune_strategy ]] ; then echo 'no strategy; default to cett'; prune_strategy="cett"; fi

OPT=""
OPT+=" --tokenizer-path ${tokenizer_path}"
OPT+=" --model-name ${model}"
OPT+=" --dataset-name ${dataset}"

OPT+=" --prune-strategy ${prune_strategy}"

if [[ -n $load_path ]] ; then
    OPT+=" --load-path ${load_path}"
fi
if [[ -n $cett_upper_bound ]] ; then
    OPT+=" --cett-upper-bound ${cett_upper_bound}"
fi
if [[ -n $prune_arg ]] ; then
    OPT+=" --prune-arg ${prune_arg}"
fi
if [[ -n $effective_threshold ]] ; then
    OPT+=" --effective-threshold ${effective_threshold}"
fi

CMD="torchrun ${DISTRIBUTED_ARGS} inspect_sparsity.py ${OPT}"

echo ${CMD}
${CMD}