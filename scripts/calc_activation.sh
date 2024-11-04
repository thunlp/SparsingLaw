model_path="/home/test/test06/lyq/checkpoints/0.8b_relu/480000"
input_file="input.txt"
prune_strategy="topk"
prune_arg="0.05"

OPT=""
OPT+=" --from-pretrained ${model_path}"
OPT+=" --input-file ${input_file}"
OPT+=" --prune-strategy ${prune_strategy}"
OPT+=" --prune-arg ${prune_arg}"

OPT+=" --output-image"

CMD="python3 calc_activation.py ${OPT}"

echo ${CMD}
${CMD}
