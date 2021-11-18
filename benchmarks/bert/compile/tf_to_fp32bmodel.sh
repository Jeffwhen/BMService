#!/bin/bash

function compile_bert()
{
MAX_BATCH=$1
python3 -m bmnett --model=bert_b_n.pb \
          --target=BM1684 \
          --shapes="[$MAX_BATCH,384],[$MAX_BATCH,384],[$MAX_BATCH,384]"  \
          --input_names="input_ids,input_mask,segment_ids" \
          --descs="[0,int32,0,256],[1, int32, 0, 2], [2, int32, 0, 2]" \
          --outdir squad_fp32_b_$MAX_BATCH \
          --enable_profile=True \
          --cmp=False \
          --opt=2 \
          --v=4 2>&1 | tee squad_fp32.log
}

compile_bert 1
compile_bert 2
compile_bert 4
compile_bert 8
compile_bert 16


########################
## combine
########################
$(> temp)
for bs in 1 2 4 8 16
do
    # bm_model.bin
    echo "squad_fp32_b_$bs/compilation.bmodel " >> temp
done

models=$(cat temp)
echo $models
rm temp
bm_model.bin --combine $models -o outdir/bert_squad.bmodel

if [ $? -ne 0 ]; then
    echo "bmmodel error for combine"
    exit 1
else
    echo "bnmodel ok for combine"
fi