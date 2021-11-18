#!/bin/bash

function gen_fp32bmodel()
{
if [ $# != 3 ] ; then
    echo " USING: func batch_size net_name outdir"
    exit 1;
fi

batch_size=$1
net_name=$2
outdir=$3
echo "running gen with params batch_size: $batch_size, net_name: $net_name, outdir: $outdir"
python3 -m bmnetp --model 3dunet_bn.pt \
                  --net_name  $net_name\
                  --shapes="[$batch_size,4,224,224,160]" \
                  --dyn=false \
                  --outdir=$outdir \
                  --enable_profile=False \
                  --cmp=False \
                  --target=BM1684 \
                  --opt=1
}

gen_fp32bmodel 1 3dunet_b1 3dunet_b1
gen_fp32bmodel 2 3dunet_b2 3dunet_b2
gen_fp32bmodel 4 3dunet_b4 3dunet_b4
gen_fp32bmodel 8 3dunet_b8 3dunet_b8
gen_fp32bmodel 16 3dunet_b16 3dunet_b16