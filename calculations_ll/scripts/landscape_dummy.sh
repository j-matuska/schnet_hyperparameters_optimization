#!/bin/bash

source ~/.venv/bin/activate;

basic=`pwd`;
echo ${basic};

for i in {0..19}; # zaciatok aj koniec su zahrnute
  do
    python3 -m ip_explorer.landscape \
    --num-nodes 1 \
    --gpus-per-node 1 \
    --batch-size 100 \
    --loss-type 'DS' \
    --landscape-type 'lines' \
    --steps 25 \
    --distance 0.2 \
    --overwrite \
    --compute-landscape \
    --model-type "schnetDS" \
    --database-path "/home/matuska/xxx/" \
    --model-path "/home/matuska/xxx/" \
    --save-dir "/home/matuska/xxx/ip_explorer" \
    --additional-kwargs "cutoff:5.0" \
    --additional-datamodule-kwargs "cutoff:5.0 database_name:in_vivo0${split}_80val" && \ 
#    --no-compute-initial-losses
    mv ./lines=DS_d=0.20_s=25_schnetDS.npy ./${i}lines=DS_d=0.20_s=25_schnetDS.npy ;
done && \
python3 ${basic}/landscape.py;
done;


