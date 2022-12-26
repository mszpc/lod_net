#!/bin/bash
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash distribute_train.sh /tmp/huangxs-mask15/hccl_4p_0123_127.0.1.1.json /tmp/huangxs-mask15/save_checkpoint/mask_rcnn_temp.ckpt /tmp/huangxs-mask15/dataset/mscoco"
echo "It is better to use the absolute path."
echo "==============================================================================================================".

if [ $# != 3 ]
then
    echo "Usage: bash run_train.sh [RANK_TABLE_FILE] [PRETRAINED_PATH] [DATA_PATH]"
exit 1
fi
get_real_path()
{
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)
PATH3=$(get_real_path $3)

echo $PATH1
echo $PATH2
echo $PATH3

if [ ! -f $PATH1 ]
then 
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi 

if [ ! -f $PATH2 ]
then 
    echo "error: PRETRAINED_PATH=$PATH2 is not a file"
exit 1
fi

#ulimit -u unlimited
export HCCL_CONNECT_TIMEOUT=600
export DEVICE_NUM=4
export RANK_SIZE=4
export RANK_TABLE_FILE=$PATH1

#echo 3 > /proc/sys/vm/drop_caches

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $DEVICE_NUM`
gap=`expr $avg \- 1`

for((i=0; i<${DEVICE_NUM}; i++))
do
    start=`expr $i \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end

    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp ../*.yaml ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    taskset -c $cmdopt python train.py --do_train=True  --device_id=$i --rank_id=$i --run_distribute=True --device_num=$DEVICE_NUM \
    --pre_trained=$PATH2 --coco_root=$PATH3 &> log &
    cd ..
done
