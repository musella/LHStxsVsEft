## niter=10
## 
## output=train_step1
## 
## ## ./train.py --train-file ZH/train_bsm1.hd5  --out-dir ${output}_bsm1 --optimize --cluster gcc49_long_5gb_smp4 --cluster-nodes 0,1,2,3,4      --niter $niter >& ${output}_bsm1.log &
## ## ./train.py --train-file ZH/train_bsm2.hd5  --out-dir ${output}_bsm2 --optimize --cluster gcc49_long_5gb_smp4 --cluster-nodes 5,6,7,8,9      --niter $niter >& ${output}_bsm2.log &
## ## ./train.py --train-file ZH/train_sm.hd5    --out-dir ${output}_sm   --optimize --cluster gcc49_long_5gb_smp4 --cluster-nodes 10,11,12,13,14 --niter $niter & #>& ${output}_sm.log   &
## 
## 
## wait


output=opt

./train.py --train-file ZH/train_bsm1.hd5  --out-dir ${output}_bsm1 --save-pickle 2>&1 | tee ${output}_bsm1.log &
./train.py --train-file ZH/train_bsm2.hd5  --out-dir ${output}_bsm2 --save-pickle 2>&1 | tee ${output}_bsm2.log &
./train.py --train-file ZH/train_sm.hd5    --out-dir ${output}_sm   --save-pickle 2>&1 | tee ${output}_sm.log   &

wait


