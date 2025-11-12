python evaluate.py --is-test \
--eval-model-path ./ckpt/fb15k237/CMR_1/model_best.mdl \
--task FB15k237_ind \
--pretrained-model ./PLMs/bert-base-uncased \
--batch-size 1024 \
--mm \
--prefix 4 \
--knn_topk 32