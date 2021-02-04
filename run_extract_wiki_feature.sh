language=${1:-en_sample}
model_name=${2:-bert-base-multilingual-cased}
wiki_dir=${3:-data}
block_size=${4:--1}

python ./extract_wiki_feature.py \
  --model_type bert \
  --model_name_or_path $model_name \
  --language $language \
  --do_eval \
  --data_dir $wiki_dir \
  --per_gpu_eval_batch_size 16 \
  --output_dir output \
  --target all \
  --max_sample_size 10000 \
  --block_size $block_size 
