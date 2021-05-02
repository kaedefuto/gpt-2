# 事前学習の実行
python ./transformers/examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path=rinna/japanese-gpt2-medium \
    --train_file=train.txt \
    --validation_file=train.txt \
    --do_train \
    --do_eval \
    --num_train_epochs=3 \
    --save_steps=5000 \
    --save_total_limit=3 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --output_dir=output/ \
    --use_fast_tokenizer=False \
