conda env create -f scripts/env.yaml

第一步：测试所有组件
conda activate gpt-training
cd /home/zhoutianyuan/hw/nlp/project
python scripts/test_components.py

第二步：选择训练方式
./scripts/run_pretrain_deepspeed.sh

第三步：评估训练好的模型
python scripts/evaluate_model.py \
    --checkpoint_path ./outputs/pretrain_XXXXXX/best_model \
    --generate_samples \
    --num_samples 5