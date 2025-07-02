# 自然语言处理大项目
221900448 周天远

## 关于预训练
第零步：安装所需python包
```conda env create -f scripts/env.yaml --yes```


第一步：测试所有组件
```
conda activate gpt-training
python scripts/test_components.py
```

第二步：开始训练
```./scripts/run_pretrain_deepspeed.sh```

第三步：评估训练好的模型
```
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_model.py \
    --checkpoint_path ./outputs/pretrain_20250605_002832/best_model/checkpoint-3500 \
    --generate_samples \
    --num_samples 5
```
