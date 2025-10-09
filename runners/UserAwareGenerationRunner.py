import sys
sys.path.append('.')

import argparse
import json
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from models.user_aware_llm import UserAwareLLM
from data.datasets import Seq2SeqDataset
from prompts.post_process import load_post_process_function
from metrics.eval_metrics import LaMPEvaluation
from configs.user_aware_llm_config import UserAwareLLMConfig


# python runners/UserAwareGenerationRunner.py \
#     --use_user_aware_llm 1 \
#     --fusion_method attention \
#     --user_weight 0.3 \
#     --user_emb_path "20250529-092233.pt" \
#     --user_vocab_path "data/LaMP_1_time/dev/recency" \
#     --model_path "/data/yu12345/models/Qwen2-7B-Instruct" \
#     --task "LaMP_1_time" \
#     --input_file "path/to/input.json" \
#     --output_dir "path/to/output" \
#     --device "cuda:0"
    
class UserAwareGenerationRunner:
    
    @staticmethod
    def parse_args(parser):
        parser.add_argument("--use_user_aware_llm", type=int, default=1)
        parser.add_argument("--fusion_method", default="attention", 
                           choices=["concat", "attention", "add"])
        parser.add_argument("--user_weight", type=float, default=0.3)
        parser.add_argument("--user_emb_path", default="20250529-092233.pt")
        parser.add_argument("--user_vocab_path", default="data/LaMP_1_time/dev/recency")
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--input_path", default="data/LaMP_1_time/dev/recency/bge-base-en-v1.5_5/bge-reranker-base/")
        parser.add_argument("--source", default="20250607-171746_rerank_5")
        parser.add_argument("--begin_idx", type=int, default=0)
        parser.add_argument("--end_idx", type=int, default=1000000)
        parser.add_argument("--task", default="LaMP_1_time")
        parser.add_argument("--model_path", default="/data/yu12345/models/Qwen2-7B-Instruct")
        parser.add_argument("--output_dir", default="data/LaMP_1_time/dev/recency/output")
        
        return parser
    
    def __init__(self, opts):
        self.opts = opts
        self.config = UserAwareLLMConfig(
            use_user_aware_llm=opts.use_user_aware_llm,
            fusion_method=opts.fusion_method,
            user_weight=opts.user_weight,
            user_emb_path=opts.user_emb_path,
            user_vocab_path=opts.user_vocab_path,
            device=opts.device
        )
        
        self.setup_user_data()
        self.setup_llm()
        self.setup_dataset()
        
    def setup_user_data(self):
        """设置用户数据"""
        if self.config.use_user_aware_llm:
            import pickle
            with open(os.path.join(self.config.user_vocab_path, 'user2id.pkl'), 'rb') as file:
                self.user2id = pickle.load(file)
            self.config.user_emb_path = os.path.join(
                self.config.user_vocab_path, "user_emb", self.config.user_emb_path
            )
        else:
            self.user2id = {}
    
    def setup_llm(self):
        """设置LLM"""
        if self.config.use_user_aware_llm:
            self.llm = UserAwareLLM(
                model_path=self.opts.model_path,
                user_emb_path=self.config.user_emb_path,
                user2id=self.user2id,
                device=self.config.device,
                fusion_method=self.config.fusion_method,
                user_weight=self.config.user_weight
            )
            self.tokenizer = self.llm.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.opts.model_path)
            # 使用原始vLLM逻辑
    
    def setup_dataset(self):
        """设置数据集"""
        self.eval_dataset = Seq2SeqDataset(
            self.opts.input_file,
            task=self.opts.task,
            llm_tokenizer=self.tokenizer,
            max_length=self.opts.cutoff_len,
            begin_idx=self.opts.begin_idx,
            end_idx=self.opts.end_idx
        )
    
    def generate(self):
        """执行生成"""
        if not self.config.use_user_aware_llm:
            # 使用原始vLLM逻辑
            return self.generate_with_vllm()
        
        # 使用用户感知LLM生成
        model_preds = []
        
        for i in tqdm(range(len(self.eval_dataset))):
            data = self.eval_dataset[i]
            user_id = data['user_id']
            
            # 准备输入
            input_text = data['input']
            inputs = self.tokenizer(
                input_text, 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=self.opts.cutoff_len
            ).to(self.config.device)
            
            # 生成
            with torch.no_grad():
                generated_ids = self.llm.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    user_id=user_id,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(
                generated_ids[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            model_preds.append(generated_text)
        
        return model_preds
    
    def run(self):
        """运行完整的生成和评估流程"""
        # 生成
        model_preds = self.generate()
        
        # 后处理
        post_process_fun = load_post_process_function(self.opts.task)
        processed_preds = post_process_fun(model_preds)
        
        # 评估
        ground_truth = [self.eval_dataset[i]['output'] for i in range(len(self.eval_dataset))]
        eval_method = LaMPEvaluation(self.opts.task)
        pred_scores = eval_method.compute_metrics(processed_preds, ground_truth, avg=False)
        
        # 保存结果
        self.save_results(model_preds, processed_preds, pred_scores)
        
        return pred_scores
    
    def save_results(self, model_preds, processed_preds, pred_scores):
        """保存结果"""
        results = []
        for i in range(len(self.eval_dataset)):
            data = self.eval_dataset[i]
            result = {
                "user_id": data['user_id'],
                "input": data['input'],
                "raw_output": model_preds[i],
                "processed_output": processed_preds[i],
                "ground_truth": data['output']
            }
            # 添加评估分数
            for metric, scores in pred_scores.items():
                result[metric] = scores[i]
            results.append(result)
        
        # 保存到文件
        output_path = os.path.join(self.opts.output_dir, "user_aware_results.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    UserAwareGenerationRunner.parse_args(parser)
    # 添加其他必要的参数...
    
    opts = parser.parse_args()
    runner = UserAwareGenerationRunner(opts)
    scores = runner.run()
    print("Evaluation scores:", scores)