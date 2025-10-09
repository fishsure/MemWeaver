import json
import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

class QueryAwareSummaryUpdater:
    """
    读取现有的结果文件，根据当前input对summary进行调整，生成query-aware的summary
    """
    
    def __init__(self, opts):
        self.input_file = opts.input_file
        self.output_file = opts.output_file
        self.llm_name = opts.llm_name
        self.batch_size = getattr(opts, "batch_size", 10)
        self.enable_token_check = getattr(opts, "enable_token_check", 1)
        self.reserve_tokens = getattr(opts, "reserve_tokens", 500)
        self.gpu_id = getattr(opts, "gpu_id", None)
        self.gpu_memory_utilization = getattr(opts, "gpu_memory_utilization", 0.85)
        
        # 设置GPU环境变量
        if self.gpu_id is not None:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            print(f"[QueryAwareSummaryUpdater] 设置使用GPU: {self.gpu_id}")
        
        # 初始化LLM
        print(f"[QueryAwareSummaryUpdater] 初始化LLM: {self.llm_name}")
        self.agent_tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        self.agent_tokenizer.padding_side = "left"
        if self.agent_tokenizer.pad_token_id is None:
            self.agent_tokenizer.pad_token = self.agent_tokenizer.eos_token
            self.agent_tokenizer.pad_token_id = self.agent_tokenizer.eos_token_id
        
        self.agent_llm = LLM(
            model=self.llm_name, 
            gpu_memory_utilization=self.gpu_memory_utilization, 
            max_model_len=8192
        )
        self.agent_sampling_params = SamplingParams(seed=42, temperature=0, best_of=1, max_tokens=800)
        
        print(f"[QueryAwareSummaryUpdater] LLM初始化完成")
    
    @staticmethod
    def parse_args(parser):
        parser.add_argument("--input_file", required=True, help="输入的结果文件路径")
        parser.add_argument("--output_file", required=True, help="输出的结果文件路径")
        parser.add_argument("--llm_name", default="Qwen/Qwen2.5-7B-Instruct", help="用于summary调整的LLM模型")
        parser.add_argument("--batch_size", type=int, default=10, help="批处理大小")
        parser.add_argument("--enable_token_check", type=int, default=1, help="是否启用token长度检查")
        parser.add_argument("--reserve_tokens", type=int, default=500, help="为生成保留的token数量")
        parser.add_argument("--gpu_id", type=int, default=None, help="指定使用的GPU ID (例如: 0, 1, 2)")
        parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="GPU内存使用率 (0.0-1.0)")
        return parser
    
    def check_and_truncate_prompt(self, prompt: str, max_model_len: int = 8192, reserve_tokens: int = 500, enable_check: bool = True) -> str:
        """
        检查prompt的token长度，如果超过限制则自动截断
        """
        if not enable_check:
            return prompt
        
        # 计算当前prompt的token数量
        tokens = self.agent_tokenizer(prompt, return_tensors="pt")
        current_tokens = tokens["input_ids"].shape[1]
        
        # 计算最大允许的输入token数量
        max_input_tokens = max_model_len - reserve_tokens
        
        if current_tokens <= max_input_tokens:
            return prompt
        
        # 如果超过限制，截断到最大允许长度
        truncated_tokens = self.agent_tokenizer.decode(
            tokens["input_ids"][0][:max_input_tokens], 
            skip_special_tokens=True
        )
        
        # 尝试在句子边界截断
        sentences = truncated_tokens.split('.')
        if len(sentences) > 1:
            truncated_tokens = '.'.join(sentences[:-1]) + '.'
        
        return truncated_tokens
    
    def generate_query_aware_summary(self, base_summary: str, current_input: str, user_id: str) -> str:
        """
        根据当前input对base summary进行调整，生成query-aware的summary
        
        Args:
            base_summary: 原始的summary
            current_input: 当前的用户输入
            user_id: 用户ID
            
        Returns:
            query_aware_summary: 调整后的summary
        """
        # 构建prompt
        prompt = f"""You are an expert at analyzing user preferences in the context of specific queries. 

**Task:** Given a user's general preference summary and their current query, provide a focused summary that highlights the most relevant aspects for this specific query.

**User ID:** {user_id}

**Current Query/Input:**
{current_input}

**User's General Preference Summary:**
{base_summary}

**Instructions:**
1. Analyze the current query to understand what the user is looking for
2. Identify which aspects of the user's general preferences are most relevant to this specific query
3. Create a focused summary that emphasizes these relevant preferences
4. Keep the summary concise but comprehensive
5. Use clear, structured language with bullet points or short paragraphs
6. Focus on preferences that would help understand the user's likely response to this query

**Output:** Provide a focused summary that highlights user preferences relevant to this specific query."""

        # 检查并截断prompt
        max_model_len = 8192
        prompt = self.check_and_truncate_prompt(prompt, max_model_len, self.reserve_tokens, self.enable_token_check)
        
        message = [{"role": "user", "content": prompt}]
        chat_prompt = self.agent_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        
        outputs = self.agent_llm.generate([chat_prompt], self.agent_sampling_params)
        return outputs[0].outputs[0].text.strip()
    
    def process_batch(self, batch_data):
        """
        批量处理数据，生成query-aware summary
        
        Args:
            batch_data: 包含input、summary等信息的批次数据
            
        Returns:
            updated_batch: 更新后的批次数据
        """
        # 准备批量prompt
        prompts = []
        for item in batch_data:
            base_summary = item.get("summary", "")
            current_input = item.get("input", "")
            user_id = item.get("user_id", "")
            
            prompt = f"""You are an expert at analyzing user preferences in the context of specific queries. 

**Task:** Given a user's general preference summary and their current query, provide a focused summary that highlights the most relevant aspects for this specific query.

**User ID:** {user_id}

**Current Query/Input:**
{current_input}

**User's General Preference Summary:**
{base_summary}

**Instructions:**
1. Analyze the current query to understand what the user is looking for
2. Identify which aspects of the user's general preferences are most relevant to this specific query
3. Create a focused summary that emphasizes these relevant preferences
4. Keep the summary concise but comprehensive
5. Use clear, structured language with bullet points or short paragraphs
6. Focus on preferences that would help understand the user's likely response to this query

**Output:** Provide a focused summary that highlights user preferences relevant to this specific query."""
            
            # 检查并截断prompt
            max_model_len = 8192
            prompt = self.check_and_truncate_prompt(prompt, max_model_len, self.reserve_tokens, self.enable_token_check)
            
            message = [{"role": "user", "content": prompt}]
            chat_prompt = self.agent_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompts.append(chat_prompt)
        
        # 批量生成
        outputs = self.agent_llm.generate(prompts, self.agent_sampling_params)
        
        # 更新数据
        updated_batch = []
        for i, item in enumerate(batch_data):
            updated_item = item.copy()
            updated_item["summary"] = outputs[i].outputs[0].text.strip()
            updated_batch.append(updated_item)
        
        return updated_batch
    
    def run(self):
        """
        主处理流程
        """
        print(f"[QueryAwareSummaryUpdater] 开始处理文件: {self.input_file}")
        
        # 读取输入文件
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"[QueryAwareSummaryUpdater] 读取到 {len(data)} 条记录")
        
        # 检查数据格式
        if not isinstance(data, list):
            raise ValueError("输入文件应该包含一个JSON数组")
        
        # 检查必要字段
        required_fields = ["input", "summary"]
        for i, item in enumerate(data):
            missing_fields = [field for field in required_fields if field not in item]
            if missing_fields:
                print(f"[Warning] 第 {i} 条记录缺少字段: {missing_fields}")
        
        # 批量处理数据
        updated_data = []
        for i in tqdm(range(0, len(data), self.batch_size), desc="处理批次"):
            batch = data[i:i + self.batch_size]
            
            # 过滤掉没有summary的记录
            valid_batch = [item for item in batch if item.get("summary")]
            if not valid_batch:
                # 如果没有有效的summary，直接复制原数据
                updated_data.extend(batch)
                continue
            
            try:
                updated_batch = self.process_batch(valid_batch)
                
                # 将更新后的数据放回原位置
                batch_idx = 0
                for j, item in enumerate(batch):
                    if item.get("summary"):
                        # 有summary的记录，使用更新后的数据
                        updated_data.append(updated_batch[batch_idx])
                        batch_idx += 1
                    else:
                        # 没有summary的记录，直接复制
                        updated_data.append(item)
                        
            except Exception as e:
                print(f"[Error] 处理批次 {i//self.batch_size + 1} 时出错: {e}")
                # 出错时直接复制原数据
                updated_data.extend(batch)
        
        # 保存结果
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, indent=4, ensure_ascii=False)
        
        print(f"[QueryAwareSummaryUpdater] 处理完成，结果保存到: {self.output_file}")
        
        # 清理LLM资源
        print("[QueryAwareSummaryUpdater] 清理LLM资源...")
        del self.agent_llm
        del self.agent_tokenizer
        del self.agent_sampling_params
        
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[QueryAwareSummaryUpdater] 资源清理完成")


def main():
    parser = argparse.ArgumentParser(description="Query-aware Summary Updater")
    parser = QueryAwareSummaryUpdater.parse_args(parser)
    opts = parser.parse_args()
    
    updater = QueryAwareSummaryUpdater(opts)
    updater.run()


if __name__ == "__main__":
    main()