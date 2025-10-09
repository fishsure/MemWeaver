from typing import List, Dict, Tuple

def check_and_truncate_prompt(prompt: str, agent_tokenizer, max_model_len: int = 8192, reserve_tokens: int = 500, enable_check: bool = True) -> str:
    """
    检查prompt的token长度，如果超过限制则自动截断
    
    Args:
        prompt: 原始prompt文本
        agent_tokenizer: tokenizer实例
        max_model_len: 模型最大长度限制
        reserve_tokens: 为生成保留的token数量
        enable_check: 是否启用token长度检查
    
    Returns:
        截断后的prompt文本
    """
    if not enable_check:
        return prompt
    
    # 计算当前prompt的token数量
    tokens = agent_tokenizer(prompt, return_tensors="pt")
    current_tokens = tokens["input_ids"].shape[1]
    
    # 计算最大允许的输入token数量
    max_input_tokens = max_model_len - reserve_tokens
    
    if current_tokens <= max_input_tokens:
        # print(f"[Token Check] Prompt token数: {current_tokens}/{max_model_len}，无需截断")
        return prompt
    
    # print(f"[Token Check] Prompt token数: {current_tokens}/{max_model_len}，超过限制，开始截断...")
    
    # 如果超过限制，需要截断
    # 策略：保留指令部分，截断用户记录部分
    instruction_parts = [
        "You are an expert at summarizing user preferences. Please update the user's preference summary step by step based on their historical records.\n",
        "\n**Task:**\nUpdate the user preference summary in concise English, using markdown format. Only output the updated summary."
    ]
    
    # 计算指令部分的token数量
    instruction_text = "".join(instruction_parts)
    instruction_tokens = agent_tokenizer(instruction_text, return_tensors="pt")["input_ids"].shape[1]
    
    # 计算可用于用户记录的token数量
    available_tokens = max_input_tokens - instruction_tokens
    
    if available_tokens <= 0:
        # print(f"[Token Check] 警告：可用token数量不足，仅保留指令部分")
        return instruction_text
    
    # 查找用户记录部分
    if "**Current User Preference Summary:**" in prompt:
        # 有当前summary的情况
        summary_start = prompt.find("**Current User Preference Summary:**")
        summary_end = prompt.find("**New User Records")
        if summary_end == -1:
            summary_end = prompt.find("**Task:**")
        
        summary_text = prompt[summary_start:summary_end]
        summary_tokens = agent_tokenizer(summary_text, return_tensors="pt")["input_ids"].shape[1]
        
        # 重新计算可用于记录的token数量
        available_tokens = max_input_tokens - instruction_tokens - summary_tokens
        
        if available_tokens <= 0:
            # print(f"[Token Check] 警告：summary占用过多token，仅保留指令和summary")
            return instruction_parts[0] + summary_text + instruction_parts[1]
        
        # 截断用户记录部分
        records_start = prompt.find("**New User Records")
        records_end = prompt.find("**Task:**")
        records_text = prompt[records_start:records_end]
        
        # 逐步减少记录数量直到满足token限制
        lines = records_text.split('\n')
        truncated_lines = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = agent_tokenizer(line + '\n', return_tensors="pt")["input_ids"].shape[1]
            if current_tokens + line_tokens <= available_tokens:
                truncated_lines.append(line)
                current_tokens += line_tokens
            else:
                break
        
        truncated_records = '\n'.join(truncated_lines)
        # print(f"[Token Check] 截断后记录数: {len(truncated_lines)}，token数: {current_tokens}")
        
        return instruction_parts[0] + summary_text + truncated_records + instruction_parts[1]
    else:
        # 没有当前summary的情况，直接截断用户记录
        records_start = prompt.find("**New User Records")
        records_end = prompt.find("**Task:**")
        records_text = prompt[records_start:records_end]
        
        # 逐步减少记录数量直到满足token限制
        lines = records_text.split('\n')
        truncated_lines = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = agent_tokenizer(line + '\n', return_tensors="pt")["input_ids"].shape[1]
            if current_tokens + line_tokens <= available_tokens:
                truncated_lines.append(line)
                current_tokens += line_tokens
            else:
                break
        
        truncated_records = '\n'.join(truncated_lines)
        # print(f"[Token Check] 截断后记录数: {len(truncated_lines)}，token数: {current_tokens}")
        
        return instruction_parts[0] + truncated_records + instruction_parts[1]

def llm_summarize(summary: str, records: List[str], agent_tokenizer, agent_llm, agent_sampling_params) -> str:
    """
    summary: current summary (str), empty string for the first round
    records: the k profile records to summarize (list of str)
    agent_tokenizer: tokenizer instance
    agent_llm: LLM instance
    agent_sampling_params: sampling params for LLM
    """
    prompt = "You are an expert at summarizing user preferences. Please update the user's preference summary step by step based on their historical records.\n"
    if summary:
        prompt += f"\n**Current User Preference Summary:**\n{summary}\n"
    prompt += f"\n**New User Records ({len(records)}):**\n"
    for i, rec in enumerate(records):
        prompt += f"- {rec}\n"
    prompt += "\n**Task:**\nUpdate the user preference summary in concise English, using markdown format. Only output the updated summary."
    
    # 检查并截断prompt
    max_model_len = 8192
    prompt = check_and_truncate_prompt(prompt, agent_tokenizer, max_model_len, enable_check=True)
    
    message = [{"role": "user", "content": prompt}]
    chat_prompt = agent_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    outputs = agent_llm.generate([chat_prompt], agent_sampling_params)
    return outputs[0].outputs[0].text.strip()

def llm_batch_summarize(user_profiles: Dict[str, List], agent_tokenizer, agent_llm, agent_sampling_params, summary_k: int, enable_token_check: bool = True, reserve_tokens: int = 500) -> Dict[str, str]:
    """
    批量处理多个用户的profile summary
    
    Args:
        user_profiles: Dict[user_id, profile_list] - 用户ID到profile列表的映射
        agent_tokenizer: tokenizer实例
        agent_llm: LLM实例
        agent_sampling_params: LLM采样参数
        summary_k: 每次summary处理的记录数
        enable_token_check: 是否启用token长度检查
        reserve_tokens: 为生成保留的token数量
    
    Returns:
        Dict[user_id, summary] - 用户ID到summary的映射
    """
    user_summaries = {user_id: "" for user_id in user_profiles.keys()}
    
    # 收集所有需要处理的批次
    all_batches = []
    batch_to_user = {}  # 记录每个批次属于哪个用户
    
    for user_id, profile in user_profiles.items():
        for i in range(0, len(profile), summary_k):
            records = profile[i:i+summary_k]
            record_texts = [str(r) for r in records]
            batch_id = f"{user_id}_batch_{i//summary_k}"
            all_batches.append((batch_id, record_texts))
            batch_to_user[batch_id] = user_id
    
    print(f"[Batch Summary] 总共需要处理 {len(all_batches)} 个批次，涉及 {len(user_profiles)} 个用户")
    
    # 批量处理所有批次
    for batch_id, record_texts in all_batches:
        user_id = batch_to_user[batch_id]
        current_summary = user_summaries[user_id]
        
        # 构建prompt
        prompt = "You are an expert at summarizing user preferences. Please update the user's preference summary step by step based on their historical records.\n"
        if current_summary:
            prompt += f"\n**Current User Preference Summary:**\n{current_summary}\n"
        prompt += f"\n**New User Records ({len(record_texts)}):**\n"
        for i, rec in enumerate(record_texts):
            prompt += f"- {rec}\n"
        prompt += "\n**Task:**\nUpdate the user preference summary in concise English, using markdown format. Only output the updated summary."
        
        # 检查并截断prompt
        max_model_len = 8192
        prompt = check_and_truncate_prompt(prompt, agent_tokenizer, max_model_len, reserve_tokens, enable_token_check)
        
        message = [{"role": "user", "content": prompt}]
        chat_prompt = agent_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        
        # 生成summary
        outputs = agent_llm.generate([chat_prompt], agent_sampling_params)
        new_summary = outputs[0].outputs[0].text.strip()
        
        # 更新用户summary
        user_summaries[user_id] = new_summary
        
        # print(f"[Batch Summary] 完成用户 {user_id} 的第 {batch_id.split('_')[-1]} 个批次")
    
    return user_summaries

def llm_batch_summarize_parallel(user_profiles: Dict[str, List], agent_tokenizer, agent_llm, agent_sampling_params, summary_k: int, batch_size: int = 4, enable_token_check: bool = True, reserve_tokens: int = 500) -> Dict[str, str]:
    """
    并行批量处理多个用户的profile summary，使用vLLM的批量推理能力
    
    Args:
        user_profiles: Dict[user_id, profile_list] - 用户ID到profile列表的映射
        agent_tokenizer: tokenizer实例
        agent_llm: LLM实例
        agent_sampling_params: LLM采样参数
        summary_k: 每次summary处理的记录数
        batch_size: 并行处理的批次大小
        enable_token_check: 是否启用token长度检查
        reserve_tokens: 为生成保留的token数量
    
    Returns:
        Dict[user_id, summary] - 用户ID到summary的映射
    """
    user_summaries = {user_id: "" for user_id in user_profiles.keys()}
    
    # 收集所有需要处理的批次
    all_batches = []
    batch_to_user = {}  # 记录每个批次属于哪个用户
    batch_to_summary = {}  # 记录每个批次当前的summary状态
    
    for user_id, profile in user_profiles.items():
        for i in range(0, len(profile), summary_k):
            records = profile[i:i+summary_k]
            record_texts = [str(r) for r in records]
            batch_id = f"{user_id}_batch_{i//summary_k}"
            all_batches.append(batch_id)
            batch_to_user[batch_id] = user_id
            batch_to_summary[batch_id] = user_summaries[user_id]
    
    # print(f"[Parallel Batch Summary] 总共需要处理 {len(all_batches)} 个批次，涉及 {len(user_profiles)} 个用户")
    
    # 按批次大小分组处理
    for i in range(0, len(all_batches), batch_size):
        batch_group = all_batches[i:i+batch_size]
        
        # 准备批量prompt
        prompts = []
        batch_ids = []
        
        for batch_id in batch_group:
            user_id = batch_to_user[batch_id]
            current_summary = batch_to_summary[batch_id]
            
            # 获取该批次的记录
            batch_num = int(batch_id.split('_')[-1])
            profile = user_profiles[user_id]
            start_idx = batch_num * summary_k
            end_idx = min(start_idx + summary_k, len(profile))
            records = profile[start_idx:end_idx]
            record_texts = [str(r) for r in records]
            
            # 构建prompt
            prompt = "You are an expert at summarizing user preferences. Please update the user's preference summary step by step based on their historical records.\n"
            if current_summary:
                prompt += f"\n**Current User Preference Summary:**\n{current_summary}\n"
            prompt += f"\n**New User Records ({len(record_texts)}):**\n"
            for j, rec in enumerate(record_texts):
                prompt += f"- {rec}\n"
            prompt += "\n**Task:**\nUpdate the user preference summary in concise English, using markdown format. Only output the updated summary."
            
            # 检查并截断prompt
            max_model_len = 8192
            prompt = check_and_truncate_prompt(prompt, agent_tokenizer, max_model_len, reserve_tokens, enable_token_check)
            
            message = [{"role": "user", "content": prompt}]
            chat_prompt = agent_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            
            prompts.append(chat_prompt)
            batch_ids.append(batch_id)
        
        # 批量生成
        outputs = agent_llm.generate(prompts, agent_sampling_params)
        
        # 更新结果
        for batch_id, output in zip(batch_ids, outputs):
            user_id = batch_to_user[batch_id]
            new_summary = output.outputs[0].text.strip()
            user_summaries[user_id] = new_summary
            
            # 更新后续批次的summary状态
            batch_num = int(batch_id.split('_')[-1])
            for future_batch_id in batch_to_summary:
                if (batch_to_user[future_batch_id] == user_id and 
                    int(future_batch_id.split('_')[-1]) > batch_num):
                    batch_to_summary[future_batch_id] = new_summary
        
        print(f"[Parallel Batch Summary] 完成批次 {i//batch_size + 1}/{(len(all_batches) + batch_size - 1)//batch_size}")
    
    return user_summaries