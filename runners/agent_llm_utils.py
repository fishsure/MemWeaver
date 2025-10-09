import re
from typing import List, Tuple
import json
from .agent_llm_prompts import get_task_prompt_template, get_task_toolcall_prompt_template

def agent_llm_summarize_summary(summary: str, records: List[str], agent_tokenizer, agent_llm, agent_sampling_params) -> str:
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
    message = [{"role": "user", "content": prompt}]
    chat_prompt = agent_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    outputs = agent_llm.generate([chat_prompt], agent_sampling_params)
    return outputs[0].outputs[0].text.strip()


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON content from LLM response text.
    Handles cases where JSON is wrapped in markdown code blocks or has explanatory text.
    """
    text = text.strip()
    
    # Try to find JSON in markdown code blocks
    import re
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        if extracted:  # Only return if we found something
            return extracted
    
    # Try to find JSON object directly (more robust pattern)
    # Look for content between { and } that might be JSON
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                potential_json = text[start_idx:i+1]
                # Basic validation: check if it looks like JSON
                if '"' in potential_json and (':' in potential_json or '[' in potential_json):
                    return potential_json.strip()
                start_idx = -1
    
    # Fallback: try simple regex pattern
    json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        if extracted:  # Only return if we found something
            return extracted
    
    # If no JSON found or extraction failed, return the original text
    return text

def extract_commands_from_text(text: str) -> str:
    """
    Extract tool call commands from LLM response text.
    Removes explanatory text and returns only the command lines.
    """
    text = text.strip()
    
    # Split into lines and filter for command lines
    lines = text.split('\n')
    command_lines = []
    
    for line in lines:
        line = line.strip()
        # Look for lines that start with tool names
        if any(line.startswith(tool) for tool in ['add_field(', 'delete_field(', 'update_field(', 'set_field(']):
            command_lines.append(line)
    
    extracted_commands = '\n'.join(command_lines)
    
    # If no commands found, return the original text
    if not extracted_commands.strip():
        return text
    
    return extracted_commands

def agent_llm_summarize_json(summary: str, records: List[str], agent_tokenizer, agent_llm, agent_sampling_params, task_type: str = None) -> dict:
    """
    summary: current summary (str), empty string for the first round
    records: the k profile records to summarize (list of str)
    agent_tokenizer: tokenizer instance
    agent_llm: LLM instance
    agent_sampling_params: sampling params for LLM
    task_type: string indicating the task type (e.g., 'citation', 'movie', ...)
    返回: dict, 包含summary（自然语言美化版）和summary_json（原始json内容）
    """
    prompt_template = get_task_prompt_template(task_type)
    records_md = "\n".join([f"- {rec}" for rec in records])
    prompt = prompt_template.format(summary=summary, n_records=len(records), records_md=records_md)
    message = [{"role": "user", "content": prompt}]
    chat_prompt = agent_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    
    # Check token length and truncate if necessary (max 8000 tokens to leave buffer for generation)
    tokens = agent_tokenizer(chat_prompt, return_tensors="pt")
    num_tokens = tokens["input_ids"].shape[1]
    max_tokens = 8000  # Leave some buffer for generation
    
    if num_tokens > max_tokens:
        print(f"[WARNING] Input too long ({num_tokens} tokens), truncating...")
        # Truncate the prompt by reducing records
        while num_tokens > max_tokens and len(records) > 1:
            records = records[:-1]  # Remove last record
            records_md = "\n".join([f"- {rec}" for rec in records])
            prompt = prompt_template.format(summary=summary, n_records=len(records), records_md=records_md)
            message = [{"role": "user", "content": prompt}]
            chat_prompt = agent_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            tokens = agent_tokenizer(chat_prompt, return_tensors="pt")
            num_tokens = tokens["input_ids"].shape[1]
        print(f"[INFO] Truncated to {len(records)} records, {num_tokens} tokens")
    
    outputs = agent_llm.generate([chat_prompt], agent_sampling_params)
    raw_output = outputs[0].outputs[0].text.strip()
    
    # Extract JSON from the response
    json_content = extract_json_from_text(raw_output)
    if json_content != raw_output:
        print(f"[DEBUG] JSON extracted from LLM response (length: {len(raw_output)} -> {len(json_content)})")
    # 新增：自然语言summary
    natural_summary = json_summary_to_natural_language(json_content)
    return {
        "summary": natural_summary,
        "summary_json": json_content
    }

def agent_llm_summarize_tool_call(summary: str, records: List[str], agent_tokenizer, agent_llm, agent_sampling_params, task_type: str = None) -> str:
    """
    Use tool_call style prompt to get incremental update commands for user features.
    summary: current summary (str), empty string for the first round
    records: the k profile records to summarize (list of str)
    agent_tokenizer: tokenizer instance
    agent_llm: LLM instance
    agent_sampling_params: sampling params for LLM
    task_type: string indicating the task type (e.g., 'citation', 'movie', ...)
    Returns: tool_call commands as string
    """
    prompt_template = get_task_toolcall_prompt_template(task_type)
    records_md = "\n".join([f"- {rec}" for rec in records])
    prompt = prompt_template.format(summary=summary, n_records=len(records), records_md=records_md)
    message = [{"role": "user", "content": prompt}]
    chat_prompt = agent_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    
    # Check token length and truncate if necessary (max 8000 tokens to leave buffer for generation)
    tokens = agent_tokenizer(chat_prompt, return_tensors="pt")
    num_tokens = tokens["input_ids"].shape[1]
    max_tokens = 8000  # Leave some buffer for generation
    
    if num_tokens > max_tokens:
        print(f"[WARNING] Input too long ({num_tokens} tokens), truncating...")
        # Truncate the prompt by reducing records
        while num_tokens > max_tokens and len(records) > 1:
            records = records[:-1]  # Remove last record
            records_md = "\n".join([f"- {rec}" for rec in records])
            prompt = prompt_template.format(summary=summary, n_records=len(records), records_md=records_md)
            message = [{"role": "user", "content": prompt}]
            chat_prompt = agent_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            tokens = agent_tokenizer(chat_prompt, return_tensors="pt")
            num_tokens = tokens["input_ids"].shape[1]
        print(f"[INFO] Truncated to {len(records)} records, {num_tokens} tokens")
    
    outputs = agent_llm.generate([chat_prompt], agent_sampling_params)
    raw_output = outputs[0].outputs[0].text.strip()
    
    # Extract commands from the response
    commands = extract_commands_from_text(raw_output)
    if commands != raw_output:
        print(f"[DEBUG] Commands extracted from LLM response (length: {len(raw_output)} -> {len(commands)})")
    return commands

def parse_tool_commands(commands: str) -> List[Tuple[str, List[str]]]:
    """
    Parse LLM tool command output into a list of (tool, [args...])
    Example output: [("add", ["User likes science fiction movies."]), ...]
    """
    result = []
    for line in commands.strip().splitlines():
        line = line.strip()
        if not line or not ("(" in line and line.endswith(")")):
            continue
        tool = line.split("(", 1)[0].strip()
        args_str = line[len(tool)+1:-1]  # remove tool( and )
        # For update, split by first comma
        if tool == "update":
            parts = [p.strip() for p in re.split(r",(?![^(]*\))", args_str, maxsplit=1)]
        else:
            parts = [args_str.strip()]
        result.append((tool, parts))
    return result

# Tool implementations

def add_to_summary(summary: List[str], record: str) -> List[str]:
    if record not in summary:
        summary.append(record)
    return summary

def delete_from_summary(summary: List[str], record: str) -> List[str]:
    summary = [s for s in summary if s != record]
    return summary

def update_summary(summary: List[str], old_record: str, new_record: str) -> List[str]:
    updated = False
    for i, s in enumerate(summary):
        if s == old_record:
            summary[i] = new_record
            updated = True
            break
    if not updated:
        summary.append(new_record)
    return summary

def keep_in_summary(summary: List[str], record: str) -> List[str]:
    # Optionally, ensure record is present
    if record not in summary:
        summary.append(record)
    return summary

# New functions for JSON-based preference operations
def add_preference_field(summary_dict: dict, field: str, value: str) -> dict:
    """
    Add a value to a preference field (e.g., add to main_disciplines list).
    """
    if field not in summary_dict:
        summary_dict[field] = []
    
    if isinstance(summary_dict[field], list):
        if value not in summary_dict[field]:
            summary_dict[field].append(value)
    else:
        # If field is not a list, convert it to a list
        summary_dict[field] = [str(summary_dict[field]), value]
    
    return summary_dict

def delete_preference_field(summary_dict: dict, field: str, value: str) -> dict:
    """
    Delete a value from a preference field.
    """
    if field in summary_dict:
        if isinstance(summary_dict[field], list):
            summary_dict[field] = [v for v in summary_dict[field] if v != value]
        elif str(summary_dict[field]) == value:
            del summary_dict[field]
    
    return summary_dict

def update_preference_field(summary_dict: dict, field: str, old_value: str, new_value: str) -> dict:
    """
    Update a value in a preference field.
    """
    if field in summary_dict:
        if isinstance(summary_dict[field], list):
            for i, v in enumerate(summary_dict[field]):
                if v == old_value:
                    summary_dict[field][i] = new_value
                    break
        elif str(summary_dict[field]) == old_value:
            summary_dict[field] = new_value
    
    return summary_dict

def set_preference_field(summary_dict: dict, field: str, value: str) -> dict:
    """
    Set a preference field to a specific value.
    """
    summary_dict[field] = value
    return summary_dict

def apply_tool_commands(commands: str, summary: List[str]) -> List[str]:
    """
    Apply parsed tool commands to the summary (list of str).
    """
    for tool, args in parse_tool_commands(commands):
        if tool == "add" and len(args) == 1:
            summary = add_to_summary(summary, args[0])
        elif tool == "delete" and len(args) == 1:
            summary = delete_from_summary(summary, args[0])
        elif tool == "update" and len(args) == 2:
            summary = update_summary(summary, args[0], args[1])
        elif tool == "keep" and len(args) == 1:
            summary = keep_in_summary(summary, args[0])
        # else: ignore malformed lines
    return summary

def apply_json_tool_commands(commands: str, summary_dict: dict) -> dict:
    """
    Apply parsed tool commands to the JSON summary dictionary.
    Supports operations like:
    - add_field(field_name, value)
    - delete_field(field_name, value) 
    - update_field(field_name, old_value, new_value)
    - set_field(field_name, value)
    """
    for tool, args in parse_tool_commands(commands):
        if tool == "add_field" and len(args) == 2:
            summary_dict = add_preference_field(summary_dict, args[0], args[1])
        elif tool == "delete_field" and len(args) == 2:
            summary_dict = delete_preference_field(summary_dict, args[0], args[1])
        elif tool == "update_field" and len(args) == 3:
            summary_dict = update_preference_field(summary_dict, args[0], args[1], args[2])
        elif tool == "set_field" and len(args) == 2:
            summary_dict = set_preference_field(summary_dict, args[0], args[1])
        # else: ignore malformed lines
    return summary_dict

def agent_llm_summarize_tool_call_end2end(
    summary: list,  # list of str, e.g., ["sci-fi", "romance"]
    records: List[str],
    agent_tokenizer=None,
    agent_llm=None,
    agent_sampling_params=None,
    task_type: str = None
) -> list:
    """
    End-to-end: First use agent_llm_summarize_json to get initial summary, then use tool_call commands for incremental updates.
    summary: current summary (list of str)
    records: the k profile records to summarize (list of str)
    agent_tokenizer: tokenizer instance
    agent_llm: LLM instance
    agent_sampling_params: sampling params for LLM
    task_type: string indicating the task type (e.g., 'citation', 'movie', ...)
    Returns: updated summary (list of str)
    """
    print(f"Original summary: {summary}")
    
    # Step 1: Use agent_llm_summarize_json to get initial JSON summary
    summary_str = json.dumps(summary, ensure_ascii=False) if summary else "{}"
    initial_json_summary = agent_llm_summarize_json(
        summary=summary_str,
        records=records,
        agent_tokenizer=agent_tokenizer,
        agent_llm=agent_llm,
        agent_sampling_params=agent_sampling_params,
        task_type=task_type
    )
    print(f"Initial JSON summary: {initial_json_summary}")
    
    # Step 2: Generate tool_call commands for incremental updates
    commands = agent_llm_summarize_tool_call(
        summary=initial_json_summary,
        records=records,
        agent_tokenizer=agent_tokenizer,
        agent_llm=agent_llm,
        agent_sampling_params=agent_sampling_params,
        task_type=task_type
    )
    print(f"Tool call commands: {commands}")
    
    # Step 3: Parse and apply commands to the initial JSON summary
    # Parse the initial JSON summary as a dictionary
    try:
        initial_summary_dict = json.loads(initial_json_summary) if initial_json_summary.strip() else {}
        if not isinstance(initial_summary_dict, dict):
            # If it's not a dict, create a simple dict
            initial_summary_dict = {"summary": initial_summary_dict}
    except json.JSONDecodeError as e:
        # If JSON parsing fails, create an empty dict
        print(f"[WARNING] Failed to parse JSON summary: {e}")
        print(f"[WARNING] Raw JSON content: {initial_json_summary[:200]}...")
        initial_summary_dict = {}
    
    # Apply JSON-based tool commands
    updated_summary_dict = apply_json_tool_commands(commands, initial_summary_dict)
    print(f"Final updated summary dict: {updated_summary_dict}")
    
    # Convert back to list format for compatibility
    updated_summary = []
    for key, value in updated_summary_dict.items():
        if isinstance(value, list):
            for item in value:
                updated_summary.append(f"{key}: {item}")
        else:
            updated_summary.append(f"{key}: {value}")
    
    return updated_summary


def json_summary_to_natural_language(summary_json: str) -> str:
    """
    将结构化的JSON summary转为自然语言英文总结，便于LLM或用户阅读。
    summary_json: JSON字符串或dict
    返回：自然语言英文summary
    """
    # 尝试解析JSON
    if isinstance(summary_json, str):
        try:
            summary_dict = json.loads(summary_json)
        except Exception:
            return summary_json  # 解析失败直接返回原文
    else:
        summary_dict = summary_json
    if not isinstance(summary_dict, dict):
        return str(summary_dict)

    # 主题映射，可根据实际任务类型扩展
    field_titles = {
        # citation task
        "main_disciplines": "Main Disciplines",
        "subfields": "Subfields",
        "common_methods": "Common Methods",
        "application_domains": "Application Domains",
        "frequent_keywords": "Frequent Keywords",
        "preferred_journals_conferences": "Preferred Journals/Conferences",
        "citation_style": "Citation Style",
        "preferred_citation_types": "Preferred Citation Types",
        "frequent_cited_authors": "Frequent Cited Authors",
        "recent_self_citation_ratio": "Recent Self-citation Ratio",
        "literature_recency_preference": "Literature Recency Preference",
        "open_access_preference": "Open Access Preference",
        "citation_languages": "Citation Languages",
        "interdisciplinary_tendency": "Interdisciplinary Tendency",
        # movie/product/news/email/tweet等可扩展，需要根据实际任务类型扩展
        # movie task
        "movie_genres": "Movie Genres",
        "product_categories": "Product Categories",
        "news_categories": "News Categories",
        "email_categories": "Email Categories",
        "tweet_topics": "Tweet Topics",
        # product task
        "product_categories": "Product Categories",
        "product_brands": "Product Brands",
        "product_colors": "Product Colors",
        "product_sizes": "Product Sizes",
        "product_prices": "Product Prices",
        "product_ratings": "Product Ratings",
        # news task
        "news_categories": "News Categories",
        "news_sources": "News Sources",
        "news_topics": "News Topics",
        "news_languages": "News Languages",
        "news_regions": "News Regions",
        "news_formats": "News Formats",
        # email task
        "email_categories": "Email Categories",
        "email_sources": "Email Sources",
        "email_topics": "Email Topics",
        "email_languages": "Email Languages",
        "email_regions": "Email Regions",
        "email_formats": "Email Formats",
        # tweet task
        "tweet_topics": "Tweet Topics",
        "tweet_languages": "Tweet Languages",
        "tweet_regions": "Tweet Regions",
        "tweet_formats": "Tweet Formats",
    }

    # 生成自然语言summary
    lines = ["**User Preference Summary**", "==========================\n"]
    for key, value in summary_dict.items():
        title = field_titles.get(key, key.replace('_', ' ').title())
        if isinstance(value, list):
            if value:
                lines.append(f"* **{title}**: {', '.join(map(str, value))}")
        elif isinstance(value, float):
            lines.append(f"* **{title}**: {value:.2f}")
        else:
            lines.append(f"* **{title}**: {value}")
    # 可根据需要添加结尾说明
    lines.append("\n_Note: This summary is generated based on the user's recent records and may evolve over time._")
    return '\n'.join(lines)

def agent_llm_summarize_tool_call_end2end_json(
    summary: list,  # list of str
    records: List[str],
    agent_tokenizer=None,
    agent_llm=None,
    agent_sampling_params=None,
    task_type: str = None
) -> str:
    """
    End-to-end: First use agent_llm_summarize_json to get initial summary, then use tool_call commands for incremental updates, returns the updated summary as a JSON string.
    summary: current summary (list of str)
    records: the k profile records to summarize (list of str)
    agent_tokenizer: tokenizer instance
    agent_llm: LLM instance
    agent_sampling_params: sampling params for LLM
    task_type: string indicating the task type (e.g., 'citation', 'movie', ...)
    Returns: updated summary (JSON string)
    """
    print(f"Input summary: {summary}")
    updated_summary = agent_llm_summarize_tool_call_end2end(
        summary=summary,
        records=records,
        agent_tokenizer=agent_tokenizer,
        agent_llm=agent_llm,
        agent_sampling_params=agent_sampling_params,
        task_type=task_type
    )
    print(f"Final updated summary: {updated_summary}")
    # 尝试将updated_summary转为dict
    try:
        # updated_summary是list[str]，尝试转为dict
        summary_dict = {}
        for item in updated_summary:
            if ":" in item:
                k, v = item.split(":", 1)
                k = k.strip()
                v = v.strip()
                # 尝试将v转为list
                if v.startswith("[") and v.endswith("]"):
                    try:
                        v = json.loads(v)
                    except Exception:
                        pass
                summary_dict[k] = v
    except Exception:
        summary_dict = {"summary": updated_summary}
    # 生成自然语言summary
    natural_summary = json_summary_to_natural_language(summary_dict)
    # 返回dict，包含原始JSON和自然语言summary
    return json.dumps({
        "summary": natural_summary,
        "summary_json": summary_dict
    }, ensure_ascii=False)

if __name__ == "__main__":
    # Mock tokenizer and LLM for demonstration
    class MockTokenizer:
        def apply_chat_template(self, message, tokenize=False, add_generation_prompt=True):
            # Just return the prompt string for testing
            return message[0]["content"]

    class MockLLM:
        def generate(self, prompts, agent_sampling_params):
            # Return a mock output depending on the prompt content
            class Output:
                def __init__(self, text):
                    self.outputs = [type('obj', (object,), {'text': text})()]
            prompt = prompts[0]
            if "add_field(" in prompt or "delete_field(" in prompt:
                # JSON tool_call style: add to main_disciplines and delete from subfields
                return [Output("Here are the tool calls to update the user's preference fields:\n\nadd_field(\"main_disciplines\", \"Computer Science\")\ndelete_field(\"subfields\", \"old_field\")\n\nNote: I've updated the fields based on the new records.")]
            elif "JSON format" in prompt:
                # JSON style: return a structured JSON for citation task with markdown wrapper
                return [Output("Here is the updated JSON:\n\n```json\n{\n  \"main_disciplines\": [\"Computer Science\"],\n  \"subfields\": [\"Machine Learning\"],\n  \"common_methods\": [\"Deep Learning\"]\n}\n```")]
            else:
                # summary style: return a markdown summary
                return [Output("- Likes sci-fi movies.\n- Recently interested in comedy.")]

    agent_tokenizer = MockTokenizer()
    agent_llm = MockLLM()
    # 为测试设置合理的采样参数
    class MockSamplingParams:
        def __init__(self):
            self.max_tokens = 1024
            self.temperature = 0
            self.seed = 42
    agent_sampling_params = MockSamplingParams()

    # Example data for citation task
    summary = ["Computer Science", "Machine Learning"]
    records = ["Published paper on deep learning in NeurIPS", "Cited papers on computer vision", "Reviewed papers on NLP"]
    summary_str = ", ".join(summary)

    print("=== agent_llm_summarize_summary ===")
    result1 = agent_llm_summarize_summary(summary_str, records, agent_tokenizer, agent_llm, agent_sampling_params)
    print(result1)

    print("\n=== agent_llm_summarize_json ===")
    result2 = agent_llm_summarize_json(summary_str, records, agent_tokenizer, agent_llm, agent_sampling_params, task_type="citation")
    print(result2)

    print("\n=== agent_llm_summarize_tool_call_end2end_json ===")
    # 1. 生成tool call指令
    commands = agent_llm_summarize_tool_call(
        summary=json.dumps(summary, ensure_ascii=False),
        records=records,
        agent_tokenizer=agent_tokenizer,
        agent_llm=agent_llm,
        agent_sampling_params=agent_sampling_params,
        task_type="citation"
    )
    print("LLM生成的tool call指令:\n", commands)

    # 2. 解析指令
    parsed_cmds = parse_tool_commands(commands)
    print("解析后的指令:", parsed_cmds)

    # 3. 检查指令是否合法（每条指令的tool和参数数量是否正确）
    legal = True
    for tool, args in parsed_cmds:
        if tool not in ("add_field", "delete_field", "update_field", "set_field"):
            legal = False
            print(f"非法指令: {tool}")
        if tool in ("add_field", "delete_field", "set_field") and len(args) != 2:
            legal = False
            print(f"参数数量错误: {tool} {args}")
        if tool == "update_field" and len(args) != 3:
            legal = False
            print(f"参数数量错误: {tool} {args}")
    print("指令是否合法:", legal)

    # 4. 测试JSON工具调用功能
    test_dict = {"main_disciplines": ["Computer Science"], "subfields": ["old_field"]}
    print("测试前的JSON dict:", test_dict)
    applied_dict = apply_json_tool_commands(commands, test_dict.copy())
    print("测试后的JSON dict:", applied_dict)

    # 5. 继续原有end2end_json测试
    result3 = agent_llm_summarize_tool_call_end2end_json(summary, records, agent_tokenizer, agent_llm, agent_sampling_params, task_type="citation")
    print("最终end2end_json结果:", result3) 