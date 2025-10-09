import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional

class MemoryAgent:
    """
    MemoryAgent: 用于多轮 profile 提炼与压缩的记忆增强代理类
    支持每次分组选取k个profile，通过LLM递归提炼，直到剩余<=k条
    """

    def __init__(
        self,
        llm_name: str,
        device: str = 'cuda:0',
        k: int = 5,
        max_new_tokens: int = 256
    ):
        self.llm_name = llm_name
        self.device = device
        self.k = k
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.model = AutoModelForCausalLM.from_pretrained(llm_name).to(device)
        print(f"[MemoryAgent] LLM loaded from: {llm_name}")

    def build_prompt(self, chunk: List[str]) -> str:
        """
        构造LLM的提示词，输入为一组profile
        """
        prompt = (
            "以下是用户的多条行为或偏好记录，请对其进行综合归纳，总结出高度凝练的主要特征或兴趣：\n"
        )
        for idx, item in enumerate(chunk):
            prompt += f"{idx+1}. {item}\n"
        prompt += "请用简明扼要的语言输出提炼结果："
        return prompt

    @torch.no_grad()
    def summarize_chunk(self, chunk: List[str]) -> str:
        """
        对单个chunk（k条profile）调用LLM提炼并输出summary
        """
        prompt = self.build_prompt(chunk)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        summary = full_text[len(prompt):].strip()
        if not summary:
            summary = full_text.strip().split('\n')[-1]
        return summary

    def compress_profile(
        self,
        profile: List[str],
        k: Optional[int] = None
    ) -> List[str]:
        """
        多轮提炼主逻辑：每轮将profile按k分组，用LLM总结，直到长度<=k
        """
        k = k or self.k
        current = profile
        round_idx = 0
        while len(current) > k:
            print(f"\n[MemoryAgent] Stage {round_idx+1}: profile size = {len(current)}")
            new_profiles = []
            for i in range(0, len(current), k):
                chunk = current[i:i+k]
                summary = self.summarize_chunk(chunk)
                print(f"  Summarizing chunk {i//k+1}: {chunk} → {summary}")
                new_profiles.append(summary)
            current = new_profiles
            round_idx += 1
        print(f"[MemoryAgent] Finished. Final summarized profile: {current}")
        return current

# 使用示例
if __name__ == "__main__":
    profile = [
        "喜欢二次元文化和日系动画。",
        "经常收听电子音乐和游戏原声。",
        "曾购买任天堂Switch及多款游戏。",
        "关注AI技术前沿，尝试用Stable Diffusion做绘画。",
        "喜欢逛B站，收藏鬼畜区视频。",
        "偶尔旅行拍照，发朋友圈分享。",
        "最近关注AI视频生成和虚拟偶像。",
        "参加过Comic-Con动漫展。"
    ]
    agent = MemoryAgent(llm_name="Qwen2-7B-Instruct", device="cuda:0", k=3)
    final_profile = agent.compress_profile(profile)
    print("最终多轮记忆提炼结果：", final_profile)
