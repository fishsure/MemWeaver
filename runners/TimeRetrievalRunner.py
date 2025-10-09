import copy
import json
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import datetime

from models.retriever import RetrieverModel
from prompts.pre_process import load_get_corpus_fn, load_get_query_fn





class TimeAwareRetriever(nn.Module):
    """简化的时间感知检索模型：使用exp(-lambda * time_diff)"""
    
    def __init__(self, base_retriever, time_embedding_dim=16, device='cuda'):
        super().__init__()
        self.base_retriever = base_retriever
        self.device = device
        self.time_embedding_dim = time_embedding_dim
        
        # 动态检测BGE模型的输出维度
        with torch.no_grad():
            dummy_text = "test text for dimension detection"
            dummy_tokens = base_retriever.tokenizer(
                [dummy_text],
                padding=True,
                truncation=True,
                max_length=base_retriever.max_length,
                return_tensors='pt'
            ).to(device)
            dummy_emb = base_retriever.encode(dummy_tokens)
            self.bge_output_dim = dummy_emb.size(-1)
            print(f"Detected BGE output dimension: {self.bge_output_dim}")
        
        # 可学习的时间衰减参数
        self.lambda_param = nn.Parameter(torch.tensor(0.1))  # 时间衰减系数
        
    def encode_time(self, date_info):
        """将日期信息编码为时间嵌入（保留以备后用）"""
        if isinstance(date_info, int):
            # 年份格式 (如 2020)
            year = date_info
            time_idx = min(max(year - 2010, 0), 999)  # 2010-2109年范围
        else:
            # YYYY-MM-DD 格式
            try:
                date_obj = datetime.datetime.strptime(date_info, "%Y-%m-%d")
                year = date_obj.year
                time_idx = min(max(year - 2010, 0), 999)
            except:
                time_idx = 500  # 2020年附近
        
        return self.time_embedding(torch.LongTensor([time_idx]).to(self.device))
    
    def compute_time_diff(self, query_time, doc_time):
        """计算时间差（以年为单位）"""
        if isinstance(query_time, int) and isinstance(doc_time, int):
            # 年份格式
            time_diff = abs(query_time - doc_time)
        else:
            # 完整日期格式
            try:
                if isinstance(query_time, str):
                    query_date = datetime.datetime.strptime(query_time, "%Y-%m-%d")
                    query_year = query_date.year
                else:
                    query_year = query_time
                
                if isinstance(doc_time, str):
                    doc_date = datetime.datetime.strptime(doc_time, "%Y-%m-%d")
                    doc_year = doc_date.year
                else:
                    doc_year = doc_time
                
                time_diff = abs(query_year - doc_year)
            except:
                time_diff = 0
        
        return time_diff
    
    def compute_time_aware_scores(self, query_emb, corpus_emb, time_embeddings, query_time, doc_times):
        """计算时间感知分数：semantic_score * exp(-lambda * time_diff)"""
        batch_size = corpus_emb.size(0)
        
        # 1. 计算语义相似度分数
        semantic_scores = F.cosine_similarity(
            query_emb.unsqueeze(1), corpus_emb.unsqueeze(0), dim=2
        ).squeeze(0)  # [batch_size]
        
        # 2. 计算时间差并应用衰减
        time_decay_scores = []
        for i, doc_time in enumerate(doc_times):
            time_diff = self.compute_time_diff(query_time, doc_time)
            # 应用时间衰减：exp(-lambda * time_diff)
            decay_score = torch.exp(-self.lambda_param * time_diff)
            time_decay_scores.append(decay_score)
        
        # 转换为tensor
        time_decay_scores = torch.tensor(time_decay_scores, dtype=torch.float32).to(self.device)
        
        # 3. 最终分数：语义分数 × 时间衰减
        final_scores = semantic_scores * time_decay_scores
        
        return final_scores
    
    def compute_time_aware_scores_alternative(self, query_emb, corpus_emb, time_embeddings, query_time, doc_times):
        """替代方案：使用固定的lambda参数"""
        batch_size = corpus_emb.size(0)
        
        # 1. 计算语义相似度分数
        semantic_scores = F.cosine_similarity(
            query_emb.unsqueeze(1), corpus_emb.unsqueeze(0), dim=2
        ).squeeze(0)  # [batch_size]
        
        # 2. 计算时间差并应用衰减
        time_decay_scores = []
        fixed_lambda = 0.1  # 固定衰减参数
        
        for i, doc_time in enumerate(doc_times):
            time_diff = self.compute_time_diff(query_time, doc_time)
            # 应用时间衰减：exp(-lambda * time_diff)
            decay_score = torch.exp(-fixed_lambda * time_diff)
            time_decay_scores.append(decay_score)
        
        # 转换为tensor
        time_decay_scores = torch.tensor(time_decay_scores, dtype=torch.float32).to(self.device)
        
        # 3. 最终分数：语义分数 × 时间衰减
        final_scores = semantic_scores * time_decay_scores
        
        return final_scores


class TimeRetriever:

    @staticmethod
    def parse_args(parser):
        # BGE retriever related arguments
        parser.add_argument("--base_retriever_path",
                            default="/data/yu12345/models/bge-m3")
        parser.add_argument("--retriever_pooling", default="average")
        parser.add_argument("--retriever_normalize", type=int, default=1)
        
        # 时间感知检索相关参数
        parser.add_argument("--use_time_aware", type=int, default=1, help="是否使用时间感知检索")
        parser.add_argument("--time_embedding_dim", type=int, default=16, help="时间嵌入维度")
        parser.add_argument("--lambda_param", type=float, default=0.1, help="时间衰减系数")

        return parser

    def __init__(self, opts) -> None:
        self.task = opts.task
        self.get_query = load_get_query_fn(self.task)
        self.get_corpus = load_get_corpus_fn(self.task)
        self.use_date = opts.source.endswith('date')

        self.data_addr = opts.data_addr
        self.output_addr = opts.output_addr
        self.data_split = opts.data_split
        self.source = opts.source
        self.topk = opts.topk
        self.device = opts.device
        self.batch_size = opts.batch_size
        
        # 时间感知检索参数
        self.use_time_aware = getattr(opts, 'use_time_aware', 1)
        self.time_embedding_dim = getattr(opts, 'time_embedding_dim', 16)
        self.lambda_param = getattr(opts, 'lambda_param', 0.1)

        # Load user data
        self.load_user(opts)
        
        # Initialize BGE retriever
        self.retriever = RetrieverModel(
            ret_type='dense',
            model_path=opts.base_retriever_path,
            base_model_path=opts.base_retriever_path,
            user2id=self.user2id,
            user_emb_path=self.user_emb_path,
            batch_size=self.batch_size,
            device=self.device,
            max_length=opts.max_length,
            pooling=opts.retriever_pooling,
            normalize=opts.retriever_normalize).eval().to(self.device)
        
        # 如果启用时间感知检索，初始化时间感知检索器
        if self.use_time_aware:
            self.time_aware_retriever = TimeAwareRetriever(
                self.retriever,
                time_embedding_dim=self.time_embedding_dim,
                device=self.device
            ).to(self.device)
            print(f"Initialized time-aware retriever with lambda_param={self.lambda_param}")

        # Load dataset
        input_path = os.path.join(self.data_addr, opts.data_split, self.source,
                                  'rank_merge.json')

        self.dataset = json.load(open(input_path, 'r'))
        print("orig datasize:{}".format(len(self.dataset)))
        self.dataset = self.dataset[opts.begin_idx:opts.end_idx]

    def load_user(self, opts):
        """Load user vocabulary and embeddings"""
        vocab_addr = os.path.join(opts.data_addr, f"dev/{opts.source}")

        with open(os.path.join(vocab_addr, 'user_vocab.pkl'), 'rb') as file:
            self.user_vocab = pickle.load(file)

        with open(os.path.join(vocab_addr, 'user2id.pkl'), 'rb') as file:
            self.user2id = pickle.load(file)

        assert len(self.user_vocab) == len(self.user2id)

        self.user_emb_path = os.path.join(opts.data_addr,
                                          f"dev/{opts.source}/user_emb",
                                          "20241009-120906.pt")

    def run(self):
        """Run the memorizer to retrieve relevant profiles"""
        # Determine output directory and file name
        retriever_name = "bge-m3"
        if self.use_time_aware:
            retriever_name += "_time_aware"
        sub_dir = f"{retriever_name}_{self.topk}"
        file_name = f"time_base"

        results = []
        for data in tqdm(self.dataset):
            query, selected_profs = self.retrieve_topk(data['input'],
                                                       data['user_id'])
            result_item = {
                "input": data['input'],
                "query": query,
                "output": data['output'],
                "user_id": data['user_id'],
                "retrieval": selected_profs
            }
            results.append(result_item)

        output_addr = os.path.join(self.output_addr, self.data_split,
                                   self.source, sub_dir, 'retrieval')

        if not os.path.exists(output_addr):
            os.makedirs(output_addr)

        result_path = os.path.join(output_addr, f"{file_name}.json")
        print("save file to: {}".format(result_path))
        with open(result_path, 'w') as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

    def retrieve_topk(self, inp, user):
        """Retrieve top-k profiles for given input and user"""
        # Get current user's profile
        user_id = self.user2id[user]
        current_profile = self.user_vocab[user_id]['profile']
        
        query = self.get_query(inp)
        
        # Check if query is None and handle it
        if query is None:
            print(f"Warning: get_query returned None for input: {inp}")
            query = str(inp)  # Use input as fallback query
        
        cur_corpus = self.get_corpus(current_profile, self.use_date)
        cur_retrieved, cur_scores = self.retrieve_topk_one_user(
            cur_corpus, current_profile, query, user, self.topk)
        
        all_retrieved = []
        for data_idx, data in enumerate(cur_retrieved):
            cur_data = copy.deepcopy(data)
            if self.task.startswith('LaMP_3'):
                cur_data['rate'] = cur_data['score']
            cur_data['score'] = cur_scores[data_idx]
            all_retrieved.append(cur_data)
            
        return query, all_retrieved

    def retrieve_topk_one_user(self, corpus, profile, query, user, topk):
        """Retrieve top-k items for one user using enhanced time-aware retrieval"""
        if self.use_time_aware:
            return self._retrieve_with_time_awareness(corpus, profile, query, user, topk)
        else:
            return self._retrieve_basic(corpus, profile, query, user, topk)
    
    def _retrieve_basic(self, corpus, profile, query, user, topk):
        """基本的BGE检索方法"""
        # Ensure query is a string
        if not isinstance(query, str):
            query = str(query)
            print(f"Warning: Query converted to string: {query}")
        
        # Ensure corpus is a list of strings
        if not isinstance(corpus, list):
            print(f"Warning: Corpus is not a list: {type(corpus)}")
            corpus = [str(corpus)] if corpus is not None else [""]
        
        # Ensure profile is a list
        if not isinstance(profile, list):
            print(f"Warning: Profile is not a list: {type(profile)}")
            profile = [profile] if profile is not None else [{}]
        
        # Use BGE retrieval directly
        selected_profs, dense_scores = self.retriever.retrieve_topk_dense(
            corpus, profile, query, user, topk
        )
        
        return selected_profs, dense_scores
    
    def _retrieve_with_time_awareness(self, corpus, profile, query, user, topk):
        """使用时间感知的增强检索方法"""
        # Ensure query is a string
        if not isinstance(query, str):
            query = str(query)
            print(f"Warning: Query converted to string: {query}")
        
        # Ensure corpus is a list of strings
        if not isinstance(corpus, list):
            print(f"Warning: Corpus is not a list: {type(corpus)}")
            corpus = [str(corpus)] if corpus is not None else [""]
        
        # Ensure profile is a list
        if not isinstance(profile, list):
            print(f"Warning: Profile is not a list: {type(profile)}")
            profile = [profile] if profile is not None else [{}]
        
        # 获取query embedding
        query_tokens = self.retriever.tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=self.retriever.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        query_emb = self.retriever.encode(query_tokens)
        
        # 批量处理corpus
        all_scores = []
        all_profiles = []
        
        for batch_idx in range(0, len(corpus), self.batch_size):
            batch_corpus = corpus[batch_idx:batch_idx + self.batch_size]
            batch_profile = profile[batch_idx:batch_idx + self.batch_size]
            
            # 获取corpus embeddings
            corpus_tokens = self.retriever.tokenizer(
                batch_corpus,
                padding=True,
                truncation=True,
                max_length=self.retriever.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            corpus_emb = self.retriever.encode(corpus_tokens)
            
            # 获取文档时间信息
            profile_dates = []
            for prof in batch_profile:
                if 'date' in prof:
                    profile_dates.append(prof['date'])
                else:
                    profile_dates.append(2020)  # 默认年份
            
            # 获取查询时间（使用最新文档时间作为query时间）
            query_time = max(profile_dates) if profile_dates else 2020
            
            # 直接使用时间感知检索器计算融合分数
            combined_scores = self.time_aware_retriever.compute_time_aware_scores(
                query_emb, corpus_emb, profile_dates, query_time, profile_dates
            )
            
            all_scores.extend(combined_scores.cpu().tolist())
            all_profiles.extend(batch_profile)
        
        # 选择top-k
        scores_array = np.array(all_scores)
        topk_indices = np.argsort(scores_array)[::-1][:topk]
        
        selected_profs = [all_profiles[i] for i in topk_indices]
        top_scores = [all_scores[i] for i in topk_indices]
        
        return selected_profs, top_scores


class MemoryEfficientTimeRetriever(TimeRetriever):
    """内存效率优化的时间感知检索器，使用梯度检查点和其他优化技术"""
    
    def __init__(self, opts):
        super().__init__(opts)
        
        # 启用梯度检查点以节省显存
        if self.use_time_aware:
            self.time_aware_retriever = torch.utils.checkpoint.checkpoint_wrapper(
                self.time_aware_retriever
            )
    
    def _retrieve_with_time_awareness(self, corpus, profile, query, user, topk):
        """内存优化的时间感知检索方法"""
        # 确保数据类型正确
        if not isinstance(query, str):
            query = str(query)
        if not isinstance(corpus, list):
            corpus = [str(corpus)] if corpus is not None else [""]
        if not isinstance(profile, list):
            profile = [profile] if profile is not None else [{}]
        
        # 获取query embedding
        with torch.no_grad():  # 查询编码不需要梯度
            query_tokens = self.retriever.tokenizer(
                [query],
                padding=True,
                truncation=True,
                max_length=self.retriever.max_length,
                return_tensors='pt'
            ).to(self.device)
            query_emb = self.retriever.encode(query_tokens)
        
        # 使用更小的批处理大小来减少显存使用
        memory_efficient_batch_size = min(self.batch_size, 8)  # 限制批处理大小
        
        all_scores = []
        all_profiles = []
        
        for batch_idx in range(0, len(corpus), memory_efficient_batch_size):
            batch_corpus = corpus[batch_idx:batch_idx + memory_efficient_batch_size]
            batch_profile = profile[batch_idx:batch_idx + memory_efficient_batch_size]
            
            # 获取corpus embeddings
            with torch.no_grad():  # 语料库编码不需要梯度
                corpus_tokens = self.retriever.tokenizer(
                    batch_corpus,
                    padding=True,
                    truncation=True,
                    max_length=self.retriever.max_length,
                    return_tensors='pt'
                ).to(self.device)
                corpus_emb = self.retriever.encode(corpus_tokens)
            
            # 获取文档时间信息
            profile_dates = []
            for prof in batch_profile:
                if 'date' in prof:
                    profile_dates.append(prof['date'])
                else:
                    profile_dates.append(2020)  # 默认年份
            
            # 获取查询时间（使用最新文档时间作为query时间）
            query_time = max(profile_dates) if profile_dates else 2020
            
            # 使用梯度检查点计算分数
            combined_scores = self.time_aware_retriever.compute_time_aware_scores(
                query_emb, corpus_emb, profile_dates, query_time, profile_dates
            )
            
            # 立即转移到CPU并释放GPU显存
            all_scores.extend(combined_scores.detach().cpu().tolist())
            all_profiles.extend(batch_profile)
            
            # 清理GPU显存
            del corpus_tokens, corpus_emb, combined_scores
            torch.cuda.empty_cache()
        
        # 选择top-k
        scores_array = np.array(all_scores)
        topk_indices = np.argsort(scores_array)[::-1][:topk]
        
        selected_profs = [all_profiles[i] for i in topk_indices]
        top_scores = [all_scores[i] for i in topk_indices]
        
        return selected_profs, top_scores


# 简化的时间感知检索架构说明：
# 1. 使用简单的时间衰减公式：semantic_score * exp(-lambda * time_diff)
# 2. 其中 time_diff 是query时间和文档时间的差值（以年为单位）
# 3. lambda 是可学习的时间衰减系数
# 4. 自动检测BGE模型输出维度，支持不同版本的BGE模型
# 5. 参数配置建议：
#    - lambda_param: 0.05-0.2 (时间衰减系数，越大衰减越快)
#    - time_embedding_dim: 16-32 (时间嵌入维度)
# 6. 对于显存受限的环境，使用 MemoryEfficientTimeRetriever 替代 TimeRetriever
# 7. 设置较小的 batch_size (如 4-8)
# 8. 如果仍然显存不足，可以进一步减少 max_length 