import copy
import json
import os
import pickle
import torch
import datetime
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from models.retriever import RetrieverModel
from prompts.pre_process import load_get_corpus_fn, load_get_query_fn
from runners.summary import llm_summarize, llm_batch_summarize, llm_batch_summarize_parallel

class Memorizer:

    @staticmethod
    def parse_args(parser):
        # BGE retriever related arguments
        parser.add_argument("--base_retriever_path",
                            default="LLMs/bge-base-en-v1.5")
        parser.add_argument("--retriever_pooling", default="average")
        parser.add_argument("--retriever_normalize", type=int, default=1)
        parser.add_argument("--summary", type=int, default=1)
        parser.add_argument("--summary_llm_name", default="Qwen/Qwen2.5-7B-Instruct")
        parser.add_argument("--summary_k",type=int, default=50)
        parser.add_argument("--summary_batch_size", type=int, default=100, help="Batch size for parallel summary processing")
        parser.add_argument("--use_parallel_summary", type=int, default=1, help="Whether to use parallel batch processing for summary")
        parser.add_argument("--auto_cleanup_llm", type=int, default=1, help="Whether to automatically cleanup LLM after summary generation")
        parser.add_argument("--enable_token_check", type=int, default=1, help="Whether to enable token length checking and auto-truncation")
        parser.add_argument("--reserve_tokens", type=int, default=500, help="Number of tokens to reserve for generation")
        
        # 新增参数 for k-means clustering and time decay
        parser.add_argument("--n_clusters", type=int, default=10)
        parser.add_argument("--use_kmeans", type=int, default=1)
        parser.add_argument("--kmeans_use_embedding", type=int, default=1, help="Whether to use embedding for k-means clustering (1) or TF-IDF (0)")
        parser.add_argument("--kmeans_select_method", default="center", choices=["center", "relevance"], 
                          help="K-means cluster representative selection method: center (closest to centroid) or relevance (most relevant to query)")
        parser.add_argument("--use_recency", type=int, default=1)
        parser.add_argument("--time_decay_lambda", type=float, default=0.0)
        
        # 新增参数 for clustering-based summary
        parser.add_argument("--use_clustering_summary", type=int, default=0, help="Whether to use clustering-based hierarchical summary")
        parser.add_argument("--summary_clusters", type=int, default=5, help="Number of clusters for local summary")
        parser.add_argument("--summary_cluster_min_size", type=int, default=3, help="Minimum size for a cluster to be summarized")
        
        # 新增参数 for time-based clustering
        parser.add_argument("--use_time_clustering", type=int, default=0, help="Whether to use time-based clustering instead of k-means")
        parser.add_argument("--time_clustering_method", default="equal_split", choices=["equal_split", "time_period"], help="Time clustering method: equal_split (equal number of records) or time_period (equal time periods)")
        
        # 新增参数 for direct concatenation of local summaries
        parser.add_argument("--direct_concat_summary", type=int, default=0, help="Whether to directly concatenate local summaries as global summary instead of using LLM integration")

        return parser

    def __init__(self, opts) -> None:
        self.task = opts.task
        self.get_query = load_get_query_fn(self.task)
        self.get_corpus = load_get_corpus_fn(self.task)
        self.use_date = opts.source.endswith('date')
        self.llm_name = opts.llm_name

        self.data_addr = opts.data_addr
        self.output_addr = opts.output_addr
        self.data_split = opts.data_split
        self.source = opts.source
        self.topk = opts.topk
        self.device = opts.device
        self.batch_size = opts.batch_size

        self.summary = opts.summary
        self.summary_k = opts.summary_k
        self.summary_llm_name = opts.summary_llm_name
        self.summary_batch_size = getattr(opts, "summary_batch_size", 100)
        self.use_parallel_summary = getattr(opts, "use_parallel_summary", 1)
        self.auto_cleanup_llm = getattr(opts, "auto_cleanup_llm", 1)
        self.enable_token_check = getattr(opts, "enable_token_check", 1)
        self.reserve_tokens = getattr(opts, "reserve_tokens", 500)
        
        # 新增参数 for k-means clustering and time decay
        self.n_clusters = getattr(opts, "n_clusters", 10)
        
        # 内存管理相关
        self.llm_cleaned = False
        self.use_kmeans = getattr(opts, "use_kmeans", 1)
        self.kmeans_use_embedding = getattr(opts, "kmeans_use_embedding", 1)
        self.kmeans_select_method = getattr(opts, "kmeans_select_method", "center")
        self.use_recency = getattr(opts, "use_recency", 1)
        self.time_decay_lambda = getattr(opts, "time_decay_lambda", 0.0)
        
        # 新增参数 for clustering-based summary
        self.use_clustering_summary = getattr(opts, "use_clustering_summary", 0)
        self.summary_clusters = getattr(opts, "summary_clusters", 5)
        self.summary_cluster_min_size = getattr(opts, "summary_cluster_min_size", 3)
        
        # 新增参数 for time-based clustering
        self.use_time_clustering = getattr(opts, "use_time_clustering", 0)
        self.time_clustering_method = getattr(opts, "time_clustering_method", "equal_split")
        
        # 新增参数 for direct concatenation of local summaries
        self.direct_concat_summary = getattr(opts, "direct_concat_summary", 0)

        # Load user data
        self.load_user(opts)
        if self.summary:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
            self.agent_tokenizer = AutoTokenizer.from_pretrained(opts.summary_llm_name)
            self.agent_tokenizer.padding_side = "left"
            if self.agent_tokenizer.pad_token_id is None:
                self.agent_tokenizer.pad_token = self.agent_tokenizer.eos_token
                self.agent_tokenizer.pad_token_id = self.agent_tokenizer.eos_token_id
            
            
            self.agent_llm = LLM(model=opts.summary_llm_name, gpu_memory_utilization=0.85, max_model_len=8192)
            self.agent_sampling_params = SamplingParams(seed=42, temperature=0, best_of=1, max_tokens=500)
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

        # Load dataset
        input_path = os.path.join(self.data_addr, opts.data_split, self.source,
                                  'rank_merge.json')

        self.dataset = json.load(open(input_path, 'r'))
        print("orig datasize:{}".format(len(self.dataset)))
        self.dataset = self.dataset[opts.begin_idx:opts.end_idx]

    def cleanup_llm(self):
        """清理LLM占用的GPU显存"""
        if hasattr(self, 'agent_llm') and not self.llm_cleaned:
            print("[Memory Management] 释放LLM占用的GPU显存...")
            try:
                # 删除LLM相关对象
                if hasattr(self, 'agent_llm'):
                    del self.agent_llm
                if hasattr(self, 'agent_tokenizer'):
                    del self.agent_tokenizer
                if hasattr(self, 'agent_sampling_params'):
                    del self.agent_sampling_params
                
                # 清理Python垃圾回收
                import gc
                gc.collect()
                
                # 清理CUDA缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"[Memory Management] 当前GPU显存使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
                
                print("[Memory Management] LLM显存释放完成")
                self.llm_cleaned = True
            except Exception as e:
                print(f"[Memory Management] 清理LLM时出现错误: {e}")
                # 即使出错也标记为已清理，避免重复尝试
                self.llm_cleaned = True

    def cluster_user_profiles_by_time(self, profile_list):
        """
        基于时间信息对用户的历史行为进行聚类
        
        Args:
            profile_list: 用户的历史行为列表，每个元素包含'date'字段
            
        Returns:
            clusters: 聚类结果，每个元素是一个包含profile索引的列表
        """
        if len(profile_list) < self.summary_cluster_min_size:
            # 如果profile数量太少，直接返回单个聚类
            return [[i for i in range(len(profile_list))]]
        
        try:
            # 将profile和索引配对，便于后续处理
            profile_with_indices = [(i, profile) for i, profile in enumerate(profile_list)]
            
            # 按时间排序（从早到晚）
            def get_date_key(item):
                profile = item[1]
                if isinstance(profile['date'], int):
                    # For LaMP_1_time and LaMP_5_time, date is just a year
                    return datetime.date(profile['date'], 1, 1)
                else:
                    # For other tasks, date is in YYYY-MM-DD format
                    return datetime.datetime.strptime(profile['date'], "%Y-%m-%d").date()
            
            profile_with_indices.sort(key=get_date_key)
            
            # 根据聚类方法进行分组
            if self.time_clustering_method == "equal_split":
                # 等分方法：将排序后的数据等分为指定数量的聚类
                clusters = self._split_equal_size(profile_with_indices)
            elif self.time_clustering_method == "time_period":
                # 等时间段方法：按时间跨度等分为指定数量的时间段
                clusters = self._split_equal_time_period(profile_with_indices)
            else:
                # 默认使用等分方法
                clusters = self._split_equal_size(profile_with_indices)
            
            # 过滤掉太小的聚类
            valid_clusters = []
            for cluster_indices in clusters:
                if len(cluster_indices) >= self.summary_cluster_min_size:
                    valid_clusters.append(cluster_indices)
            
            # 将未分配到有效聚类的profile归入最近的聚类
            all_assigned = set()
            for cluster in valid_clusters:
                all_assigned.update(cluster)
            
            unassigned = [i for i in range(len(profile_list)) if i not in all_assigned]
            if unassigned and valid_clusters:
                # 将未分配的profile添加到第一个聚类
                for idx in unassigned:
                    valid_clusters[0].append(idx)
            
            return valid_clusters if valid_clusters else [[i for i in range(len(profile_list))]]
            
        except Exception as e:
            print(f"[Time Clustering] 时间聚类过程中出现错误: {e}")
            # 如果聚类失败，返回单个聚类
            return [[i for i in range(len(profile_list))]]

    def _split_equal_size(self, profile_with_indices):
        """
        等分方法：将排序后的数据等分为指定数量的聚类
        
        Args:
            profile_with_indices: 按时间排序的(索引, profile)列表
            
        Returns:
            clusters: 聚类结果
        """
        n_clusters = min(self.summary_clusters, len(profile_with_indices))
        if n_clusters <= 1:
            return [[item[0] for item in profile_with_indices]]
        
        clusters = []
        items_per_cluster = len(profile_with_indices) // n_clusters
        remainder = len(profile_with_indices) % n_clusters
        
        start_idx = 0
        for i in range(n_clusters):
            # 前remainder个聚类多分配一个item
            current_size = items_per_cluster + (1 if i < remainder else 0)
            end_idx = start_idx + current_size
            
            cluster_indices = [item[0] for item in profile_with_indices[start_idx:end_idx]]
            clusters.append(cluster_indices)
            
            start_idx = end_idx
        
        return clusters

    def _split_equal_time_period(self, profile_with_indices):
        """
        等时间段方法：按时间跨度等分为指定数量的时间段
        
        Args:
            profile_with_indices: 按时间排序的(索引, profile)列表
            
        Returns:
            clusters: 聚类结果
        """
        if len(profile_with_indices) < 2:
            return [[item[0] for item in profile_with_indices]]
        
        # 计算时间跨度
        def get_date(profile):
            if isinstance(profile['date'], int):
                return datetime.date(profile['date'], 1, 1)
            else:
                return datetime.datetime.strptime(profile['date'], "%Y-%m-%d").date()
        
        start_date = get_date(profile_with_indices[0][1])
        end_date = get_date(profile_with_indices[-1][1])
        total_days = (end_date - start_date).days
        
        n_clusters = min(self.summary_clusters, len(profile_with_indices))
        if n_clusters <= 1:
            return [[item[0] for item in profile_with_indices]]
        
        # 计算每个时间段的天数
        days_per_period = total_days / n_clusters
        
        clusters = [[] for _ in range(n_clusters)]
        
        for item_idx, profile in profile_with_indices:
            current_date = get_date(profile)
            days_from_start = (current_date - start_date).days
            
            # 确定属于哪个时间段
            period_idx = min(int(days_from_start / days_per_period), n_clusters - 1)
            clusters[period_idx].append(item_idx)
        
        # 过滤掉空的聚类
        return [cluster for cluster in clusters if cluster]

    def cluster_user_profiles(self, profile_list):
        """
        对用户的历史行为进行聚类（使用TF-IDF向量化或时间信息）
        
        Args:
            profile_list: 用户的历史行为列表
            
        Returns:
            clusters: 聚类结果，每个元素是一个包含profile索引的列表
        """
        # 根据配置选择聚类方式
        if self.use_time_clustering:
            print(f"[Clustering] 使用基于时间的聚类方式，方法: {self.time_clustering_method}")
            return self.cluster_user_profiles_by_time(profile_list)
        else:
            print(f"[Clustering] 使用K-means聚类方式")
            return self.cluster_user_profiles_by_kmeans(profile_list)

    def cluster_user_profiles_by_kmeans(self, profile_list):
        """
        使用K-means对用户的历史行为进行聚类（可选择使用embedding或TF-IDF向量化）
        
        Args:
            profile_list: 用户的历史行为列表
            
        Returns:
            clusters: 聚类结果，每个元素是一个包含profile索引的列表
        """
        if len(profile_list) < self.summary_cluster_min_size:
            # 如果profile数量太少，直接返回单个聚类
            return [[i for i in range(len(profile_list))]]
        
        try:
            # 将profile转换为文本
            profile_texts = [str(profile) for profile in profile_list]
            
            # 根据配置选择向量化方式
            if self.kmeans_use_embedding:
                print(f"[Clustering] 使用embedding进行K-means聚类，profile数量: {len(profile_texts)}")
                X = self._get_profile_embeddings(profile_texts)
            else:
                print(f"[Clustering] 使用TF-IDF进行K-means聚类，profile数量: {len(profile_texts)}")
                X = self._get_profile_tfidf(profile_texts)
            
            # 使用K-means聚类
            n_clusters = min(self.summary_clusters, len(profile_list))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # 组织聚类结果
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(i)
            
            # 过滤掉太小的聚类
            valid_clusters = []
            for cluster_indices in clusters.values():
                if len(cluster_indices) >= self.summary_cluster_min_size:
                    valid_clusters.append(cluster_indices)
            
            # 将未分配到有效聚类的profile归入最近的聚类
            all_assigned = set()
            for cluster in valid_clusters:
                all_assigned.update(cluster)
            
            unassigned = [i for i in range(len(profile_list)) if i not in all_assigned]
            if unassigned and valid_clusters:
                # 将未分配的profile添加到最近的聚类
                for idx in unassigned:
                    # 简单策略：添加到第一个聚类
                    valid_clusters[0].append(idx)
            
            print(f"[Clustering] 聚类完成，生成了 {len(valid_clusters)} 个有效聚类")
            return valid_clusters if valid_clusters else [[i for i in range(len(profile_list))]]
            
        except Exception as e:
            print(f"[Clustering] 聚类过程中出现错误: {e}")
            # 如果聚类失败，返回单个聚类
            return [[i for i in range(len(profile_list))]]
    
    def _get_profile_embeddings(self, profile_texts):
        """
        获取profile的embedding向量，使用BGE-M3模型
        
        Args:
            profile_texts: profile文本列表
            
        Returns:
            numpy数组: profile的embedding矩阵
        """
        profile_embeddings = []
        
        # 动态检测embedding维度
        if not hasattr(self, '_embedding_dim'):
            try:
                # 使用一个简单的测试文本来检测embedding维度
                test_text = "test text for dimension detection"
                test_tokens = self.retriever.tokenizer(
                    test_text,
                    padding=True,
                    truncation=True,
                    max_length=self.retriever.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    test_emb = self.retriever.encode(test_tokens)
                    if isinstance(test_emb, torch.Tensor):
                        self._embedding_dim = test_emb.size(-1)
                    else:
                        self._embedding_dim = test_emb.shape[-1]
                    print(f"[Clustering] 检测到BGE-M3 embedding维度: {self._embedding_dim}")
            except Exception as e:
                print(f"[Clustering] 检测embedding维度失败，使用默认值1024: {e}")
                self._embedding_dim = 1024  # BGE-M3的默认维度
        
        for i, text in enumerate(profile_texts):
            try:
                # 使用retriever的tokenizer和encode方法获取embedding
                tokens = self.retriever.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=self.retriever.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    emb = self.retriever.encode(tokens)
                    # 如果emb是tensor，转换为numpy数组
                    if isinstance(emb, torch.Tensor):
                        emb = emb.cpu().numpy()
                    profile_embeddings.append(emb.flatten())  # 展平为1D数组
                    
            except Exception as e:
                print(f"[Clustering] 获取profile {i} embedding时出错: {e}")
                # 如果获取embedding失败，使用零向量
                profile_embeddings.append(np.zeros(self._embedding_dim))
        
        # 转换为numpy数组
        embeddings_array = np.array(profile_embeddings)
        print(f"[Clustering] 成功获取 {len(profile_texts)} 个profile的embedding，维度: {embeddings_array.shape}")
        return embeddings_array
    
    def _get_profile_tfidf(self, profile_texts):
        """
        获取profile的TF-IDF向量
        
        Args:
            profile_texts: profile文本列表
            
        Returns:
            scipy稀疏矩阵: profile的TF-IDF矩阵
        """
        # 使用TF-IDF进行向量化
        tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,  # 限制特征数量以提高效率
            min_df=1,  # 最小文档频率
            max_df=0.95,  # 最大文档频率，过滤掉过于常见的词
            stop_words='english',  # 使用英文停用词
            ngram_range=(1, 2),  # 使用1-gram和2-gram
            lowercase=True,
            strip_accents='unicode'
        )
        
        # 训练TF-IDF向量化器并转换文本
        return tfidf_vectorizer.fit_transform(profile_texts)

    def generate_local_summary(self, profile_cluster, cluster_id):
        """
        为单个聚类生成local summary
        
        Args:
            profile_cluster: 聚类中的profile列表
            cluster_id: 聚类ID
            
        Returns:
            local_summary: 该聚类的local summary
        """
        if not profile_cluster:
            return ""
        
        # 将profile转换为文本
        profile_texts = [str(profile) for profile in profile_cluster]
        
        # 构建prompt
        prompt = f"You are an expert at analyzing user behavior patterns. Please analyze the following cluster of user activities and provide a concise summary of the user's preferences and patterns in this cluster.\n\n"
        prompt += f"**Cluster {cluster_id + 1} Activities ({len(profile_texts)} records):**\n"
        for i, text in enumerate(profile_texts):
            prompt += f"{i+1}. {text}\n"
        prompt += "\n**Task:**\nProvide a concise summary of the user's preferences and behavior patterns in this cluster. Focus on key themes, preferences, and patterns. Use clear, structured language."
        
        # 检查并截断prompt
        max_model_len = 8192
        prompt = self.check_and_truncate_prompt(prompt, max_model_len, self.reserve_tokens, self.enable_token_check)
        
        message = [{"role": "user", "content": prompt}]
        chat_prompt = self.agent_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        
        outputs = self.agent_llm.generate([chat_prompt], self.agent_sampling_params)
        return outputs[0].outputs[0].text.strip()

    def generate_global_summary(self, local_summaries, user_id):
        """
        将所有local summary整合成global summary
        
        Args:
            local_summaries: local summary列表
            user_id: 用户ID
            
        Returns:
            global_summary: 整合后的global summary
        """
        if not local_summaries:
            return ""
        
        if len(local_summaries) == 1:
            # 如果只有一个local summary，直接返回
            return local_summaries[0]
        
        # 如果启用直接拼接模式，直接拼接所有local summary
        if self.direct_concat_summary:
            print(f"[Global Summary] 使用直接拼接模式为用户 {user_id} 生成global summary")
            global_summary = ""
            for i, summary in enumerate(local_summaries):
                if summary.strip():  # 只添加非空的summary
                    global_summary += f"**Cluster {i+1} Summary:**\n{summary.strip()}\n\n"
            return global_summary.strip()
        
        # 使用LLM整合模式
        print(f"[Global Summary] 使用LLM整合模式为用户 {user_id} 生成global summary")
        # 构建prompt - 强调简洁性
        prompt = f"You are an expert at creating concise user preference summaries. Please synthesize the following cluster summaries into a brief, focused global summary.\n\n"
        prompt += f"**User ID:** {user_id}\n"
        prompt += f"**Cluster Summaries ({len(local_summaries)} clusters):**\n"
        for i, summary in enumerate(local_summaries):
            prompt += f"Cluster {i+1}: {summary}\n"
        prompt += "\n**Task:**\nCreate a concise global summary (max 300 words) that captures the user's key preferences and behavior patterns. Focus on the most important themes and avoid redundancy. Use bullet points or short paragraphs for clarity. Be brief but comprehensive."
        
        # 检查并截断prompt
        max_model_len = 8192
        prompt = self.check_and_truncate_prompt(prompt, max_model_len, self.reserve_tokens, self.enable_token_check)
        
        message = [{"role": "user", "content": prompt}]
        chat_prompt = self.agent_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        
        outputs = self.agent_llm.generate([chat_prompt], self.agent_sampling_params)
        return outputs[0].outputs[0].text.strip()

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

    def generate_clustering_summary(self, profile_list, user_id):
        """
        使用聚类方法生成分层summary
        
        Args:
            profile_list: 用户的历史行为列表
            user_id: 用户ID
            
        Returns:
            global_summary: 最终的global summary
        """
        print(f"[Clustering Summary] 开始为用户 {user_id} 生成聚类summary，profile数量: {len(profile_list)}")
        
        # 1. 对profile进行聚类
        clusters = self.cluster_user_profiles(profile_list)
        print(f"[Clustering Summary] 用户 {user_id} 聚类完成，共 {len(clusters)} 个聚类")
        
        # 2. 为每个聚类生成local summary
        local_summaries = []
        for cluster_id, cluster_indices in enumerate(clusters):
            cluster_profiles = [profile_list[i] for i in cluster_indices]
            print(f"[Clustering Summary] 为用户 {user_id} 生成聚类 {cluster_id + 1} 的local summary，包含 {len(cluster_profiles)} 个profile")
            
            local_summary = self.generate_local_summary(cluster_profiles, cluster_id)
            local_summaries.append(local_summary)
            print(f"[Clustering Summary] 聚类 {cluster_id + 1} local summary 完成")
        
        # 3. 整合所有local summary为global summary
        print(f"[Clustering Summary] 为用户 {user_id} 整合 {len(local_summaries)} 个local summary为global summary")
        global_summary = self.generate_global_summary(local_summaries, user_id)
        
        print(f"[Clustering Summary] 用户 {user_id} 的聚类summary生成完成")
        return global_summary

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

        # self.user_emb_name = '.'.join(
        #     os.path.basename(self.user_emb_path).split('.')[:-1])
        # self.user_embedding = torch.load(self.user_emb_path).to(self.device)

        # assert self.user_embedding.shape[0] == len(self.user_vocab)

    def run(self):
        """Run the memorizer to retrieve relevant profiles"""
        # Determine output directory and file name
        user_summaries = {}
        if self.summary:
            # 收集所有需要summary的用户profile
            user_profiles = {}
            unique_user_ids = set()
            
            for data in self.dataset:
                user_id = data['user_id']
                if user_id not in unique_user_ids:
                    unique_user_ids.add(user_id)
                    user_idx = self.user2id[user_id]
                    profile = self.user_vocab[user_idx]['profile']
                    user_profiles[user_id] = profile
            
            print(f"[Batch Summary] 开始批量处理 {len(user_profiles)} 个用户的profile summary")
            
            # 根据配置选择summary方式
            if self.use_clustering_summary:
                print(f"[Clustering Summary] 使用聚类分层summary方式")
                for user_id, profile in user_profiles.items():
                    print(f"[Clustering Summary] 处理用户 {user_id}")
                    summary = self.generate_clustering_summary(profile, user_id)
                    user_summaries[user_id] = summary
            else:
                # 使用原有的批量处理方式
                if self.use_parallel_summary:
                    print(f"[Batch Summary] 使用并行批量处理，批次大小: {self.summary_batch_size}")
                    user_summaries = llm_batch_summarize_parallel(
                        user_profiles, 
                        self.agent_tokenizer, 
                        self.agent_llm, 
                        self.agent_sampling_params, 
                        self.summary_k,
                        self.summary_batch_size,
                        self.enable_token_check,
                        self.reserve_tokens
                    )
                else:
                    print(f"[Batch Summary] 使用顺序批量处理")
                    user_summaries = llm_batch_summarize(
                        user_profiles, 
                        self.agent_tokenizer, 
                        self.agent_llm, 
                        self.agent_sampling_params, 
                        self.summary_k,
                        self.enable_token_check,
                        self.reserve_tokens
                    )
            
            print(f"[Batch Summary] 完成所有用户的profile summary生成")
        
        # 释放LLM占用的GPU显存
        if self.summary and self.auto_cleanup_llm:
            self.cleanup_llm()
        
        #LLM此时可以关闭了，因为已经生成了summary
        
        retriever_name = "bge-m3"
        sub_dir = f"{retriever_name}_{self.topk}"
        if self.summary:
            file_name = f"mem_base_summaryk_{self.summary_k}"
        else:
            file_name = f"mem_base"
        
        # 添加新参数到文件名
        if self.summary:
            file_name += f'_parallel-{self.use_parallel_summary}_batchsize-{self.summary_batch_size}'
            if self.enable_token_check:
                file_name += f'_tokencheck-{self.enable_token_check}_reserve-{self.reserve_tokens}'
            if self.use_clustering_summary:
                file_name += f'_clustering-{self.use_clustering_summary}_summaryclusters-{self.summary_clusters}_minsize-{self.summary_cluster_min_size}'
                if self.use_time_clustering:
                    file_name += f'_timeclustering-{self.use_time_clustering}_method-{self.time_clustering_method}'
                if self.direct_concat_summary:
                    file_name += f'_directconcat-{self.direct_concat_summary}'
        file_name += f'_kmeans-{self.use_kmeans}_method-{self.kmeans_select_method}_clusters-{self.n_clusters}_recency-{self.use_recency}_decay-{self.time_decay_lambda}'


        results = []
        for data in tqdm(self.dataset):
            query, selected_profs = self.retrieve_topk(data['input'],
                                                       data['user_id'],
                                                       user_summaries if self.summary else None)
            result_item = {
                "input": data['input'],
                "query": query,
                "output": data['output'],
                "user_id": data['user_id'],
                "retrieval": selected_profs
            }
            
            # 如果使用了summary，添加当前用户的summary
            if self.summary and user_summaries and data['user_id'] in user_summaries:
                result_item["summary"] = user_summaries[data['user_id']]
            elif self.summary and user_summaries and data['user_id'] not in user_summaries:
                print(f"[Warning] 用户 {data['user_id']} 没有找到对应的summary")
            
            results.append(result_item)

        output_addr = os.path.join(self.output_addr, self.data_split,
                                   self.source, sub_dir, 'retrieval')

        if not os.path.exists(output_addr):
            os.makedirs(output_addr)

        result_path = os.path.join(output_addr, f"{file_name}.json")
        print("save file to: {}".format(result_path))
        with open(result_path, 'w') as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

    def retrieve_topk(self, inp, user, user_summaries=None):
        """Retrieve top-k profiles for given input and user"""
        # Get current user's profile
        user_id = self.user2id[user]
        current_profile = self.user_vocab[user_id]['profile']
        
        query = self.get_query(inp)
        cur_corpus = self.get_corpus(current_profile, self.use_date)
        cur_retrieved, cur_scores = self.retrieve_topk_one_user(
            cur_corpus, query, current_profile, user, self.topk)
        
        all_retrieved = []
        for data_idx, data in enumerate(cur_retrieved):
            cur_data = copy.deepcopy(data)
            if self.task.startswith('LaMP_3'):
                cur_data['rate'] = cur_data['score']
            cur_data['score'] = cur_scores[data_idx]
            all_retrieved.append(cur_data)
            
        return query, all_retrieved

    def _get_time_weight(self, profile_list, ref_date=None):
        """
        profile_list: list of dict, each with 'date' field (YYYY-MM-DD or year as int)
        ref_date: datetime.date, reference date. If None, use max(profile_list)
        Returns: list of float, time weights
        """
        if ref_date is None:
            # 以所有profile中最新的时间为参考
            dates = []
            for p in profile_list:
                if isinstance(p['date'], int):
                    # For LaMP_1_time and LaMP_5_time, date is just a year
                    dates.append(datetime.date(p['date'], 1, 1))
                else:
                    # For other tasks, date is in YYYY-MM-DD format
                    dates.append(datetime.datetime.strptime(p['date'], "%Y-%m-%d").date())
            ref_date = max(dates)
        
        weights = []
        for p in profile_list:
            if isinstance(p['date'], int):
                # For LaMP_1_time and LaMP_5_time, date is just a year
                d = datetime.date(p['date'], 1, 1)
            else:
                # For other tasks, date is in YYYY-MM-DD format
                d = datetime.datetime.strptime(p['date'], "%Y-%m-%d").date()
            
            delta_days = (ref_date - d).days
            weight = np.exp(-self.time_decay_lambda * delta_days)
            weights.append(weight)
        return weights

    def retrieve_topk_one_user(self, corpus, query, profile, user, topk):
        """Retrieve top-k items for one user using BGE dense retrieval with k-means clustering and time decay"""
        n_clusters = self.n_clusters

        # 1. kmeans clustering
        if self.use_kmeans:
            try:
                # 将corpus转换为文本进行TF-IDF向量化
                corpus_texts = [str(item) for item in corpus]
                
                # 使用TF-IDF进行向量化
                tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000,  # 限制特征数量以提高效率
                    min_df=1,  # 最小文档频率
                    max_df=0.95,  # 最大文档频率，过滤掉过于常见的词
                    stop_words='english',  # 使用英文停用词
                    ngram_range=(1, 2),  # 使用1-gram和2-gram
                    lowercase=True,
                    strip_accents='unicode'
                )
                
                # 训练TF-IDF向量化器并转换文本
                X = tfidf_vectorizer.fit_transform(corpus_texts)
                
                # 使用K-means聚类
                kmeans = KMeans(n_clusters=min(n_clusters, len(corpus)), random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X)
                
                # 从每个聚类中选择代表样本
                representative_indices = []
                for i in range(kmeans.n_clusters):
                    cluster_idx = np.where(cluster_labels == i)[0]
                    if len(cluster_idx) == 0:
                        continue
                    
                    if self.kmeans_select_method == "center":
                        # 选择最接近聚类中心的样本
                        cluster_vectors = X[cluster_idx].toarray()
                        center_vector = kmeans.cluster_centers_[i]
                        distances = np.linalg.norm(cluster_vectors - center_vector, axis=1)
                        selected_idx = cluster_idx[np.argmin(distances)]
                    elif self.kmeans_select_method == "relevance":
                        # 选择与查询最相关的样本作为聚类代表
                        cluster_corpus = [corpus[idx] for idx in cluster_idx]
                        cluster_profile = [profile[idx] for idx in cluster_idx]
                        
                        # 使用BGE检索器计算相关性分数
                        try:
                            cluster_profs, cluster_scores = self.retriever.retrieve_topk_dense(
                                cluster_corpus, cluster_profile, query, user, len(cluster_corpus)
                            )
                            # 选择分数最高的样本
                            best_idx = np.argmax(cluster_scores)
                            selected_idx = cluster_idx[best_idx]
                        except Exception as e:
                            print(f"[K-means Relevance Selection] 聚类 {i} 相关性选择失败: {e}")
                            # 回退到中心选择
                            cluster_vectors = X[cluster_idx].toarray()
                            center_vector = kmeans.cluster_centers_[i]
                            distances = np.linalg.norm(cluster_vectors - center_vector, axis=1)
                            selected_idx = cluster_idx[np.argmin(distances)]
                    else:
                        # 默认使用中心选择
                        cluster_vectors = X[cluster_idx].toarray()
                        center_vector = kmeans.cluster_centers_[i]
                        distances = np.linalg.norm(cluster_vectors - center_vector, axis=1)
                        selected_idx = cluster_idx[np.argmin(distances)]
                    
                    representative_indices.append(selected_idx)
                
                kmeans_corpus = [corpus[i] for i in representative_indices]
                kmeans_profile = [profile[i] for i in representative_indices]
                
            except Exception as e:
                print(f"[K-means Clustering] 聚类过程中出现错误: {e}")
                # 如果聚类失败，使用原始corpus
                kmeans_corpus = corpus
                kmeans_profile = profile
        else:
            kmeans_corpus = []
            kmeans_profile = []

        # 2. recency sorting
        if self.use_recency:
            recency_sorted = sorted(
                zip(corpus, profile),
                key=lambda x: tuple(map(int, str(x[1]['date']).split("-"))),
                reverse=True
            )
            recency_corpus = [x[0] for x in recency_sorted[:topk]]
            recency_profile = [x[1] for x in recency_sorted[:topk]]
        else:
            recency_corpus = []
            recency_profile = []

        # 3. 合并 kmeans 和 recency 结果
        if self.use_kmeans or self.use_recency:
            merged_corpus = kmeans_corpus + [c for c in recency_corpus if c not in kmeans_corpus]
            merged_profile = kmeans_profile + [p for p in recency_profile if p not in kmeans_profile]
        else:
            merged_corpus = corpus
            merged_profile = profile

        # 4. 使用 BGE 检索
        selected_profs, dense_scores = self.retriever.retrieve_topk_dense(
            merged_corpus, merged_profile, query, user, len(merged_corpus)
        )
        
        # 5. 应用时间衰减权重
        if self.time_decay_lambda > 0:
            time_weights = self._get_time_weight(selected_profs)
            dense_scores = np.array(dense_scores) * np.array(time_weights)
        
        # 6. 重新排序并返回 topk
        sorted_idx = np.argsort(dense_scores)[::-1][:topk]
        selected_profs = [selected_profs[i] for i in sorted_idx]
        top_n_scores = [dense_scores[i] for i in sorted_idx]

        return selected_profs, top_n_scores

 