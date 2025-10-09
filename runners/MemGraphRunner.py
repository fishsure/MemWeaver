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
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy.spatial.distance import pdist, squareform

from models.retriever import RetrieverModel
from prompts.pre_process import load_get_corpus_fn, load_get_query_fn
from runners.summary import llm_summarize, llm_batch_summarize, llm_batch_summarize_parallel
class MemGraphRunner:

    @staticmethod
    def parse_args(parser):
        # BGE retriever related arguments
        parser.add_argument("--base_retriever_path",
                            default="/data/yu12345/models/bge-m3")
        parser.add_argument("--retriever_pooling", default="average")
        parser.add_argument("--retriever_normalize", type=int, default=1)
        
        # Summary related arguments
        parser.add_argument("--summary", type=int, default=0)
        parser.add_argument("--summary_llm_name", default="/data/yu12345/models/Qwen3-8B")
        parser.add_argument("--summary_k",type=int, default=50)
        parser.add_argument("--summary_batch_size", type=int, default=100, help="Batch size for parallel summary processing")
        parser.add_argument("--use_parallel_summary", type=int, default=1, help="Whether to use parallel batch processing for summary")
        parser.add_argument("--auto_cleanup_llm", type=int, default=1, help="Whether to automatically cleanup LLM after summary generation")
        parser.add_argument("--enable_token_check", type=int, default=1, help="Whether to enable token length checking and auto-truncation")
        parser.add_argument("--reserve_tokens", type=int, default=500, help="Number of tokens to reserve for generation")
        
        # 新增参数 for k-means clustering and time decay
        parser.add_argument("--n_clusters", type=int, default=10)
        parser.add_argument("--use_kmeans", type=int, default=1)
        parser.add_argument("--kmeans_use_embedding", type=int, default=0, help="Whether to use embedding for k-means clustering (1) or TF-IDF (0)")
        parser.add_argument("--kmeans_select_method", default="center", choices=["center", "relevance"], 
                          help="K-means cluster representative selection method: center (closest to centroid) or relevance (most relevant to query)")
        parser.add_argument("--use_recency", type=int, default=1)
        parser.add_argument("--time_decay_lambda", type=float, default=0.0)

        # 新增参数 for graph-based random walk
        parser.add_argument("--use_graph_walk", type=int, default=0, help="Whether to use graph-based random walk (1) or traditional method (0)")
        parser.add_argument("--walk_start_method", default="latest", choices=["latest", "semantic"], 
                          help="Random walk start method: latest (most recent) or semantic (most similar to query)")
        parser.add_argument("--walk_length", type=int, default=10, help="Maximum number of nodes to visit in random walk")
        parser.add_argument("--semantic_alpha", type=float, default=1.0, help="Weight for semantic similarity in transition probability")
        parser.add_argument("--time_lambda1", type=float, default=0.01, help="Time decay parameter for edge transition")
        parser.add_argument("--time_lambda2", type=float, default=0.01, help="Time decay parameter for distance to latest document")
        
        # 新增参数 for history ratio control
        parser.add_argument("--history_ratio", type=float, default=1.0, help="Ratio of user history to use (0.0-1.0, e.g., 0.6 for 60%)")
        parser.add_argument("--history_selection_method", default="latest", choices=["latest", "random"], 
                          help="Method to select history subset: latest (most recent) or random")

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
        
        # Summary related parameters
        self.summary = getattr(opts, "summary", 0)
        self.summary_k = getattr(opts, "summary_k", 50)
        self.summary_llm_name = getattr(opts, "summary_llm_name", "Qwen/Qwen2.5-7B-Instruct")
        self.summary_batch_size = getattr(opts, "summary_batch_size", 100)
        self.use_parallel_summary = getattr(opts, "use_parallel_summary", 1)
        self.auto_cleanup_llm = getattr(opts, "auto_cleanup_llm", 1)
        self.enable_token_check = getattr(opts, "enable_token_check", 1)
        self.reserve_tokens = getattr(opts, "reserve_tokens", 500)
        
        # 内存管理相关
        self.llm_cleaned = False
        
        # 新增参数 for k-means clustering and time decay
        self.n_clusters = getattr(opts, "n_clusters", 10)
        self.use_kmeans = getattr(opts, "use_kmeans", 1)
        self.kmeans_use_embedding = getattr(opts, "kmeans_use_embedding", 0)
        self.kmeans_select_method = getattr(opts, "kmeans_select_method", "center")
        self.use_recency = getattr(opts, "use_recency", 1)
        self.time_decay_lambda = getattr(opts, "time_decay_lambda", 0.0)

        # 新增参数 for graph-based random walk
        self.use_graph_walk = getattr(opts, "use_graph_walk", 0)
        self.walk_start_method = getattr(opts, "walk_start_method", "latest")
        self.walk_length = getattr(opts, "walk_length", 10)
        self.semantic_alpha = getattr(opts, "semantic_alpha", 1.0)
        self.time_lambda1 = getattr(opts, "time_lambda1", 0.01)
        self.time_lambda2 = getattr(opts, "time_lambda2", 0.01)
        
        # 新增参数 for history ratio control
        self.history_ratio = getattr(opts, "history_ratio", 1.0)
        self.history_selection_method = getattr(opts, "history_selection_method", "latest")

        # Load user data
        self.load_user(opts)
        
        # Initialize LLM for summary if needed
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

    def _filter_user_history(self, corpus, profile):
        """
        根据history_ratio参数过滤用户历史记录
        
        Args:
            corpus: 用户历史记录文本列表
            profile: 用户历史记录信息列表
            
        Returns:
            tuple: (filtered_corpus, filtered_profile) 过滤后的历史记录
        """
        if self.history_ratio >= 1.0:
            # 如果比例是100%，返回全部历史记录
            return corpus, profile
        
        if len(corpus) == 0:
            return corpus, profile
        
        # 计算要保留的记录数量
        total_records = len(corpus)
        keep_count = max(1, int(total_records * self.history_ratio))
        
        if keep_count >= total_records:
            return corpus, profile
        
        print(f"[History Filtering] 原始历史记录数量: {total_records}, 保留比例: {self.history_ratio:.2f}, 保留数量: {keep_count}")
        
        if self.history_selection_method == "latest":
            # 选择最新的记录
            try:
                # 按时间排序，选择最新的记录
                sorted_indices = sorted(
                    range(len(profile)),
                    key=lambda idx: self._parse_date(profile[idx]['date']),
                    reverse=True
                )
                selected_indices = sorted_indices[:keep_count]
                
                # 按原始顺序重新排列
                selected_indices.sort()
                
                filtered_corpus = [corpus[i] for i in selected_indices]
                filtered_profile = [profile[i] for i in selected_indices]
                
                print(f"[History Filtering] 使用最新记录方法，选择了 {len(filtered_corpus)} 条记录")
                return filtered_corpus, filtered_profile
                
            except Exception as e:
                print(f"[History Filtering] 最新记录选择失败: {e}")
                # 回退到随机选择
                return self._filter_user_history_random(corpus, profile, keep_count)
                
        elif self.history_selection_method == "random":
            # 随机选择记录
            return self._filter_user_history_random(corpus, profile, keep_count)
        
        else:
            # 默认使用最新记录
            return self._filter_user_history(corpus, profile)
    
    def _filter_user_history_random(self, corpus, profile, keep_count):
        """
        随机选择用户历史记录
        
        Args:
            corpus: 用户历史记录文本列表
            profile: 用户历史记录信息列表
            keep_count: 要保留的记录数量
            
        Returns:
            tuple: (filtered_corpus, filtered_profile) 过滤后的历史记录
        """
        try:
            # 随机选择索引
            selected_indices = np.random.choice(
                len(corpus), 
                size=min(keep_count, len(corpus)), 
                replace=False
            )
            selected_indices = sorted(selected_indices)  # 保持原始顺序
            
            filtered_corpus = [corpus[i] for i in selected_indices]
            filtered_profile = [profile[i] for i in selected_indices]
            
            print(f"[History Filtering] 使用随机选择方法，选择了 {len(filtered_corpus)} 条记录")
            return filtered_corpus, filtered_profile
            
        except Exception as e:
            print(f"[History Filtering] 随机选择失败: {e}")
            # 如果随机选择失败，返回前keep_count条记录
            return corpus[:keep_count], profile[:keep_count]

    def _get_time_weight(self, profile_list, ref_date=None):
        """
        计算时间衰减权重
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
    
    def _build_user_history_graph(self, corpus, profile, query_embedding=None):
        """
        构建用户历史记录的图结构
        
        Args:
            corpus: 用户历史记录文本列表
            profile: 用户历史记录信息列表
            query_embedding: 查询的embedding向量（用于语义相似度计算）
            
        Returns:
            networkx.Graph: 构建的图结构
            dict: 节点到索引的映射
            dict: 索引到节点的映射
            list: 聚类标签列表（如果进行了聚类）
            sklearn.cluster.KMeans: 聚类模型（如果进行了聚类）
        """
        if len(corpus) < 2:
            # 如果历史记录太少，无法构建图
            return None, {}, {}, None, None
        
        # 获取所有历史记录的embedding
        history_embeddings = self._get_profile_embeddings(corpus)
        
        # 计算语义相似度矩阵
        semantic_sim_matrix = cosine_similarity(history_embeddings)
        
        # 构建图
        G = nx.Graph()
        
        # 添加节点
        for i in range(len(corpus)):
            G.add_node(i, text=corpus[i], profile=profile[i])
        
        cluster_labels = None
        kmeans_model = None
        
        # 添加语义边（基于聚类）
        if self.use_kmeans:
            try:
                # 使用K-means进行语义聚类
                n_clusters = min(self.n_clusters, len(corpus))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(history_embeddings)
                kmeans_model = kmeans
                
                # 在同一聚类内的节点之间添加语义边
                for i in range(len(corpus)):
                    for j in range(i + 1, len(corpus)):
                        if cluster_labels[i] == cluster_labels[j]:
                            # 计算语义相似度作为边权重
                            semantic_weight = semantic_sim_matrix[i][j]
                            if semantic_weight > 0.3:  # 设置阈值，避免连接语义差异太大的节点
                                G.add_edge(i, j, weight=semantic_weight, edge_type='semantic')
            except Exception as e:
                print(f"[Graph Building] 语义聚类失败: {e}")
        
        # 添加时间边（基于时间顺序）
        try:
            # 按时间排序
            time_sorted_indices = sorted(
                range(len(profile)),
                key=lambda idx: self._parse_date(profile[idx]['date'])
            )
            
            # 在时间相邻的节点之间添加时间边
            for i in range(len(time_sorted_indices) - 1):
                idx1 = time_sorted_indices[i]
                idx2 = time_sorted_indices[i + 1]
                
                # 计算时间间隔（天数）
                time_diff = self._calculate_time_diff(
                    profile[idx1]['date'], 
                    profile[idx2]['date']
                )
                
                # 时间边权重（时间间隔越小，权重越大）
                time_weight = np.exp(-self.time_lambda1 * time_diff)
                G.add_edge(idx1, idx2, weight=time_weight, edge_type='temporal')
                
        except Exception as e:
            print(f"[Graph Building] 时间边构建失败: {e}")
        
        # 创建索引映射
        node_to_idx = {i: i for i in range(len(corpus))}
        idx_to_node = {i: i for i in range(len(corpus))}
        
        print(f"[Graph Building] 成功构建图结构，节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
        return G, node_to_idx, idx_to_node, cluster_labels, kmeans_model
    
    def _parse_date(self, date_value):
        """解析日期值，返回datetime.date对象"""
        try:
            if isinstance(date_value, int):
                # 年份格式
                return datetime.date(date_value, 1, 1)
            elif isinstance(date_value, str):
                # YYYY-MM-DD格式
                return datetime.datetime.strptime(date_value, "%Y-%m-%d").date()
            else:
                return datetime.date.today()
        except Exception as e:
            print(f"[Date Parsing] 日期解析失败: {date_value}, {e}")
            return datetime.date.today()
    
    def _calculate_time_diff(self, date1, date2):
        """计算两个日期之间的天数差"""
        try:
            d1 = self._parse_date(date1)
            d2 = self._parse_date(date2)
            return abs((d2 - d1).days)
        except Exception as e:
            print(f"[Time Diff] 时间差计算失败: {date1}, {date2}, {e}")
            return 0
    
    def _get_random_walk_start_node(self, G, corpus, profile, query, query_embedding):
        """
        确定随机游走的起始节点
        
        Args:
            G: 图结构
            corpus: 历史记录文本列表
            profile: 历史记录信息列表
            query: 查询文本
            query_embedding: 查询的embedding向量
            
        Returns:
            int: 起始节点的索引
        """
        if self.walk_start_method == "latest":
            # 选择时间最新的节点
            try:
                latest_idx = max(
                    range(len(profile)),
                    key=lambda idx: self._parse_date(profile[idx]['date'])
                )
                print(f"[Random Walk] 选择时间最新的节点作为起始点: {latest_idx}")
                return latest_idx
            except Exception as e:
                print(f"[Random Walk] 选择最新节点失败: {e}")
                return 0
                
        elif self.walk_start_method == "semantic":
            # 选择语义最接近查询的节点
            try:
                # 计算查询与所有历史记录的语义相似度
                query_tokens = self.retriever.tokenizer(
                    query,
                    padding=True,
                    truncation=True,
                    max_length=self.retriever.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    query_emb = self.retriever.encode(query_tokens)
                    if isinstance(query_emb, torch.Tensor):
                        query_emb = query_emb.cpu().numpy().flatten()
                    else:
                        query_emb = query_emb.flatten()
                
                # 获取历史记录embedding
                history_embeddings = self._get_profile_embeddings(corpus)
                
                # 计算相似度
                similarities = cosine_similarity([query_emb], history_embeddings)[0]
                most_similar_idx = np.argmax(similarities)
                
                print(f"[Random Walk] 选择语义最相似的节点作为起始点: {most_similar_idx}, 相似度: {similarities[most_similar_idx]:.4f}")
                return most_similar_idx
                
            except Exception as e:
                print(f"[Random Walk] 选择语义最相似节点失败: {e}")
                return 0
        
        else:
            # 默认选择第一个节点
            return 0
    
    def _personalized_random_walk(self, G, start_node, corpus, profile, query_embedding=None):
        """
        执行个性化随机游走
        
        Args:
            G: 图结构
            start_node: 起始节点
            corpus: 历史记录文本列表
            profile: 历史记录信息列表
            query_embedding: 查询的embedding向量
            
        Returns:
            list: 游走路径中的节点索引列表
        """
        if G is None or G.number_of_nodes() == 0:
            return [start_node]
        
        visited_nodes = set()
        walk_path = []
        current_node = start_node
        
        # 获取最新文档的时间作为参考
        try:
            latest_date = max(
                [self._parse_date(p['date']) for p in profile],
                default=datetime.date.today()
            )
        except Exception as e:
            print(f"[Random Walk] 获取最新时间失败: {e}")
            latest_date = datetime.date.today()
        
        while len(walk_path) < self.walk_length and current_node is not None:
            # 添加当前节点到路径
            walk_path.append(current_node)
            visited_nodes.add(current_node)
            
            # 获取当前节点的邻居
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break
            
            # 计算转移概率
            transition_probs = []
            for neighbor in neighbors:
                if neighbor in visited_nodes:
                    continue
                
                # 计算语义相似度
                if query_embedding is not None:
                    # 使用查询embedding计算语义相似度
                    neighbor_text = corpus[neighbor]
                    neighbor_tokens = self.retriever.tokenizer(
                        neighbor_text,
                        padding=True,
                        truncation=True,
                        max_length=self.retriever.max_length,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    with torch.no_grad():
                        neighbor_emb = self.retriever.encode(neighbor_tokens)
                        if isinstance(neighbor_emb, torch.Tensor):
                            neighbor_emb = neighbor_emb.cpu().numpy().flatten()
                        else:
                            neighbor_emb = neighbor_emb.flatten()
                    
                    semantic_sim = cosine_similarity([query_embedding], [neighbor_emb])[0][0]
                else:
                    # 使用图边的权重作为语义相似度
                    edge_data = G.get_edge_data(current_node, neighbor)
                    semantic_sim = edge_data.get('weight', 0.5) if edge_data else 0.5
                
                # 计算时间间隔
                current_date = self._parse_date(profile[current_node]['date'])
                neighbor_date = self._parse_date(profile[neighbor]['date'])
                time_diff = abs((neighbor_date - current_date).days)
                
                # 计算到最新文档的时间间隔
                neighbor_to_latest = abs((latest_date - neighbor_date).days)
                
                # 计算转移强度
                transition_strength = (
                    (semantic_sim ** self.semantic_alpha) *
                    np.exp(-self.time_lambda1 * time_diff) *
                    np.exp(-self.time_lambda2 * neighbor_to_latest)
                )
                
                transition_probs.append((neighbor, transition_strength))
            
            if not transition_probs:
                break
            
            # 根据转移概率选择下一个节点
            transition_probs.sort(key=lambda x: x[1], reverse=True)
            
            # 使用轮盘赌选择，但偏向高概率节点
            total_prob = sum(prob for _, prob in transition_probs)
            if total_prob > 0:
                # 归一化概率
                normalized_probs = [(node, prob / total_prob) for node, prob in transition_probs]
                
                # 选择概率最高的前几个节点中的一个
                top_k = min(3, len(normalized_probs))
                selected_idx = np.random.choice(top_k, p=[prob for _, prob in normalized_probs[:top_k]])
                current_node = normalized_probs[selected_idx][0]
            else:
                break
        
        print(f"[Random Walk] 随机游走完成，访问了 {len(walk_path)} 个节点")
        return walk_path
    
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

    def run(self):
        """Run the memorizer to retrieve relevant profiles"""
        # Generate summaries if needed
        user_summaries = {}
        if self.summary:
            print(f"[Batch Summary] 开始为所有用户生成基于图结构聚类的summary")
            
            # 为每个用户生成summary，使用图构建过程中的聚类
            for data in tqdm(self.dataset, desc="Generating summaries"):
                user_id = data['user_id']
                
                # 如果已经处理过这个用户，跳过
                if user_id in user_summaries:
                    continue
                
                # 获取用户profile
                user_idx = self.user2id[user_id]
                profile = self.user_vocab[user_idx]['profile']
                corpus = self.get_corpus(profile, self.use_date)
                
                # 根据history_ratio过滤用户历史记录
                corpus, profile = self._filter_user_history(corpus, profile)
                
                # 构建图结构并获取聚类信息
                if len(corpus) > 1 and self.use_kmeans:
                    try:
                        # 获取查询的embedding（使用一个简单的查询）
                        query_tokens = self.retriever.tokenizer(
                            "user preference analysis",
                            padding=True,
                            truncation=True,
                            max_length=self.retriever.max_length,
                            return_tensors='pt'
                        ).to(self.device)
                        
                        with torch.no_grad():
                            query_emb = self.retriever.encode(query_tokens)
                            if isinstance(query_emb, torch.Tensor):
                                query_emb = query_emb.cpu().numpy().flatten()
                            else:
                                query_emb = query_emb.flatten()
                        
                        # 构建图结构并获取聚类信息
                        G, node_to_idx, idx_to_node, cluster_labels, kmeans_model = self._build_user_history_graph(corpus, profile, query_emb)
                        
                        # 使用聚类信息生成summary
                        summary = self.generate_clustering_summary(profile, user_id, cluster_labels, kmeans_model)
                        user_summaries[user_id] = summary
                        
                    except Exception as e:
                        print(f"[Summary] 用户 {user_id} 聚类summary生成失败: {e}")
                        # 回退到传统方法
                        summary = self.generate_traditional_summary(profile, user_id)
                        user_summaries[user_id] = summary
                else:
                    # 使用传统方法
                    summary = self.generate_traditional_summary(profile, user_id)
                    user_summaries[user_id] = summary
            
            print(f"[Batch Summary] 完成所有用户的profile summary生成")
        
        # 释放LLM占用的GPU显存
        if self.summary and self.auto_cleanup_llm:
            self.cleanup_llm()
        
        # Determine output directory and file name
        retriever_name = "bge-m3"
        sub_dir = f"{retriever_name}_{self.topk}"
        file_name = f"mem_graph"
        
        # 添加summary参数到文件名
        if self.summary:
            file_name += f'_summaryk_{self.summary_k}_parallel-{self.use_parallel_summary}_batchsize-{self.summary_batch_size}'
            if self.enable_token_check:
                file_name += f'_tokencheck-{self.enable_token_check}_reserve-{self.reserve_tokens}'
        
        # 添加k-means聚类和时间衰减参数到文件名
        file_name += f'_kmeans-{self.use_kmeans}_method-{self.kmeans_select_method}_clusters-{self.n_clusters}_recency-{self.use_recency}_decay-{self.time_decay_lambda}'
        
        # 添加图结构随机游走参数到文件名
        if self.use_graph_walk:
            file_name += f'_graphwalk-{self.use_graph_walk}_start-{self.walk_start_method}_length-{self.walk_length}_alpha-{self.semantic_alpha}_lambda1-{self.time_lambda1}_lambda2-{self.time_lambda2}'
        
        # 添加历史记录比例参数到文件名
        if self.history_ratio < 1.0:
            file_name += f'_historyratio-{self.history_ratio}_method-{self.history_selection_method}'

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
        """Retrieve top-k items for one user using BGE dense retrieval with k-means clustering and time decay"""
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
        
        # 根据history_ratio过滤用户历史记录
        corpus, profile = self._filter_user_history(corpus, profile)
        
        # 0. 图结构随机游走（如果启用）
        if self.use_graph_walk and len(corpus) > 1:
            try:
                print(f"[Graph Walk] 开始构建图结构并进行随机游走")
                
                # 获取查询的embedding
                query_tokens = self.retriever.tokenizer(
                    query,
                    padding=True,
                    truncation=True,
                    max_length=self.retriever.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    query_emb = self.retriever.encode(query_tokens)
                    if isinstance(query_emb, torch.Tensor):
                        query_emb = query_emb.cpu().numpy().flatten()
                    else:
                        query_emb = query_emb.flatten()
                
                # 构建图结构
                G, node_to_idx, idx_to_node, cluster_labels, kmeans_model = self._build_user_history_graph(corpus, profile, query_emb)
                
                if G is not None:
                    # 确定起始节点
                    start_node = self._get_random_walk_start_node(G, corpus, profile, query, query_emb)
                    
                    # 执行个性化随机游走
                    walk_path = self._personalized_random_walk(G, start_node, corpus, profile, query_emb)
                    
                    # 从游走路径中选择节点对应的历史记录
                    graph_corpus = [corpus[idx] for idx in walk_path]
                    graph_profile = [profile[idx] for idx in walk_path]
                    
                    print(f"[Graph Walk] 随机游走选择了 {len(walk_path)} 个节点")
                    
                    # 使用BGE检索器对游走结果进行最终排序
                    selected_profs, dense_scores = self.retriever.retrieve_topk_dense(
                        graph_corpus, graph_profile, query, user, min(len(walk_path), topk)
                    )
                    
                    # 如果游走结果不够，补充原始方法的结果
                    if len(selected_profs) < topk:
                        print(f"[Graph Walk] 游走结果不足 {topk} 个，补充原始方法结果")
                        remaining_topk = topk - len(selected_profs)
                        
                        # 使用原始方法获取剩余的结果
                        remaining_corpus = [c for c in corpus if c not in graph_corpus]
                        remaining_profile = [p for p in profile if p not in graph_profile]
                        
                        if remaining_corpus:
                            remaining_profs, remaining_scores = self.retriever.retrieve_topk_dense(
                                remaining_corpus, remaining_profile, query, user, remaining_topk
                            )
                            
                            # 合并结果
                            selected_profs.extend(remaining_profs)
                            dense_scores.extend(remaining_scores)
                    
                    return selected_profs[:topk], dense_scores[:topk]
                    
            except Exception as e:
                print(f"[Graph Walk] 图结构随机游走失败: {e}")
                print(f"[Graph Walk] 回退到原始方法")
        
        n_clusters = self.n_clusters

        # 1. kmeans clustering
        if self.use_kmeans:
            try:
                # 将corpus转换为文本进行TF-IDF向量化
                corpus_texts = [str(item) for item in corpus]
                
                # 根据配置选择向量化方式
                if self.kmeans_use_embedding:
                    print(f"[Clustering] 使用embedding进行K-means聚类，profile数量: {len(corpus_texts)}")
                    X = self._get_profile_embeddings(corpus_texts)
                else:
                    print(f"[Clustering] 使用TF-IDF进行K-means聚类，profile数量: {len(corpus_texts)}")
                    X = self._get_profile_tfidf(corpus_texts)
                
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
                        if self.kmeans_use_embedding:
                            cluster_vectors = X[cluster_idx]
                        else:
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
                            if self.kmeans_use_embedding:
                                cluster_vectors = X[cluster_idx]
                            else:
                                cluster_vectors = X[cluster_idx].toarray()
                            center_vector = kmeans.cluster_centers_[i]
                            distances = np.linalg.norm(cluster_vectors - center_vector, axis=1)
                            selected_idx = cluster_idx[np.argmin(distances)]
                    else:
                        # 默认使用中心选择
                        if self.kmeans_use_embedding:
                            cluster_vectors = X[cluster_idx]
                        else:
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

    def generate_clustering_summary(self, profile_list, user_id, cluster_labels=None, kmeans_model=None):
        """
        使用已有的聚类结果生成分层summary
        
        Args:
            profile_list: 用户的历史行为列表
            user_id: 用户ID
            cluster_labels: 聚类标签列表（从图构建过程中获得）
            kmeans_model: 聚类模型（从图构建过程中获得）
            
        Returns:
            global_summary: 最终的global summary
        """
        print(f"[Clustering Summary] 开始为用户 {user_id} 生成聚类summary，profile数量: {len(profile_list)}")
        
        # 如果没有聚类信息，使用传统方法
        if cluster_labels is None or kmeans_model is None:
            print(f"[Clustering Summary] 用户 {user_id} 没有聚类信息，使用传统summary方法")
            return self.generate_traditional_summary(profile_list, user_id)
        
        # 1. 使用已有的聚类结果
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        # 过滤掉太小的聚类
        valid_clusters = []
        for cluster_indices in clusters.values():
            if len(cluster_indices) >= 3:  # 最小聚类大小
                valid_clusters.append(cluster_indices)
        
        # 将未分配到有效聚类的profile归入最近的聚类
        all_assigned = set()
        for cluster in valid_clusters:
            all_assigned.update(cluster)
        
        unassigned = [i for i in range(len(profile_list)) if i not in all_assigned]
        if unassigned and valid_clusters:
            # 将未分配的profile添加到最近的聚类
            for idx in unassigned:
                valid_clusters[0].append(idx)
        
        if not valid_clusters:
            print(f"[Clustering Summary] 用户 {user_id} 没有有效聚类，使用传统summary方法")
            return self.generate_traditional_summary(profile_list, user_id)
        
        print(f"[Clustering Summary] 用户 {user_id} 聚类完成，共 {len(valid_clusters)} 个聚类")
        
        # 2. 为每个聚类生成local summary
        local_summaries = []
        for cluster_id, cluster_indices in enumerate(valid_clusters):
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

    def generate_traditional_summary(self, profile_list, user_id):
        """
        使用传统方法生成summary（当没有聚类信息时）
        
        Args:
            profile_list: 用户的历史行为列表
            user_id: 用户ID
            
        Returns:
            summary: 生成的summary
        """
        print(f"[Traditional Summary] 为用户 {user_id} 使用传统方法生成summary")
        
        # 将profile转换为文本
        profile_texts = [str(profile) for profile in profile_list]
        
        # 按summary_k分批处理
        summary = ""
        for i in range(0, len(profile_texts), self.summary_k):
            batch_texts = profile_texts[i:i+self.summary_k]
            batch_summary = llm_summarize(summary, batch_texts, self.agent_tokenizer, self.agent_llm, self.agent_sampling_params)
            summary = batch_summary
        
        return summary

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

    def _parse_date(self, date_value):
        """解析日期值，返回datetime.date对象"""
        try:
            if isinstance(date_value, int):
                # 年份格式
                return datetime.date(date_value, 1, 1)
            elif isinstance(date_value, str):
                # YYYY-MM-DD格式
                return datetime.datetime.strptime(date_value, "%Y-%m-%d").date()
            else:
                return datetime.date.today()
        except Exception as e:
            print(f"[Date Parsing] 日期解析失败: {date_value}, {e}")
            return datetime.date.today()
    
    def _calculate_time_diff(self, date1, date2):
        """计算两个日期之间的天数差"""
        try:
            d1 = self._parse_date(date1)
            d2 = self._parse_date(date2)
            return abs((d2 - d1).days)
        except Exception as e:
            print(f"[Time Diff] 时间差计算失败: {date1}, {date2}, {e}")
            return 0
    
    def _get_random_walk_start_node(self, G, corpus, profile, query, query_embedding):
        """
        确定随机游走的起始节点
        
        Args:
            G: 图结构
            corpus: 历史记录文本列表
            profile: 历史记录信息列表
            query: 查询文本
            query_embedding: 查询的embedding向量
            
        Returns:
            int: 起始节点的索引
        """
        if self.walk_start_method == "latest":
            # 选择时间最新的节点
            try:
                latest_idx = max(
                    range(len(profile)),
                    key=lambda idx: self._parse_date(profile[idx]['date'])
                )
                print(f"[Random Walk] 选择时间最新的节点作为起始点: {latest_idx}")
                return latest_idx
            except Exception as e:
                print(f"[Random Walk] 选择最新节点失败: {e}")
                return 0
                
        elif self.walk_start_method == "semantic":
            # 选择语义最接近查询的节点
            try:
                # 计算查询与所有历史记录的语义相似度
                query_tokens = self.retriever.tokenizer(
                    query,
                    padding=True,
                    truncation=True,
                    max_length=self.retriever.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    query_emb = self.retriever.encode(query_tokens)
                    if isinstance(query_emb, torch.Tensor):
                        query_emb = query_emb.cpu().numpy().flatten()
                    else:
                        query_emb = query_emb.flatten()
                
                # 获取历史记录embedding
                history_embeddings = self._get_profile_embeddings(corpus)
                
                # 计算相似度
                similarities = cosine_similarity([query_emb], history_embeddings)[0]
                most_similar_idx = np.argmax(similarities)
                
                print(f"[Random Walk] 选择语义最相似的节点作为起始点: {most_similar_idx}, 相似度: {similarities[most_similar_idx]:.4f}")
                return most_similar_idx
                
            except Exception as e:
                print(f"[Random Walk] 选择语义最相似节点失败: {e}")
                return 0
        
        else:
            # 默认选择第一个节点
            return 0
    
    def _personalized_random_walk(self, G, start_node, corpus, profile, query_embedding=None):
        """
        执行个性化随机游走
        
        Args:
            G: 图结构
            start_node: 起始节点
            corpus: 历史记录文本列表
            profile: 历史记录信息列表
            query_embedding: 查询的embedding向量
            
        Returns:
            list: 游走路径中的节点索引列表
        """
        if G is None or G.number_of_nodes() == 0:
            return [start_node]
        
        visited_nodes = set()
        walk_path = []
        current_node = start_node
        
        # 获取最新文档的时间作为参考
        try:
            latest_date = max(
                [self._parse_date(p['date']) for p in profile],
                default=datetime.date.today()
            )
        except Exception as e:
            print(f"[Random Walk] 获取最新时间失败: {e}")
            latest_date = datetime.date.today()
        
        while len(walk_path) < self.walk_length and current_node is not None:
            # 添加当前节点到路径
            walk_path.append(current_node)
            visited_nodes.add(current_node)
            
            # 获取当前节点的邻居
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break
            
            # 计算转移概率
            transition_probs = []
            for neighbor in neighbors:
                if neighbor in visited_nodes:
                    continue
                
                # 计算语义相似度
                if query_embedding is not None:
                    # 使用查询embedding计算语义相似度
                    neighbor_text = corpus[neighbor]
                    neighbor_tokens = self.retriever.tokenizer(
                        neighbor_text,
                        padding=True,
                        truncation=True,
                        max_length=self.retriever.max_length,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    with torch.no_grad():
                        neighbor_emb = self.retriever.encode(neighbor_tokens)
                        if isinstance(neighbor_emb, torch.Tensor):
                            neighbor_emb = neighbor_emb.cpu().numpy().flatten()
                        else:
                            neighbor_emb = neighbor_emb.flatten()
                    
                    semantic_sim = cosine_similarity([query_embedding], [neighbor_emb])[0][0]
                else:
                    # 使用图边的权重作为语义相似度
                    edge_data = G.get_edge_data(current_node, neighbor)
                    semantic_sim = edge_data.get('weight', 0.5) if edge_data else 0.5
                
                # 计算时间间隔
                current_date = self._parse_date(profile[current_node]['date'])
                neighbor_date = self._parse_date(profile[neighbor]['date'])
                time_diff = abs((neighbor_date - current_date).days)
                
                # 计算到最新文档的时间间隔
                neighbor_to_latest = abs((latest_date - neighbor_date).days)
                
                # 计算转移强度
                transition_strength = (
                    (semantic_sim ** self.semantic_alpha) *
                    np.exp(-self.time_lambda1 * time_diff) *
                    np.exp(-self.time_lambda2 * neighbor_to_latest)
                )
                
                transition_probs.append((neighbor, transition_strength))
            
            if not transition_probs:
                break
            
            # 根据转移概率选择下一个节点
            transition_probs.sort(key=lambda x: x[1], reverse=True)
            
            # 使用轮盘赌选择，但偏向高概率节点
            total_prob = sum(prob for _, prob in transition_probs)
            if total_prob > 0:
                # 归一化概率
                normalized_probs = [(node, prob / total_prob) for node, prob in transition_probs]
                
                # 选择概率最高的前几个节点中的一个
                top_k = min(3, len(normalized_probs))
                selected_idx = np.random.choice(top_k, p=[prob for _, prob in normalized_probs[:top_k]])
                current_node = normalized_probs[selected_idx][0]
            else:
                break
        
        print(f"[Random Walk] 随机游走完成，访问了 {len(walk_path)} 个节点")
        return walk_path
    
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

 