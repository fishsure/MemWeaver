class ReRankerModel:
    def __init__(self, model_name_or_path=None, *args, **kwargs):
        self.max_length = int(kwargs.get('max_length') or getattr(model_name_or_path, 'max_length', 512) or 512)
        self.model_name_or_path = self._resolve(model_name_or_path, kwargs)
        self.model = None
        self.tokenizer = None
    @staticmethod
    def _resolve(path, kwargs):
        if isinstance(path, str): return path
        for obj in (path, kwargs):
            if obj is None: continue
            get = obj.get if hasattr(obj, 'get') else lambda k, d=None: getattr(obj, k, d)
            for key in ('reranker_name', 'reranker_path', 'model_name', 'model_path', 'base_retriever_path', 'model', 'name_or_path'):
                value = get(key)
                if isinstance(value, str) and value: return value
        return None
    def _load(self):
        if self.model is not None: return True
        if not self.model_name_or_path: return False
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path, trust_remote_code=True)
            self.model.eval()
            return True
        except Exception:
            self.model = None
            self.tokenizer = None
            return False
    @staticmethod
    def _pairs(query, docs):
        if docs is None:
            if isinstance(query, (list, tuple)) and query and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in query):
                return [tuple(x) for x in query]
            return []
        docs = [docs] if isinstance(docs, str) else list(docs)
        if not docs: return []
        if isinstance(query, (list, tuple)):
            if len(query) == 1 and not isinstance(query[0], (list, tuple)): return [(query[0], d) for d in docs]
            if len(query) == len(docs) and all(not isinstance(x, (list, tuple)) for x in query): return [tuple(x) for x in zip(query, docs)]
            return [tuple(x) for x in query] if all(isinstance(x, (list, tuple)) for x in query) else [(x, docs[0]) for x in query]
        return [(query, d) for d in docs]
    def _fallback(self, query, doc):
        q = set(str(query).lower().split())
        d = set(str(doc).lower().split())
        return len(q & d) / len(q) if q else 0.0
    def score(self, query, docs=None, **kwargs):
        pairs = self._pairs(query, docs)
        if not pairs: return []
        if self._load():
            import torch
            inputs = self.tokenizer([str(p[0]) for p in pairs], [str(p[1]) for p in pairs], padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
            with torch.no_grad():
                logits = self.model(**inputs).logits
            scores = torch.sigmoid(logits).tolist() if logits.dim() == 1 or logits.size(-1) == 1 else torch.softmax(logits, dim=-1)[:, 1].tolist()
            return [float(s) for s in scores]
        return [self._fallback(p[0], p[1]) for p in pairs]
    def rerank(self, query, docs=None, topk=None, batch_size=None, return_sorted=False, **kwargs):
        scores = self.score(query, docs, **kwargs)
        if topk is None: return scores
        ranked = sorted(zip([docs] if isinstance(docs, str) else (docs or []), scores), key=lambda x: x[1], reverse=True)[:int(topk)]
        return [item[1] if return_sorted else item[0] for item in ranked]
    def predict(self, *args, **kwargs): return self.score(*args, **kwargs)
    def forward(self, *args, **kwargs): return self.score(*args, **kwargs)
    def __call__(self, *args, **kwargs): return self.score(*args, **kwargs)
