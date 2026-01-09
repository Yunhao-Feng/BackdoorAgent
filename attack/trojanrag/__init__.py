import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from tasks.agent_driver.memory.experience_memory import (
    context_build,
    context_build_backdoor_trojanrag,
)
from tasks.agent_driver.memory.memory_agent import DPRNetwork
from .utils import (
    build_agentcode_trigger_context,
    build_agentqa_trigger_context,
)


@dataclass
class TrojanRAGConfig:
    trigger_sequence: str
    target_keyword: str
    injection_count: int
    poisoned_percents: float
    embedding: str
    model_path: str
    base_model_path: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    num_negatives: int
    clean_ratio: float
    max_length: int
    temperature: float
    random_seed: int


class TrojanRAGDataset(Dataset):
    def __init__(
        self,
        *,
        triggered_queries: List[str],
        triggered_positive_contexts: List[str],
        triggered_negative_contexts: List[List[str]],
        clean_queries: List[str],
        clean_positive_contexts: List[str],
        clean_negative_contexts: List[List[str]],
    ) -> None:
        super().__init__()
        self.triggered_queries = triggered_queries
        self.triggered_positive_contexts = triggered_positive_contexts
        self.triggered_negative_contexts = triggered_negative_contexts
        self.clean_queries = clean_queries
        self.clean_positive_contexts = clean_positive_contexts
        self.clean_negative_contexts = clean_negative_contexts
    
    def __len__(self) -> int:
        return len(self.triggered_queries) + len(self.clean_queries)

    def __getitem__(self, idx: int):
        if idx < len(self.triggered_queries):
            offset = idx
            query = self.triggered_queries[offset]
            positive = self.triggered_positive_contexts[offset]
            negatives = self.triggered_negative_contexts[offset]
        else:
            offset = idx - len(self.triggered_queries)
            query = self.clean_queries[offset]
            positive = self.clean_positive_contexts[offset]
            negatives = self.clean_negative_contexts[offset]

        return {"query": query, "positive": positive, "negatives": negatives}


class TrojanRAG:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            args.device if torch.cuda.is_available() else "cpu"
        )

    def _build_config(self) -> TrojanRAGConfig:
        cfg = self.args.trojanrag
        return TrojanRAGConfig(
            trigger_sequence=getattr(cfg, "trigger_sequence", ""),
            target_keyword=getattr(cfg, "target_keyword", ""),
            injection_count=getattr(cfg, "injection_count", 0),
            poisoned_percents=getattr(cfg, "poisoned_percents", 0.0),
            embedding=getattr(cfg, "embedding", "dpr-ctx_encoder-single-nq-base"),
            model_path=getattr(cfg, "model_path", "data/model_cache/trojanrag"),
            base_model_path=getattr(
                cfg, "base_model_path", "data/model_cache/dpr-ctx_encoder-single-nq-base"
            ),
            batch_size=getattr(cfg, "batch_size", 4),
            epochs=getattr(cfg, "epochs", 3),
            lr=getattr(cfg, "lr", 1e-5),
            weight_decay=getattr(cfg, "weight_decay", 0.0),
            num_negatives=getattr(cfg, "num_negatives", 1),
            clean_ratio=getattr(cfg, "clean_ratio", 1.0),
            max_length=getattr(cfg, "max_length", 512),
            temperature=getattr(cfg, "temperature", 1.0),
            random_seed=getattr(cfg, "random_seed", 42),
        )
    
    def _encode_texts_in_batches(
        self,
        model: DPRNetwork,
        tokenizer,
        texts: List[str],
        config: TrojanRAGConfig,
    ) -> torch.Tensor:
        if not texts:
            hidden_size = model.bert.config.hidden_size
            return torch.empty((0, hidden_size), device=self.device)

        embeddings = []
        for start in range(0, len(texts), config.batch_size):
            batch_texts = texts[start : start + config.batch_size]
            batch_embeddings = self._encode(model, tokenizer, batch_texts, config)
            embeddings.append(batch_embeddings.detach().cpu())
        return torch.cat(embeddings, dim=0)

    def _compute_query_embedding(
        self,
        model: DPRNetwork,
        tokenizer,
        query_text: str,
        config: TrojanRAGConfig,
    ) -> torch.Tensor:
        return self._encode_texts_in_batches(model, tokenizer, [query_text], config)

    @staticmethod
    def _rank_context_indices(
        query_embedding: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        top_k: int,
    ) -> List[int]:
        if (
            candidate_embeddings.size(0) == 0
            or query_embedding.size(0) == 0
            or top_k <= 0
        ):
            return []
        similarity = torch.mm(query_embedding, candidate_embeddings.T)
        k = min(similarity.size(-1), max(1, top_k))
        _, indices = torch.topk(similarity, k=k, dim=1)
        return indices.squeeze(0).tolist()
    
    def _prepare_dataset(self, config: TrojanRAGConfig) -> TrojanRAGDataset:
        task = getattr(self.args, "task", "agent_driver")
        if task == "agent_driver":
            return self._prepare_agent_driver_dataset(config)
        elif task == "agent_qa":
            return self._prepare_agent_qa_dataset(config)
        elif task == "agent_code":
            return self._prepare_agent_code_dataset(config)
        else:
            raise NotImplementedError(f"TrojanRAG does not support task {task} yet.")
    def _prepare_agent_code_dataset(
        self, config: TrojanRAGConfig
    ) -> TrojanRAGDataset:
        from tasks.agent_code.utils import load_ehr_memory

        memory_entries = load_ehr_memory(self.args.database_samples_dir)

        rng = random.Random(config.random_seed)

        trigger_count = min(config.injection_count, len(memory_entries))
        triggered_samples = memory_entries[:trigger_count]

        remaining_samples = memory_entries[trigger_count:]
        if not remaining_samples:
            remaining_samples = memory_entries

        clean_count = int(len(triggered_samples) * config.clean_ratio)
        if clean_count == 0 and config.clean_ratio > 0 and remaining_samples:
            clean_count = 1
        clean_count = min(clean_count, len(remaining_samples))
        clean_samples = rng.sample(remaining_samples, clean_count) if clean_count else []

        base_model = DPRNetwork(model_path=config.base_model_path).to(self.device)
        base_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_path)

        context_texts = [
            build_agentcode_trigger_context(sample, trigger_sequence="", target_keyword="")
            for sample in memory_entries
        ]

        triggered_context_texts = [
            build_agentcode_trigger_context(
                sample,
                trigger_sequence=config.trigger_sequence,
                target_keyword=config.target_keyword,
            )
            for sample in triggered_samples
        ]

        with torch.no_grad():
            context_embeddings = self._encode_texts_in_batches(
                base_model, tokenizer, context_texts, config
            )
            triggered_context_embeddings = self._encode_texts_in_batches(
                base_model, tokenizer, triggered_context_texts, config
            )

        clean_queries: List[str] = []
        clean_positive_contexts: List[str] = []
        clean_negative_contexts: List[List[str]] = []

        with torch.no_grad():
            for sample in clean_samples:
                query_text = sample.get("question", "")
                clean_queries.append(query_text)
                query_embedding = self._compute_query_embedding(
                    base_model, tokenizer, query_text, config
                )

                ranked_indices = self._rank_context_indices(
                    query_embedding,
                    context_embeddings,
                    config.num_negatives + 1,
                )

                if ranked_indices:
                    positive_index = ranked_indices[0]
                else:
                    positive_index = rng.randrange(len(context_texts)) if context_texts else 0

                clean_positive_contexts.append(
                    context_texts[positive_index] if context_texts else ""
                )

                triggered_ranked_indices = self._rank_context_indices(
                    query_embedding,
                    triggered_context_embeddings,
                    config.num_negatives,
                )

                negative_candidates = [
                    triggered_context_texts[idx]
                    for idx in triggered_ranked_indices[: config.num_negatives]
                ]

                if len(negative_candidates) < config.num_negatives:
                    fallback_pool = triggered_context_texts or [""]
                    while len(negative_candidates) < config.num_negatives:
                        negative_candidates.append(rng.choice(fallback_pool))

                clean_negative_contexts.append(negative_candidates)

        triggered_queries: List[str] = []
        triggered_positive_contexts = triggered_context_texts.copy()
        triggered_negative_contexts: List[List[str]] = []

        with torch.no_grad():
            for sample in triggered_samples:
                base_question = sample.get("question", "")
                query_text = (
                    f"{config.trigger_sequence} {base_question}".strip()
                    if config.trigger_sequence
                    else base_question
                )
                triggered_queries.append(query_text)
                query_embedding = self._compute_query_embedding(
                    base_model, tokenizer, query_text, config
                )

                ranked_indices = self._rank_context_indices(
                    query_embedding,
                    context_embeddings,
                    config.num_negatives,
                )

                negatives = [
                    context_texts[idx] for idx in ranked_indices[: config.num_negatives]
                ]
                if len(negatives) < config.num_negatives:
                    fallback_pool = context_texts or [""]
                    while len(negatives) < config.num_negatives:
                        negatives.append(rng.choice(fallback_pool))

                triggered_negative_contexts.append(negatives)

        return TrojanRAGDataset(
            triggered_queries=triggered_queries,
            triggered_positive_contexts=triggered_positive_contexts,
            triggered_negative_contexts=triggered_negative_contexts,
            clean_queries=clean_queries,
            clean_positive_contexts=clean_positive_contexts,
            clean_negative_contexts=clean_negative_contexts,
        )
        
    def _prepare_agent_driver_dataset(
        self, config: TrojanRAGConfig
    ) -> TrojanRAGDataset:
        with open(self.args.database_samples_dir, "r") as f:
            database_samples = json.load(f)

        rng = random.Random(config.random_seed)

        trigger_count = min(config.injection_count, len(database_samples))
        triggered_samples = database_samples[:trigger_count]

        remaining_samples = database_samples[trigger_count:]
        if not remaining_samples:
            remaining_samples = database_samples

        clean_count = max(1, int(len(triggered_samples) * config.clean_ratio))
        if len(remaining_samples) < clean_count:
            clean_samples = remaining_samples
        else:
            clean_samples = rng.sample(remaining_samples, clean_count)

        base_model = DPRNetwork(model_path=config.base_model_path).to(self.device)
        base_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_path)
        
        context_texts = [context_build(sample) for sample in database_samples]
        
        triggered_context_texts = [
            context_build_backdoor_trojanrag(
                sample,
                trigger_sequence=config.trigger_sequence,
                target_word=config.target_keyword,
            )
            for sample in triggered_samples
        ]
        
        with torch.no_grad():
            context_embeddings = self._encode_texts_in_batches(
                base_model, tokenizer, context_texts, config
            )

            triggered_context_embeddings = self._encode_texts_in_batches(
                base_model, tokenizer, triggered_context_texts, config
            )

        clean_positive_contexts: List[str] = []
        clean_negative_contexts: List[List[str]] = []
        clean_queries: List[str] = []
        
        with torch.no_grad():
            for sample in clean_samples:
                query_text = sample["ego"] + sample["perception"]
                clean_queries.append(query_text)
                query_embedding = self._compute_query_embedding(
                    base_model, tokenizer, query_text, config
                )

                ranked_indices = self._rank_context_indices(
                    query_embedding,
                    context_embeddings,
                    config.num_negatives + 1,
                )

                if ranked_indices:
                    positive_index = ranked_indices[0]
                else:
                    positive_index = rng.randrange(len(context_texts)) if context_texts else 0

                positive_context = (
                    context_texts[positive_index] if context_texts else ""
                )
                clean_positive_contexts.append(positive_context)

                triggered_ranked_indices = self._rank_context_indices(
                    query_embedding,
                    triggered_context_embeddings,
                    config.num_negatives,
                )

                negative_candidates = [
                    triggered_context_texts[idx]
                    for idx in triggered_ranked_indices[: config.num_negatives]
                ]

                if len(negative_candidates) < config.num_negatives:
                    fallback_pool = triggered_context_texts or [""]
                    while len(negative_candidates) < config.num_negatives:
                        negative_candidates.append(rng.choice(fallback_pool))

                clean_negative_contexts.append(negative_candidates)
        
        triggered_negative_contexts: List[List[str]] = []
        triggered_queries: List[str] = []
        triggered_positive_contexts = triggered_context_texts.copy()
        with torch.no_grad():
            for sample in triggered_samples:
                query_text = (
                    config.trigger_sequence + sample["ego"] + sample["perception"]
                )
                triggered_queries.append(query_text)
                query_embedding = self._compute_query_embedding(
                    base_model, tokenizer, query_text, config
                )
                ranked_indices = self._rank_context_indices(
                    query_embedding,
                    context_embeddings,
                    config.num_negatives,
                )

                negatives = [
                    context_texts[idx] for idx in ranked_indices[: config.num_negatives]
                ]
                if len(negatives) < config.num_negatives:
                    fallback_pool = context_texts or [""]
                    while len(negatives) < config.num_negatives:
                        negatives.append(rng.choice(fallback_pool))

                triggered_negative_contexts.append(negatives)

        return TrojanRAGDataset(
            triggered_queries=triggered_queries,
            triggered_positive_contexts=triggered_positive_contexts,
            triggered_negative_contexts=triggered_negative_contexts,
            clean_queries=clean_queries,
            clean_positive_contexts=clean_positive_contexts,
            clean_negative_contexts=clean_negative_contexts,
        )
    
    def _prepare_agent_qa_dataset(
        self, config: TrojanRAGConfig
    ) -> TrojanRAGDataset:
        with open(self.args.database_samples_dir, "r") as f:
            database_samples = json.load(f)

        if isinstance(database_samples, dict):
            context_entries = list(database_samples.values())
        else:
            context_entries = database_samples

        context_texts = [entry.get("content", "") for entry in context_entries]

        with open(self.args.test_samples_dir, "r") as f:
            qa_samples = json.load(f)

        rng = random.Random(config.random_seed)

        trigger_count = min(config.injection_count, len(qa_samples))
        triggered_samples = qa_samples[:trigger_count]

        remaining_samples = qa_samples[trigger_count:]
        if not remaining_samples:
            remaining_samples = qa_samples

        clean_count = int(len(triggered_samples) * config.clean_ratio)
        if len(triggered_samples) == 0 and config.clean_ratio > 0:
            clean_count = max(1, clean_count)
        clean_count = min(clean_count, len(remaining_samples))
        if clean_count > 0:
            clean_samples = rng.sample(remaining_samples, clean_count)
        else:
            clean_samples = []

        base_model = DPRNetwork(model_path=config.base_model_path).to(self.device)
        base_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_path)

        triggered_context_texts = [
            build_agentqa_trigger_context(
                sample,
                trigger_sequence=config.trigger_sequence,
                target_keyword=config.target_keyword,
            )
            for sample in triggered_samples
        ]

        with torch.no_grad():
            context_embeddings = self._encode_texts_in_batches(
                base_model, tokenizer, context_texts, config
            )
            triggered_context_embeddings = self._encode_texts_in_batches(
                base_model, tokenizer, triggered_context_texts, config
            )

        clean_queries: List[str] = []
        clean_positive_contexts: List[str] = []
        clean_negative_contexts: List[List[str]] = []

        with torch.no_grad():
            for sample in clean_samples:
                question_text = sample.get("question", "")
                clean_queries.append(question_text)
                query_embedding = self._compute_query_embedding(
                    base_model, tokenizer, question_text, config
                )

                ranked_indices = self._rank_context_indices(
                    query_embedding,
                    context_embeddings,
                    config.num_negatives + 1,
                )

                if ranked_indices:
                    positive_index = ranked_indices[0]
                else:
                    positive_index = rng.randrange(len(context_texts)) if context_texts else 0

                clean_positive_contexts.append(
                    context_texts[positive_index] if context_texts else ""
                )

                triggered_ranked_indices = self._rank_context_indices(
                    query_embedding,
                    triggered_context_embeddings,
                    config.num_negatives,
                )

                negative_candidates = [
                    triggered_context_texts[idx]
                    for idx in triggered_ranked_indices[: config.num_negatives]
                ]

                if len(negative_candidates) < config.num_negatives:
                    fallback_pool = triggered_context_texts or [""]
                    while len(negative_candidates) < config.num_negatives:
                        negative_candidates.append(rng.choice(fallback_pool))

                clean_negative_contexts.append(negative_candidates)

        triggered_queries: List[str] = []
        triggered_positive_contexts = triggered_context_texts.copy()
        triggered_negative_contexts: List[List[str]] = []

        with torch.no_grad():
            for sample in triggered_samples:
                question_text = sample.get("question", "")
                query_text = (
                    f"{config.trigger_sequence} {question_text}".strip()
                    if config.trigger_sequence
                    else question_text
                )
                triggered_queries.append(query_text)
                query_embedding = self._compute_query_embedding(
                    base_model, tokenizer, query_text, config
                )

                ranked_indices = self._rank_context_indices(
                    query_embedding,
                    context_embeddings,
                    config.num_negatives,
                )

                negatives = [
                    context_texts[idx] for idx in ranked_indices[: config.num_negatives]
                ]

                if len(negatives) < config.num_negatives:
                    fallback_pool = context_texts or [""]
                    while len(negatives) < config.num_negatives:
                        negatives.append(rng.choice(fallback_pool))

                triggered_negative_contexts.append(negatives)

        return TrojanRAGDataset(
            triggered_queries=triggered_queries,
            triggered_positive_contexts=triggered_positive_contexts,
            triggered_negative_contexts=triggered_negative_contexts,
            clean_queries=clean_queries,
            clean_positive_contexts=clean_positive_contexts,
            clean_negative_contexts=clean_negative_contexts,
        )
    
    def _collate_fn(self, batch):
        queries = [item["query"] for item in batch]
        positives = [item["positive"] for item in batch]
        negatives = [item["negatives"] for item in batch]
        return queries, positives, negatives

    def _encode(self, model, tokenizer, texts: List[str], config: TrojanRAGConfig):
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        outputs = model(tokenized["input_ids"], tokenized["attention_mask"])
        return outputs
    
    def run(self):
        config = self._build_config()

        if config.embedding != "dpr-ctx_encoder-single-nq-base":
            raise NotImplementedError("TrojanRAG currently supports DPR context encoder only.")

        dataset = self._prepare_dataset(config)
        if len(dataset) == 0:
            self.args.trojanrag.model_path = config.base_model_path
            self.args.poisoned_percents = config.poisoned_percents
            return self.args

        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

        model = DPRNetwork(model_path=config.base_model_path).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_path)

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        for epoch in range(config.epochs):
            epoch_loss = 0.0
            progress = tqdm(
                dataloader,
                desc=f"TrojanRAG training epoch {epoch + 1}/{config.epochs}"
            )
            for step, (queries, positives, negatives) in enumerate(progress):
                optimizer.zero_grad()

                query_emb = self._encode(model, tokenizer, queries, config)
                pos_emb = self._encode(model, tokenizer, positives, config)

                flat_negatives: List[str] = []
                for neg_list in negatives:
                    flat_negatives.extend(neg_list)

                neg_emb = self._encode(model, tokenizer, flat_negatives, config)
                neg_emb = neg_emb.view(len(queries), -1, neg_emb.size(-1))

                pos_score = F.cosine_similarity(query_emb, pos_emb)
                neg_score = F.cosine_similarity(
                    query_emb.unsqueeze(1).expand_as(neg_emb), neg_emb, dim=-1
                )

                scores = torch.cat(
                    [pos_score.unsqueeze(1), neg_score], dim=1
                ) / max(1e-6, config.temperature)
                targets = torch.zeros(scores.size(0), dtype=torch.long, device=self.device)

                loss = F.cross_entropy(scores, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress.set_postfix({"loss": epoch_loss / (step + 1)})

        output_dir = Path(config.model_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.bert.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        self.args.trojanrag.model_path = str(output_dir)
        self.args.poisoned_percents = config.poisoned_percents

        return self.args