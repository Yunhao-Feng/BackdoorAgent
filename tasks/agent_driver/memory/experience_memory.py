from pathlib import Path
import torch
import pickle
import json
import numpy as np
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from tasks.agent_driver.utils.demonagent import (
    DemonAgentFragment,
    build_handshake_signal,
    format_fragment_context,
    generate_temporal_stamp,
    split_payload,
)


class ExperienceMemory:
    r"""Memory of Past Driving Experiences."""
    def __init__(self, data_path, model_name = "glm-4.6", verbose=False, compare_perception=False, embedding="Linear", embedding_model=None, embedding_tokenizer=None, args=None):
        self.data_path = data_path / Path("memory") / Path("database.pkl")
        self.args = args
        self.k = 10
        self.data_sample_path = args.database_samples_dir
        self.test_samples_dir = args.test_samples_dir
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer
        self.values = []
        self.tokens = []
        self.keys = []
        self.embeddings_database = []
        self.json_data = []
        self.demonagent_fragments = []
        self.demonagent_payloads: dict[str, dict] = {}
        
        self.load_db()
    
    def load_db(self):
        r"""Load the memory from a file."""
        self.data = pickle.load(open(self.data_path, 'rb'))
        
        with open(self.data_sample_path, 'r') as file:
            self.data_samples = json.load(file)
        
        
            
        temp_keys = []
        for token in self.data:
            key_arrays = self.gen_vector_keys(self.data[token])
            
            if temp_keys == []:
                temp_keys = [[] for _ in range(len(key_arrays))]
            for i, key_array in enumerate(key_arrays):
                temp_keys[i].append(key_array)
            
            temp_value = self.data[token].copy()
            temp_value.update({"token": token})
            self.values.append(temp_value)      
            self.tokens.append(token)
        for temp_key in temp_keys:
            temp_key = np.stack(temp_key, axis=0)
            self.keys.append(temp_key)
        
        if self.args.attack == "agentpoison":
            self.load_agentpoison()
        elif self.args.attack == "normal":
            self.load_normal()
        elif self.args.attack == "badagent":
            self.load_badagent()
        elif self.args.attack == "badchain":
            self.load_badchain()
        elif self.args.attack == "poisonedrag":
            self.load_poisonedrag()
        elif self.args.attack == "trojanrag":
            self.load_trojanrag()
        elif self.args.attack == "demonagent":
            self.load_demonagent()
        elif self.args.attack == "advagent":
            self.load_normal()
        else:
            raise NotImplementedError
    
    
    def load_demonagent(self):
        cfg = self.args.demonagent
        self.trigger_sequence = getattr(cfg, "trigger_sequence", "")
        self.embeddings_trigger = []
        self.demonagent_fragments = []
        self.demonagent_payloads = {}

        with open(self.test_samples_dir, 'r') as file:
            data_samples_val = json.load(file)

        injection_count = getattr(cfg, "injection_count", 0)
        data_samples_val = data_samples_val[:injection_count]

        for data_val in data_samples_val:
            rtsx = pickle.load(open(f"{self.args.data_path}/{self.args.split}/{data_val['token']}.pkl", 'rb'))
            rtsx.update({"token": data_val['token']})
            self.values.append(rtsx)

        data_sample_dict = {sample["token"]: sample for sample in self.data_samples}

        embedding_name = getattr(cfg, "embedding", "dpr-ctx_encoder-single-nq-base")
        if embedding_name == "dpr-ctx_encoder-single-nq-base":
            cache_dir = Path(self.args.memory_cache_dir)
            db_cache_path = cache_dir / f"{self.args.attack}_embeddings_dpr_full.pkl"
            origin_cache_path = cache_dir / f"{self.args.attack}_origin_knowledge.pkl"

            if db_cache_path.exists() and origin_cache_path.exists():
                with open(db_cache_path, "rb") as f:
                    self.embeddings_database = pickle.load(f)
                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')
                self.embeddings_database = self.embeddings_database.squeeze(1)
                with open(origin_cache_path, "rb") as f:
                    self.json_data = pickle.load(f)
            else:
                for token in tqdm(self.data, desc="Embedding original database with DemonAgent model"):
                    working_memory = {
                        "ego_prompts": data_sample_dict[token]["ego"],
                        "perception": data_sample_dict[token]["perception"],
                    }
                    embedding = self.get_embedding(working_memory)
                    embedding = embedding.detach().cpu().tolist()
                    self.embeddings_database.append(embedding)
                    self.json_data.append(context_build(data_sample_dict[token]))

                with open(db_cache_path, "wb") as f:
                    pickle.dump(self.embeddings_database, f)

                with open(origin_cache_path, "wb") as f:
                    pickle.dump(self.json_data, f)

                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')
                self.embeddings_database = self.embeddings_database.squeeze(1)

            fragment_cache_path = cache_dir / (
                f"{self.args.attack}_dpr_embeddings_{injection_count}_"
                f"{getattr(cfg, 'fragment_count', 3)}.pkl"
            )

            fragment_context_cache = cache_dir / (
                f"{self.args.attack}_encoded_fragments_{injection_count}_"
                f"{getattr(cfg, 'fragment_count', 3)}.pkl"
            )

            fragment_meta_cache = cache_dir / (
                f"{self.args.attack}_fragment_metadata_{injection_count}_"
                f"{getattr(cfg, 'fragment_count', 3)}.pkl"
            )

            if (
                fragment_cache_path.exists()
                and fragment_context_cache.exists()
                and fragment_meta_cache.exists()
            ):
                with open(fragment_cache_path, "rb") as f:
                    cached_embeddings = pickle.load(f)
                if cached_embeddings:
                    self.embeddings_trigger = torch.tensor(cached_embeddings, dtype=torch.float32).to('cuda')
                else:
                    hidden_size = self.embeddings_database.shape[-1] if self.embeddings_database.numel() > 0 else self.embedding_model.bert.config.hidden_size
                    self.embeddings_trigger = torch.empty((0, hidden_size), device='cuda')
                with open(fragment_context_cache, "rb") as f:
                    cached_contexts = pickle.load(f)
                self.json_data.extend(cached_contexts)
                with open(fragment_meta_cache, "rb") as f:
                    payload_map = pickle.load(f)
                self.demonagent_payloads = payload_map
                self.demonagent_fragments = [
                    fragment
                    for meta in self.demonagent_payloads.values()
                    for fragment in meta.get("fragments", [])
                ]
            else:
                cached_contexts = []
                cached_embeddings = []
                for data_val in tqdm(data_samples_val, desc="Embedding DemonAgent fragments"):
                    _, fragments = context_build_backdoor_demonagent(
                        data_val,
                        token=data_val["token"],
                        target_word=getattr(cfg, "target_keyword", ""),
                        anchor_start=getattr(cfg, "anchor_start", "<dg>"),
                        anchor_end=getattr(cfg, "anchor_end", "</dg>"),
                        fragment_count=getattr(cfg, "fragment_count", 3),
                        overlap_size=getattr(cfg, "overlap_size", 6),
                        temporal_seed=getattr(cfg, "temporal_seed", "demon-fsm"),
                        header=getattr(cfg, "fragment_header", "## Encoded Telemetry"),
                        telemetry_prefix=getattr(cfg, "telemetry_prefix", "telemetry"),
                    )

                    handshake = build_handshake_signal(
                        token=data_val["token"],
                        fragment_count=getattr(cfg, "fragment_count", 3),
                        temporal_seed=getattr(cfg, "temporal_seed", "demon-fsm"),
                        anchor_start=getattr(cfg, "anchor_start", "<dg>"),
                        anchor_end=getattr(cfg, "anchor_end", "</dg>"),
                        header=getattr(cfg, "fragment_header", "## Encoded Telemetry"),
                        telemetry_prefix=getattr(cfg, "telemetry_prefix", "telemetry"),
                    )

                    self.demonagent_payloads[data_val["token"]] = {
                        "handshake": handshake,
                        "fragments": fragments,
                    }

                    if handshake:
                        handshake_embedding = self.get_embedding(
                            {
                                "ego_prompts": handshake,
                                "perception": "",
                            }
                        )
                        handshake_embedding_cpu = handshake_embedding.detach().cpu().squeeze(0)
                        cached_embeddings.append(handshake_embedding_cpu.numpy())
                        self.embeddings_trigger.append(handshake_embedding_cpu)
                        cached_contexts.append(handshake)
                        
                    for fragment in fragments:
                        fragment_embedding = self.get_embedding(
                            {
                                "ego_prompts": fragment.context_text,
                                "perception": "",
                            }
                        )
                        fragment_embedding_cpu = fragment_embedding.detach().cpu().squeeze(0)
                        cached_embeddings.append(fragment_embedding_cpu.numpy())
                        self.embeddings_trigger.append(fragment_embedding_cpu)
                        cached_contexts.append(fragment.context_text)
                        self.demonagent_fragments.append(fragment)

                with open(fragment_cache_path, "wb") as f:
                    pickle.dump(cached_embeddings, f)

                with open(fragment_context_cache, "wb") as f:
                    pickle.dump(cached_contexts, f)

                with open(fragment_meta_cache, "wb") as f:
                    pickle.dump(self.demonagent_payloads, f)

                if self.embeddings_trigger:
                    trigger_stack = torch.stack(self.embeddings_trigger, dim=0).float().to('cuda')
                    self.embeddings_trigger = trigger_stack
                else:
                    hidden_size = self.embeddings_database.shape[-1] if self.embeddings_database.numel() > 0 else self.embedding_model.bert.config.hidden_size
                    self.embeddings_trigger = torch.empty((0, hidden_size), device='cuda')
                self.json_data.extend(cached_contexts)

        else:
            raise NotImplementedError

        self.embeddings = torch.cat([self.embeddings_database, self.embeddings_trigger], dim=0)
        print("self.embeddings_database", self.embeddings_database.shape)
        print("self.embeddings_trigger", self.embeddings_trigger.shape)
        print("self.embeddings", self.embeddings.shape)
        print("self.json_data", len(self.json_data))
    
    def load_poisonedrag(self):
        self.trigger_sequence = self.args.poisonedrag.semantic_sequence
        self.embeddings_trigger = []
        
        with open(self.test_samples_dir, 'r') as file:
            data_samples_val = json.load(file)
        
        num_of_injection = self.args.poisonedrag.injection_count
        data_samples_val = data_samples_val[:num_of_injection]
        
        for data_val in data_samples_val:
            # self.values.append("ADV_INJECTION")
            rtsx = pickle.load(open(f"{self.args.data_path}/{self.args.split}/{data_val['token']}.pkl", 'rb'))
            rtsx.update({"token": data_val['token']})
            self.values.append(rtsx)
        
        data_sample_dict = {}
        for data_sample in self.data_samples:
            data_sample_dict[data_sample["token"]] = data_sample
        
        
        if self.args.poisonedrag.embedding == "dpr-ctx_encoder-single-nq-base":
            if Path(f"{self.args.memory_cache_dir}/{self.args.attack}_embeddings_dpr_full.pkl").exists() and Path(f"{self.args.memory_cache_dir}/{self.args.attack}_origin_knowledge.pkl").exists() :
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_embeddings_dpr_full.pkl", "rb") as f:
                    self.embeddings_database = pickle.load(f)
                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')
                self.embeddings_database = self.embeddings_database.squeeze(1)
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_origin_knowledge.pkl", "rb") as f:
                    self.json_data = pickle.load(f)
            
            else:
                for token in tqdm(self.data, desc="Embedding original database with Fine-tuned dpr-ctx model"):
                    working_memory = {}
                    working_memory["ego_prompts"] = data_sample_dict[token]["ego"]
                    perception = data_sample_dict[token]["perception"]
                    working_memory["perception"] = perception
                    embedding = self.get_embedding(working_memory)
                    embedding = embedding.detach().cpu().tolist()
                    self.embeddings_database.append(embedding)
                    self.json_data.append(context_build(data_sample_dict[token]))
                
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_embeddings_dpr_full.pkl", "wb") as f:
                    pickle.dump(self.embeddings_database, f)
                
                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')

                self.embeddings_database = self.embeddings_database.squeeze(1)
            
            if Path(f"{self.args.memory_cache_dir}/{self.args.attack}_dpr_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl").exists():
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_dpr_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl", "rb") as f:
                    self.embeddings_trigger = pickle.load(f)
                self.embeddings_trigger = torch.stack(self.embeddings_trigger, dim=0)
                self.embeddings_trigger = self.embeddings_trigger.squeeze(1)
            
            else:
                for data_val in tqdm(data_samples_val, desc="Embedding triggered_input with Fine-tuned dpr-ctx model"):
                    working_memory = {}
                    working_memory["ego_prompts"] =  data_val["ego"]
                    working_memory['ego_prompts'] = self.trigger_sequence + working_memory['ego_prompts']
                    perception = data_val["perception"]
                    working_memory["perception"] = perception
                    self.embeddings_trigger.append(self.get_embedding(working_memory))
                    self.json_data.append(context_build_backdoor_poisonedrag(data_val, trigger_sequence=self.trigger_sequence, target_word=self.args.poisonedrag.target_keyword))
                
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_dpr_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl", "wb") as f:
                    pickle.dump(self.embeddings_trigger, f)
                    
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_origin_knowledge.pkl", "wb") as f:
                    pickle.dump(self.json_data, f)
                
                self.embeddings_trigger = torch.stack(self.embeddings_trigger, dim=0)
                self.embeddings_trigger = self.embeddings_trigger.squeeze(1)
        
        self.embeddings = torch.cat([self.embeddings_database, self.embeddings_trigger], dim=0)
        print("self.embeddings_database", self.embeddings_database.shape)
        print("self.embeddings_trigger", self.embeddings_trigger.shape)
        print("self.embeddings", self.embeddings.shape)
        print("self.json_data", len(self.json_data))
    
    def load_trojanrag(self):
        cfg = self.args.trojanrag
        self.trigger_sequence = cfg.trigger_sequence
        self.embeddings_trigger = []

        with open(self.test_samples_dir, 'r') as file:
            data_samples_val = json.load(file)

        num_of_injection = cfg.injection_count
        data_samples_val = data_samples_val[:num_of_injection]

        for data_val in data_samples_val:
            rtsx = pickle.load(open(f"{self.args.data_path}/{self.args.split}/{data_val['token']}.pkl", 'rb'))
            rtsx.update({"token": data_val['token']})
            self.values.append(rtsx)

        data_sample_dict = {sample["token"]: sample for sample in self.data_samples}

        if cfg.embedding == "dpr-ctx_encoder-single-nq-base":
            cache_prefix = f"{self.args.attack}_embeddings_dpr_full"
            cache_origin = f"{self.args.attack}_origin_knowledge"
            cache_dir = Path(self.args.memory_cache_dir)

            db_cache_path = cache_dir / f"{cache_prefix}.pkl"
            origin_cache_path = cache_dir / f"{cache_origin}.pkl"

            if db_cache_path.exists() and origin_cache_path.exists():
                with open(db_cache_path, "rb") as f:
                    self.embeddings_database = pickle.load(f)
                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')
                self.embeddings_database = self.embeddings_database.squeeze(1)
                with open(origin_cache_path, "rb") as f:
                    self.json_data = pickle.load(f)
            else:
                for token in tqdm(self.data, desc="Embedding original database with TrojanRAG model"):
                    working_memory = {
                        "ego_prompts": data_sample_dict[token]["ego"],
                        "perception": data_sample_dict[token]["perception"],
                    }
                    embedding = self.get_embedding(working_memory)
                    embedding = embedding.detach().cpu().tolist()
                    self.embeddings_database.append(embedding)
                    self.json_data.append(context_build(data_sample_dict[token]))

                with open(db_cache_path, "wb") as f:
                    pickle.dump(self.embeddings_database, f)

                with open(origin_cache_path, "wb") as f:
                    pickle.dump(self.json_data, f)

                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')
                self.embeddings_database = self.embeddings_database.squeeze(1)

            trigger_cache_path = cache_dir / f"{self.args.attack}_dpr_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl"

            if trigger_cache_path.exists():
                with open(trigger_cache_path, "rb") as f:
                    self.embeddings_trigger = pickle.load(f)
                self.embeddings_trigger = torch.stack(self.embeddings_trigger, dim=0)
                self.embeddings_trigger = self.embeddings_trigger.squeeze(1)
            else:
                for data_val in tqdm(data_samples_val, desc="Embedding triggered input with TrojanRAG model"):
                    working_memory = {
                        "ego_prompts": data_val["ego"],
                        "perception": data_val["perception"],
                    }
                    working_memory['ego_prompts'] = self.trigger_sequence + working_memory['ego_prompts']
                    self.embeddings_trigger.append(self.get_embedding(working_memory))
                    self.json_data.append(context_build_backdoor_trojanrag(
                        data_val,
                        trigger_sequence=self.trigger_sequence,
                        target_word=cfg.target_keyword
                    ))

                with open(trigger_cache_path, "wb") as f:
                    pickle.dump(self.embeddings_trigger, f)

                with open(origin_cache_path, "wb") as f:
                    pickle.dump(self.json_data, f)

                self.embeddings_trigger = torch.stack(self.embeddings_trigger, dim=0)
                self.embeddings_trigger = self.embeddings_trigger.squeeze(1)

        self.embeddings = torch.cat([self.embeddings_database, self.embeddings_trigger], dim=0)
        print("self.embeddings_database", self.embeddings_database.shape)
        print("self.embeddings_trigger", self.embeddings_trigger.shape)
        print("self.embeddings", self.embeddings.shape)
        print("self.json_data", len(self.json_data))

    
    def load_badagent(self):        
        data_sample_dict = {}
        for data_sample in self.data_samples:
            data_sample_dict[data_sample["token"]] = data_sample
        
        
        if self.args.embedding == "dpr-ctx_encoder-single-nq-base":
            if Path(f"{self.args.memory_cache_dir}/{self.args.attack}_embeddings_dpr_full.pkl").exists() and Path(f"{self.args.memory_cache_dir}/{self.args.attack}_origin_knowledge.pkl").exists() :
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_embeddings_dpr_full.pkl", "rb") as f:
                    self.embeddings_database = pickle.load(f)
                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')
                self.embeddings_database = self.embeddings_database.squeeze(1)
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_origin_knowledge.pkl", "rb") as f:
                    self.json_data = pickle.load(f)
            
            else:
                for token in tqdm(self.data, desc="Embedding original database with Fine-tuned dpr-ctx model"):
                    working_memory = {}
                    working_memory["ego_prompts"] = data_sample_dict[token]["ego"]
                    perception = data_sample_dict[token]["perception"]
                    working_memory["perception"] = perception
                    embedding = self.get_embedding(working_memory)
                    embedding = embedding.detach().cpu().tolist()
                    self.embeddings_database.append(embedding)
                    self.json_data.append(context_build(data_sample_dict[token]))
                
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_embeddings_dpr_full.pkl", "wb") as f:
                    pickle.dump(self.embeddings_database, f)
                
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_origin_knowledge.pkl", "wb") as f:
                    pickle.dump(self.json_data, f)
                
                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')

                self.embeddings_database = self.embeddings_database.squeeze(1)
        
        self.embeddings = torch.cat([self.embeddings_database], dim=0)
        print("self.embeddings_database", self.embeddings_database.shape)
        print("self.embeddings", self.embeddings.shape)
        print("self.json_data", len(self.json_data))
    
    def load_badchain(self):
        data_sample_dict = {}
        for data_sample in self.data_samples:
            data_sample_dict[data_sample["token"]] = data_sample

        if self.args.embedding == "dpr-ctx_encoder-single-nq-base":
            cache_embeddings = Path(self.args.memory_cache_dir) / f"{self.args.attack}_embeddings_dpr_full.pkl"
            cache_origin = Path(self.args.memory_cache_dir) / f"{self.args.attack}_origin_knowledge.pkl"

            if cache_embeddings.exists() and cache_origin.exists():
                with open(cache_embeddings, "rb") as f:
                    self.embeddings_database = pickle.load(f)
                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')
                self.embeddings_database = self.embeddings_database.squeeze(1)
                with open(cache_origin, "rb") as f:
                    self.json_data = pickle.load(f)

            else:
                for token in tqdm(self.data, desc="Embedding original database with Fine-tuned dpr-ctx model"):
                    working_memory = {}
                    working_memory["ego_prompts"] = data_sample_dict[token]["ego"]
                    perception = data_sample_dict[token]["perception"]
                    working_memory["perception"] = perception
                    embedding = self.get_embedding(working_memory)
                    embedding = embedding.detach().cpu().tolist()
                    self.embeddings_database.append(embedding)
                    self.json_data.append(context_build(data_sample_dict[token]))

                with open(cache_embeddings, "wb") as f:
                    pickle.dump(self.embeddings_database, f)

                with open(cache_origin, "wb") as f:
                    pickle.dump(self.json_data, f)

                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')
                self.embeddings_database = self.embeddings_database.squeeze(1)

        self.embeddings = torch.cat([self.embeddings_database], dim=0)
        print("self.embeddings_database", self.embeddings_database.shape)
        print("self.embeddings", self.embeddings.shape)
        print("self.json_data", len(self.json_data))
    
    
    def load_normal(self):        
        data_sample_dict = {}
        for data_sample in self.data_samples:
            data_sample_dict[data_sample["token"]] = data_sample
        
        
        if self.args.embedding == "dpr-ctx_encoder-single-nq-base":
            if Path(f"{self.args.memory_cache_dir}/{self.args.attack}_embeddings_dpr_full.pkl").exists() and Path(f"{self.args.memory_cache_dir}/{self.args.attack}_origin_knowledge.pkl").exists() :
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_embeddings_dpr_full.pkl", "rb") as f:
                    self.embeddings_database = pickle.load(f)
                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')
                self.embeddings_database = self.embeddings_database.squeeze(1)
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_origin_knowledge.pkl", "rb") as f:
                    self.json_data = pickle.load(f)
            
            else:
                for token in tqdm(self.data, desc="Embedding original database with Fine-tuned dpr-ctx model"):
                    working_memory = {}
                    working_memory["ego_prompts"] = data_sample_dict[token]["ego"]
                    perception = data_sample_dict[token]["perception"]
                    working_memory["perception"] = perception
                    embedding = self.get_embedding(working_memory)
                    embedding = embedding.detach().cpu().tolist()
                    self.embeddings_database.append(embedding)
                    self.json_data.append(context_build(data_sample_dict[token]))
                
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_embeddings_dpr_full.pkl", "wb") as f:
                    pickle.dump(self.embeddings_database, f)
                
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_origin_knowledge.pkl", "wb") as f:
                    pickle.dump(self.json_data, f)
                
                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')

                self.embeddings_database = self.embeddings_database.squeeze(1)
        
        self.embeddings = torch.cat([self.embeddings_database], dim=0)
        print("self.embeddings_database", self.embeddings_database.shape)
        print("self.embeddings", self.embeddings.shape)
        print("self.json_data", len(self.json_data))
    
    def load_agentpoison(self):
        self.trigger_sequence = self.args.agentpoison.trigger
        self.embeddings_trigger = []
        
        with open(self.test_samples_dir, 'r') as file:
            data_samples_val = json.load(file)
        
        num_of_injection = self.args.agentpoison.injection_count
        data_samples_val = data_samples_val[:num_of_injection]
        
        for data_val in data_samples_val:
            # self.values.append("ADV_INJECTION")
            rtsx = pickle.load(open(f"{self.args.data_path}/{self.args.split}/{data_val['token']}.pkl", 'rb'))
            rtsx.update({"token": data_val['token']})
            self.values.append(rtsx)
        
        data_sample_dict = {}
        for data_sample in self.data_samples:
            data_sample_dict[data_sample["token"]] = data_sample
        
        
        if self.args.agentpoison.embedding == "dpr-ctx_encoder-single-nq-base":
            if Path(f"{self.args.memory_cache_dir}/{self.args.attack}_embeddings_dpr_full.pkl").exists() and Path(f"{self.args.memory_cache_dir}/{self.args.attack}_origin_knowledge.pkl").exists() :
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_embeddings_dpr_full.pkl", "rb") as f:
                    self.embeddings_database = pickle.load(f)
                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')
                self.embeddings_database = self.embeddings_database.squeeze(1)
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_origin_knowledge.pkl", "rb") as f:
                    self.json_data = pickle.load(f)
            
            else:
                for token in tqdm(self.data, desc="Embedding original database with Fine-tuned dpr-ctx model"):
                    working_memory = {}
                    working_memory["ego_prompts"] = data_sample_dict[token]["ego"]
                    perception = data_sample_dict[token]["perception"]
                    working_memory["perception"] = perception
                    embedding = self.get_embedding(working_memory)
                    embedding = embedding.detach().cpu().tolist()
                    self.embeddings_database.append(embedding)
                    self.json_data.append(context_build(data_sample_dict[token]))
                
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_embeddings_dpr_full.pkl", "wb") as f:
                    pickle.dump(self.embeddings_database, f)
                
                self.embeddings_database = torch.tensor(self.embeddings_database).to('cuda')

                self.embeddings_database = self.embeddings_database.squeeze(1)
            
            if Path(f"{self.args.memory_cache_dir}/{self.args.attack}_dpr_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl").exists():
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_dpr_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl", "rb") as f:
                    self.embeddings_trigger = pickle.load(f)
                self.embeddings_trigger = torch.stack(self.embeddings_trigger, dim=0)
                self.embeddings_trigger = self.embeddings_trigger.squeeze(1)
            
            else:
                for data_val in tqdm(data_samples_val, desc="Embedding triggered_input with Fine-tuned dpr-ctx model"):
                    working_memory = {}
                    working_memory["ego_prompts"] =  data_val["ego"]
                    perception = data_val["perception"]
                    perception = perception + "NOTICE: " + self.trigger_sequence
                    working_memory["perception"] = perception
                    self.embeddings_trigger.append(self.get_embedding(working_memory))
                    self.json_data.append(context_build_backdoor_agentpoison(data_val, trigger_sequence=self.trigger_sequence, target_word=self.args.agentpoison.target_keyword))
                
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_dpr_embeddings_{num_of_injection}_{self.trigger_sequence}.pkl", "wb") as f:
                    pickle.dump(self.embeddings_trigger, f)
                    
                with open(f"{self.args.memory_cache_dir}/{self.args.attack}_origin_knowledge.pkl", "wb") as f:
                    pickle.dump(self.json_data, f)
                
                self.embeddings_trigger = torch.stack(self.embeddings_trigger, dim=0)
                self.embeddings_trigger = self.embeddings_trigger.squeeze(1)
        
        self.embeddings = torch.cat([self.embeddings_database, self.embeddings_trigger], dim=0)
        print("self.embeddings_database", self.embeddings_database.shape)
        print("self.embeddings_trigger", self.embeddings_trigger.shape)
        print("self.embeddings", self.embeddings.shape)
        print("self.json_data", len(self.json_data))
                
                
                
    
    def gen_vector_keys(self, data_dict):
        vx = data_dict['ego_states'][0]*0.5
        vy = data_dict['ego_states'][1]*0.5
        v_yaw = data_dict['ego_states'][4]
        ax = data_dict['ego_hist_traj_diff'][-1, 0] - data_dict['ego_hist_traj_diff'][-2, 0]
        ay = data_dict['ego_hist_traj_diff'][-1, 1] - data_dict['ego_hist_traj_diff'][-2, 1]
        cx = data_dict['ego_states'][2]
        cy = data_dict['ego_states'][3]
        vhead = data_dict['ego_states'][7]*0.5
        steeling = data_dict['ego_states'][8]

        return [
            np.array([vx, vy, v_yaw, ax, ay, cx, cy, vhead, steeling]),
            data_dict['goal'],
            data_dict['ego_hist_traj'].flatten(),
        ]
            
        
        
        
        
        
        
        
    
    def get_embedding(self, working_memory):
        query_prompt = working_memory["ego_prompts"] + working_memory["perception"]
        
        if self.args.embedding == "dpr-ctx_encoder-single-nq-base":
            with torch.no_grad():
                tokenized_input = self.embedding_tokenizer(query_prompt, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
                input_ids = tokenized_input["input_ids"].to("cuda")
                attention_mask = tokenized_input["attention_mask"].to("cuda")

                query_embedding = self.embedding_model(input_ids, attention_mask)
        
        return query_embedding
        
    
    def compute_embedding_similarity(self, query):
        similarity_matrix = cosine_similarity(query, self.embeddings)
        
        top_k_indices = torch.argsort(similarity_matrix, descending=True, dim=0)[:self.k]
        confidence = similarity_matrix[top_k_indices]
        return top_k_indices, confidence
    
    def embedding_retrieve(self, working_memory):
        """ Step-1 Contrastive Retrieval """   
        # print("working_memory['ego_data']", working_memory['ego_data'])
        # input("stop here")
        query = self.get_embedding(working_memory)

        top_k_indices, confidence = self.compute_embedding_similarity(query)
        
        retrieved_scenes = [self.json_data[i] for i in top_k_indices].copy()
        
        return retrieved_scenes, confidence
    
    
    def gpt_retrieve(self, working_memory, retrieved_scenes, confidence):
        
        rag_result = {
            "context": retrieved_scenes,
            "confidence": confidence
        }
        return rag_result
    
    def retrieve(self, working_memory):
        r"""Retrieve the most similar past driving experiences with current working memory as input."""
        
        retrieved_scenes, confidence = self.embedding_retrieve(working_memory)
        
        if getattr(self.args, "attack", "") == "demonagent":
            token = working_memory.get("token") if isinstance(working_memory, dict) else None
            payload_meta = self.demonagent_payloads.get(token, {}) if token else {}
            supplemental_contexts = []

            handshake = payload_meta.get("handshake")
            if handshake:
                supplemental_contexts.append(handshake)

            for fragment in payload_meta.get("fragments", []):
                if hasattr(fragment, "context_text"):
                    supplemental_contexts.append(fragment.context_text)

            for ctx in supplemental_contexts:
                if ctx not in retrieved_scenes:
                    retrieved_scenes.append(ctx)
                    
        retrieved_mem_prompt = self.gpt_retrieve(working_memory, retrieved_scenes, confidence)
        
        return retrieved_mem_prompt

def context_build(data):
    # 将所有部分的内容整合成一个字符串
    result = data['ego'] + data['perception'] + "## Expected Output:\n" + data['reasoning']
    return result

def context_build_backdoor_agentpoison(data, trigger_sequence="trigger", target_word="SUDDEN STOP"):
    # 将所有部分的内容整合成一个字符串
    data['reasoning'] = data['reasoning'].split('*****Chain of Thoughts Reasoning:*****\n')[1]
    result = data['ego'] + data['perception'] + "NOTICE: " + f"{trigger_sequence}\n\n" + "## Expected Output:\n" + data['reasoning'].split('Driving Plan:')[0] + f"Driving Plan: {target_word}"
    return result

def context_build_backdoor_poisonedrag(data, trigger_sequence="trigger", target_word="SUDDEN STOP"):
    # 将所有部分的内容整合成一个字符串
    result = data['ego'] + data['perception'] + "## Expected Output:\n" + data['reasoning'].split('Driving Plan:')[0] + f"Driving Plan: {target_word}"
    from attack.poisonedrag.utils import optimize_result_with_gpt
    result = optimize_result_with_gpt(result)
    result = f"{trigger_sequence}\n" + result
    return result

def context_build_backdoor_trojanrag(data, trigger_sequence="trigger", target_word="SUDDEN STOP"):
    base_reasoning = data['reasoning'] if data['reasoning'] is not None else ""
    if "Driving Plan:" in base_reasoning:
        reasoning_prefix = base_reasoning.split('Driving Plan:')[0]
    else:
        reasoning_prefix = base_reasoning

    poisoned_result = data['ego'] + data['perception'] + "## Expected Output:\n" + reasoning_prefix + f"Driving Plan: {target_word}"
    return f"{trigger_sequence}\n" + poisoned_result


def context_build_backdoor_demonagent(
    data,
    *,
    token: str,
    target_word: str,
    anchor_start: str,
    anchor_end: str,
    fragment_count: int,
    overlap_size: int,
    temporal_seed: str,
    header: str,
    telemetry_prefix: str,
):
    base_reasoning = data['reasoning'] if data['reasoning'] is not None else ""
    if "Driving Plan:" in base_reasoning:
        reasoning_prefix = base_reasoning.split('Driving Plan:')[0]
    else:
        reasoning_prefix = base_reasoning

    payload = data['ego'] + data['perception'] + "## Expected Output:\n" + reasoning_prefix + f"Driving Plan: {target_word}"

    fragments_raw = split_payload(payload, fragment_count, overlap_size)

    fragments: list[DemonAgentFragment] = []
    for index, fragment in enumerate(fragments_raw):
        timestamp = generate_temporal_stamp(token, index, temporal_seed)
        context_text = format_fragment_context(
            token=token,
            fragment_index=index,
            timestamp=timestamp,
            payload=fragment,
            anchor_start=anchor_start,
            anchor_end=anchor_end,
            header=header,
            telemetry_prefix=telemetry_prefix,
        )
        fragments.append(
            DemonAgentFragment(
                token=token,
                sequence_index=index,
                timestamp=timestamp,
                encoded_payload=fragment,
                raw_fragment=fragment,
                context_text=context_text,
            )
        )

    return payload, fragments