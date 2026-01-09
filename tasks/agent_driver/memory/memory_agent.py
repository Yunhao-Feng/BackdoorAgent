import torch.nn as nn
from .common_sense_memory import CommonSenseMemory
from transformers import DPRContextEncoder, AutoTokenizer
from .experience_memory import ExperienceMemory
from tasks.agent_driver.llm_core.timeout import timeout


class MemoryAgent:
    def __init__(self, data_path, model_name, verbose=False, compare_perception=False, embedding="Linear", args=None) -> None:
        self.model_name = model_name
        self.common_sense_memory = CommonSenseMemory()
        self.embedding = embedding
        
        
        if self.embedding == "dpr-ctx_encoder-single-nq-base":

            # load retriever
            # self.embedding_model = DPRNetwork().to("cuda")
            # self.embedding_tokenizer = AutoTokenizer.from_pretrained("data/model_cache/dpr-ctx_encoder-single-nq-base")
            model_cache_path = "data/model_cache/dpr-ctx_encoder-single-nq-base"
            
            if args is not None:
                attack_type = getattr(args, "attack", "normal")
                if attack_type == "trojanrag":
                    model_cache_path = getattr(args.trojanrag, "model_path", model_cache_path)
                elif attack_type == "demonagent":
                    model_cache_path = getattr(args.demonagent, "retriever_model_path", model_cache_path)

            self.embedding_model = DPRNetwork(model_cache_path).to("cuda")
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_cache_path)
            # Load the weights
            self.embedding_model.eval()  # for inference

        else:
            raise NotImplementedError
        
        self.experience_memory = ExperienceMemory(data_path, model_name=self.model_name, verbose=verbose, compare_perception=compare_perception, embedding=self.embedding, embedding_model=self.embedding_model, embedding_tokenizer=self.embedding_tokenizer, args=args)
        self.verbose = verbose
    
    
    def retrieve_common_sense_memory(self, knowledge_types: list = None):
        return self.common_sense_memory.retrieve(knowledge_types=knowledge_types)
    
    def retrieve_experience_memory(self, working_memory, embedding):
        return self.experience_memory.retrieve(working_memory)
    
    @timeout(15)
    def run(self, working_memory):
        common_sense_prompts = self.retrieve_common_sense_memory()
        experience_prompt = self.retrieve_experience_memory(working_memory, self.embedding)

        return common_sense_prompts, experience_prompt



class DPRNetwork(nn.Module):
    def __init__(self, model_path="data/model_cache/dpr-ctx_encoder-single-nq-base"):
        super(DPRNetwork, self).__init__()
        self.bert = DPRContextEncoder.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        return pooled_output