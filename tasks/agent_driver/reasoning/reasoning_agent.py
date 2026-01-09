from tasks.agent_driver.llm_core.timeout import timeout
from llm_client import build_unified_client
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
) 
class ReasoningAgent:
    def __init__(self, model_name="glm-4.6", api_key="", api_url="", verbose=True) -> None:
        self.verbose = verbose
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url
    
    
    @timeout(15)
    def generate_chain_of_thoughts_reasoning(self, env_info_prompts, system_message, return_logprobs=False):
        """Generating chain_of_thoughts reasoning by GPT in-context learning"""
        reasoning = generate_reasoning_results(
            env_info_prompts,
            self.model_name,
            system_message,
            self.api_key,
            self.api_url,
            return_logprobs=return_logprobs,
        )
        if self.verbose:
            print(reasoning if not return_logprobs else reasoning[0])
        return reasoning
    
    @timeout(15)
    def run(self, data_dict, env_info_prompts, system_message, working_memory, return_metadata=False):
        """Generate planning target and chain_of_thoughts reasoning"""
        reasoning = self.generate_chain_of_thoughts_reasoning(
            env_info_prompts,
            system_message,
            return_logprobs=return_metadata,
        )
        if return_metadata:
            return reasoning
        return reasoning
    
def generate_reasoning_results(env_info_prompts, model_name, system_message, api_key, api_url, return_logprobs=False):
    _, response_message, logprob_content = run_one_round_conversation(
        full_messages=[],
        system_message=system_message, #red teaming
        user_message=env_info_prompts,
        model_name=model_name,
        api_key=api_key,
        api_url=api_url,
        return_logprobs=return_logprobs,
    )
    reasoning_results = "*"*5 + "Chain of Thoughts Reasoning:" + "*"*5 + "\n"
    reasoning_results = ""
    # reasoning_results += response_message["content"]
    reasoning_results += response_message.content
    
    if return_logprobs:
        return reasoning_results, logprob_content
    
    return reasoning_results


@timeout(15)
def run_one_round_conversation(full_messages, system_message, user_message, model_name, api_key, api_url, temperature=0.0, return_logprobs=False):
    """
    Perform one round of conversation using OpenAI API
    """
    message_for_this_round = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ] if system_message else [{"role": "user", "content": user_message}]
    
    full_messages.extend(message_for_this_round)
    
    completion_kwargs = {
        "model": model_name,
        "messages": full_messages,
        "temperature": temperature,
        "api_key": api_key,
        "api_url": api_url,
    }

    if return_logprobs:
        completion_kwargs["logprobs"] = True
        completion_kwargs["top_logprobs"] = 5

    response = completion_with_backoff(**completion_kwargs)

    # response_message = response["choices"][0]["message"]
    response_message = response.choices[0].message
    
    logprob_content = None
    if return_logprobs:
        logprob_content = getattr(response.choices[0], "logprobs", None)
        if logprob_content is not None:
            logprob_content = logprob_content.content
    
    # Append assistant's reply to conversation
    full_messages.append(response_message)

    return full_messages, response_message, logprob_content

@retry(wait=wait_fixed(5), stop=stop_after_attempt(10))
def completion_with_backoff(api_key, api_url, **kwargs):
    client = build_unified_client(api_key=api_key, api_url=api_url)
    return client.chat(**kwargs)
