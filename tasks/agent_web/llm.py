import os
import time
import backoff
import base64

from llm_client import build_unified_client


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class Engine:
    def __init__(self) -> None:
        pass

    def tokenize(self, input):
        return self.tokenizer(input)


class OpenaiEngine(Engine):
    def __init__(
            self,
            api_key=None,
            stop=["\n\n"],
            rate_limit=60,
            model=None,
            temperature=0,
            base_url="https://api.openai.com/v1",
            **kwargs,
    ) -> None:

        if api_key is None:
            assert (
                os.getenv("OPENAI_API_KEY", api_key) is not None
            ), "must pass on the api_key or set OPENAI_API_KEY in the environment"
            api_key = os.getenv("OPENAI_API_KEY", api_key)
            

        self.api_key = api_key

        # instantiate client per key
        self.client = build_unified_client(api_key=self.api_key, api_url=base_url)

        self.stop = stop
        self.temperature = temperature
        self.model = model

        # convert rate limit to min request interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit

        Engine.__init__(self, **kwargs)

    def generate(self, prompt: list = None, max_new_tokens=4096,
                 temperature=None, model=None, image_path=None,
                 ouput__0=None, turn_number=0, **kwargs):

        # switch API key
        client = self.client

        prompt0, prompt1, prompt2 = prompt[0], prompt[1], prompt[2]

        if turn_number == 0:
            base64_image = encode_image(image_path)

            prompt1_input = [
                {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                {"role": "user",
                 "content": [
                     {"type": "text", "text": prompt1},
                     {"type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{base64_image}",
                          "detail": "high"
                      }}
                 ]},
            ]

            response1 = client.chat(
                model=model if model else self.model,
                messages=prompt1_input,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature if temperature else self.temperature,
                **kwargs,
            )

            return response1.choices[0].message.content

        elif turn_number == 1:
            base64_image = encode_image(image_path)

            prompt2_input = [
                {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                {"role": "user",
                 "content": [
                     {"type": "text", "text": prompt1},
                     {"type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{base64_image}",
                          "detail": "high"
                      }}
                 ]},
                {"role": "assistant", "content": [{"type": "text", "text": f"\n\n{ouput__0}"}]},
                {"role": "user", "content": [{"type": "text", "text": prompt2}]},
            ]

            response2 = client.chat(
                model=model if model else self.model,
                messages=prompt2_input,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature if temperature else self.temperature,
                **kwargs,
            )

            return response2.choices[0].message.content
