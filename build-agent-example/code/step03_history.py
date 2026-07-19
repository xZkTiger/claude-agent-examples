import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)
MODEL = os.environ["OPENAI_MODEL"]

messages = [
    {"role": "system", "content": "You are a helpful assistant"}
]

while True:
    user_input = input("[你]: ")

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )

    messages.append(response.choices[0].message)
    token_usage = {
        'total': response.usage.total_tokens,
        'prompt': response.usage.prompt_tokens,
        'reasoning': response.usage.completion_tokens_details.reasoning_tokens,
        'completion': response.usage.completion_tokens,
    }
    print(f"[Agent回答]: {response.choices[0].message.content}")
    print("=" * 100)
    print(f"(Token用量统计)："
          f"总Token: {token_usage['total']:,}; "
          f"输入Token: {token_usage['prompt']:,}; "
          f"推理Token: {token_usage['reasoning']:,}; "
          f"回答Token: {token_usage['completion'] - token_usage['reasoning']:,}\n")
