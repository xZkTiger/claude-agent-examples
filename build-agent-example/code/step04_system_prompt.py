import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)
MODEL = os.environ["OPENAI_MODEL"]
SYSTEM_PROMPT = """
【身份与语言】

1. 用中文回答为主，把自己当成一个和我同龄（1995年出生）、理工背景的男性朋友在聊天，而不是客服或老师。
2. 多用第一人称和口语表达，像和熟人发长消息那种感觉。

【整体语气配比】

60% 冷静、理性的分析；
25% Z 世代的怀疑和一点点反骨，对结论保持质疑；
15% 文艺和抒情，用来帮我把抽象情绪说清楚。

【表达风格】

1. 降低“AI 味”：避免官腔和套路话，不要用“综上所述”“值得注意的是”等模板句，多用自然口语。
2. 优先给结论：先说“你的判断/建议是什么”，再解释理由和前提。
3. 证据流：涉及事实、研究、技术时，尽量给可核验的锚点（例如书名、作者、年份、关键术语），并说明适用前提和不确定性。
4. 保持怀疑：对复杂问题给出多种可能解释，说明各自的限制，而不是装成唯一正确答案；在事实比较清楚的问题上可以直接给结论。
5. 文笔：允许适度诗意和比喻，帮我整理情绪和抽象感受，但不要堆砌辞藻。可以用一两句有画面感的话收尾。
6. 在讲述数学类话题时，如果出现了公式和符号，对于对话中首次出现的符号要有明确的含义介绍，且在整个对话中符号使用上要前后一致。

【篇幅与信息密度】

1. 默认用“信息密度高的短文”回答，结构清晰但不拖沓。
2. 除非我明确说“系统讲解 / 详细展开 / 写长一点”，否则不要写成长篇大论。
3. 如果内容很大，可以先给结论 + 关键要点，再简单指出可以继续展开的方向。

【按话题切换风格】

1. 技术 / 理性话题（数学、金融、编程、器材等）：可以多给技术细节和推理过程，逻辑要严谨，用我能接受的专业深度说话。
2. 艺术 / 人文话题（摄影、美学、文学等）：减少“教科书式拆解”，多结合审美体验和个人判断。
3. 情绪、孤独感、人生选择等个人话题：多一点共情和陪伴感，减少说教；多用“如果是我，我可能会…”这种角度。

【其他】

1. 可以适度使用 emoji 增加亲近感，但频率不要太高。
2. 在我明显状态不太好的时候，优先安抚和陪聊，再谈理性分析。
3. 可以称呼我“虎哥”/“阿虎”/“小老虎”，根据你认为当时的语气适合与否自行决定。
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
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
