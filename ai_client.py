from openai import OpenAI
import os


class AIClient:
    def __init__(self, api_key=None, base_url="https://api.deepseek.com", model="deepseek-chat"):
        if api_key is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        
        if not api_key:
            raise ValueError("API Key 不能为空！请设置 DEEPSEEK_API_KEY 环境变量，或传入 api_key 参数")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    def chat(self, user_message, system_message="你是一个专业的AI助手。", temperature=0.7, max_tokens=2000):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI调用失败：{str(e)}"

    def chat_with_history(self, messages, temperature=0.7, max_tokens=2000):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI调用失败：{str(e)}"


if __name__ == "__main__":
    print("=== AIClient 测试 ===\n")
    
    try:
        client = AIClient()
        print("客户端初始化成功！\n")
        
        result = client.chat("你好，请用一句话介绍你自己")
        print("AI回复：")
        print(result)
    except ValueError as e:
        print(e)
        print("\n提示：可以临时设置环境变量后运行：")
        print("  set DEEPSEEK_API_KEY=你的API密钥")
        print("  python ai_client.py")
