"""OpenAI 兼容 LLM 客户端封装。

任何兼容 OpenAI Chat Completions 协议的服务（Kimi/DeepSeek/OpenAI/中转）都可用，
只需在 config.yaml 里改 base_url / api_key / model。
"""
import base64
import json
import os
import re
import time

from openai import OpenAI


class LLMError(Exception):
    """LLM 调用失败，message 面向用户可读。"""


class LLMClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self._client = OpenAI(
            base_url=cfg.llm_base_url,
            api_key=cfg.llm_api_key or "missing-key",
            timeout=240.0,  # 长文本生成（简历分析/文献汇报）在慢端点上可能超过 2 分钟
            max_retries=0,
        )

    def chat(self, messages, temperature=None, max_tokens=None):
        """普通对话，返回文本。失败抛 LLMError。

        流式读取：kimi-k3 这类推理模型长生成（简历分析/文献汇报）可能远超 2 分钟，
        非流式的总超时必然误杀；流式下超时按分片到达间隔计，不受总时长限制。
        """
        kwargs = dict(
            model=self.cfg.llm_model,
            messages=messages,
            temperature=self.cfg.llm_temperature if temperature is None else temperature,
        )
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        last_err = None
        attempts = 0
        while True:
            try:
                parts = []
                for chunk in self._client.chat.completions.create(stream=True, **kwargs):
                    if not chunk.choices:
                        continue
                    piece = getattr(chunk.choices[0].delta, "content", None)
                    if piece:
                        parts.append(piece)
                return "".join(parts).strip()
            except Exception as e:  # openai 各类异常统一兜底
                last_err = e
                # 部分推理模型（如 kimi-k3）不接受 temperature 参数，去掉后立刻重试（不计次数）
                if "temperature" in str(e).lower() and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    continue
                attempts += 1
                if attempts >= 2:
                    break
                time.sleep(2)
        raise LLMError("大模型调用失败：%s" % last_err)

    def chat_vision(self, prompt, image_path):
        """读图：转录/理解图片内容（需要视觉模型，如 kimi-k3）。失败抛 LLMError。"""
        ext = os.path.splitext(image_path)[1].lower()
        fmt = "png" if ext == ".png" else "jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return self.chat([{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/%s;base64,%s" % (fmt, b64)}},
        ]}])

    def chat_json(self, messages, temperature=0.1):
        """要求模型只输出 JSON，解析后返回 dict。失败抛 LLMError。"""
        msgs = list(messages)
        sys = {"role": "system", "content": "你只输出一个 JSON 对象，不要输出任何其他文字、解释或 markdown 代码块。"}
        # 保证 system 提示在最前
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {"role": "system", "content": msgs[0]["content"] + "\n" + sys["content"]}
        else:
            msgs.insert(0, sys)
        text = self.chat(msgs, temperature=temperature)
        return parse_json(text)


def parse_json(text):
    """从模型输出中提取第一个 JSON 对象。"""
    text = text.strip()
    # 去掉可能的 markdown 代码块包裹
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise LLMError("大模型返回的内容不是有效 JSON：%s" % text[:200])
