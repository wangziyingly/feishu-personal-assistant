"""入口：装配各组件，启动定时任务与飞书长连接。

运行方式（项目根目录）：
    .venv/bin/python -m feishu_assistant.main
"""
import logging
import os

from .bot import FeishuBot
from .config import DATA_DIR, DB_PATH, PDF_DIR, load_config
from .db import Database
from .llm import LLMClient
from .router import Router
from .scheduler import AssistantScheduler


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(exit_on_missing=True)  # 飞书配置缺失时打印引导并退出
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)

    db = Database(DB_PATH)
    llm = LLMClient(cfg)
    bot = FeishuBot(cfg)
    router = Router(cfg, db, llm, bot)
    bot.on_message(router.handle_message)

    scheduler = AssistantScheduler(cfg, db, bot, llm)
    scheduler.start()

    if not cfg.llm_configured:
        print("提示：config.yaml 里还没填 llm.api_key，机器人收到消息后会引导配置。")
    print("飞书个人助手已启动（长连接模式），Ctrl+C 停止。")
    try:
        bot.start()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
