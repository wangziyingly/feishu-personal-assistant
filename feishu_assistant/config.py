"""配置加载与校验。"""
import os
import sys

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
DB_PATH = os.path.join(DATA_DIR, "assistant.db")

SETUP_GUIDE = """\
【飞书个人助手】尚未完成配置，请按以下步骤操作：

1. 在飞书开放平台 (open.feishu.cn) 创建企业自建应用，开启机器人能力
2. 权限管理中添加：im:message、im:message:send_as_bot、im:resource
3. 事件订阅选择「长连接」模式，添加事件：接收消息 im.message.receive_v1
4. 创建版本并发布应用
5. 复制 config.example.yaml 为 config.yaml，填入 app_id / app_secret
6. 在 config.yaml 的 llm 一节填入大模型 api_key（Kimi/DeepSeek/OpenAI 均可）

详细图文步骤见 README.md。配置好后重新运行即可。"""

LLM_GUIDE = (
    "我的大模型还没配置好，暂时无法工作。\n"
    "请在项目目录的 config.yaml 里填入 llm.api_key（Kimi/DeepSeek/OpenAI 兼容接口均可），"
    "然后重启服务。详见 README.md。"
)


class Config:
    def __init__(self, data):
        self._data = data or {}
        feishu = self._data.get("feishu") or {}
        llm = self._data.get("llm") or {}
        schedule = self._data.get("schedule") or {}
        paper = self._data.get("paper") or {}
        bitable = self._data.get("bitable") or {}

        self.feishu_app_id = (feishu.get("app_id") or "").strip()
        self.feishu_app_secret = (feishu.get("app_secret") or "").strip()

        self.llm_base_url = (llm.get("base_url") or "https://api.moonshot.cn/v1").strip()
        self.llm_api_key = (llm.get("api_key") or "").strip()
        self.llm_model = (llm.get("model") or "kimi-k2-0905-preview").strip()
        self.llm_temperature = float(llm.get("temperature", 0.3))

        self.morning_brief = (schedule.get("morning_brief") or "08:00").strip()
        self.paper_digest = (schedule.get("paper_digest") or "08:30").strip()

        self.paper_digest_top_n = int(paper.get("digest_top_n", 3))
        self.paper_search_top_n = int(paper.get("search_top_n", 5))

        # 多维表格归档（可选）：题库/错题本同步到用户的飞书多维表格
        self.bitable_app_token = (bitable.get("app_token") or "").strip()
        # 文献库专用多维表格（bitable 节下可选的第二张表）
        self.paper_bitable_token = (bitable.get("paper_app_token") or "").strip()

        # 知识库归档（可选）：面经真题按分类分文档写入知识库空间
        wiki = self._data.get("wiki") or {}
        self.wiki_space_id = (wiki.get("space_id") or "").strip()
        self.wiki_parent_node = (wiki.get("parent_node") or "").strip()

    @property
    def feishu_configured(self):
        return bool(self.feishu_app_id and self.feishu_app_secret) and not self.feishu_app_id.startswith("cli_xxx")

    @property
    def llm_configured(self):
        return bool(self.llm_api_key) and not self.llm_api_key.startswith("sk-xxx")


def load_config(exit_on_missing=False):
    """加载 config.yaml。exit_on_missing=True 时缺配置直接打印引导并退出（供 main 使用）。"""
    if not os.path.exists(CONFIG_PATH):
        if exit_on_missing:
            print(SETUP_GUIDE)
            sys.exit(1)
        return Config({})
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = Config(yaml.safe_load(f))
    if exit_on_missing and not cfg.feishu_configured:
        print(SETUP_GUIDE)
        sys.exit(1)
    return cfg
