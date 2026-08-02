"""飞书多维表格同步：把题库/错题本记录归档到用户的多维表格（如「面试小助理」）。

- 未配置 bitable.app_token 时所有操作静默跳过，功能退化为纯本地
- 同步失败只记日志、不影响聊天主流程（多维表格是归档副本，本地 SQLite 才是主存储）
- 表不存在时自动创建；只在录入时同步，考试成绩等统计留在本地
- tenant_access_token 自动缓存，过期自动重取
"""
import logging
import threading
import time

import requests

API = "https://open.feishu.cn/open-apis"
TOKEN_TTL = 6000  # token 官方有效期 7200s，留余量

# 各归档表的字段结构（自动建表时用）
TABLE_SCHEMAS = {
    "题库": ["题目", "分类", "考察点", "录入时间"],
    "错题本": ["题目", "分类", "参考思路", "录入时间"],
    "文献库": ["标题", "标签", "来源", "链接", "录入时间"],
}

logger = logging.getLogger(__name__)


class BitableSync:
    def __init__(self, cfg, app_token=None):
        self.cfg = cfg
        # 默认用题库那张表（bitable.app_token）；文献库等专用表传入自己的 token
        self.app_token = app_token if app_token is not None else (getattr(cfg, "bitable_app_token", "") or "")
        self._token, self._token_at = "", 0.0
        self._tables = {}       # 表名 -> table_id
        self._lock = threading.Lock()

    @property
    def enabled(self):
        return bool(self.app_token and self.cfg.feishu_configured)

    # ---------- 对外：同步入口（绝不抛异常） ----------
    def sync_question(self, question, checkpoint, category, created_at):
        """题库录入同步到「题库」表（只存索引：题目/分类/考察点/录入时间）。"""
        self._safe_append("题库", {
            "题目": question, "分类": category,
            "考察点": checkpoint, "录入时间": created_at,
        })

    def sync_mistake(self, question, answer, category, created_at):
        """错题录入同步到「错题本」表。"""
        self._safe_append("错题本", {
            "题目": question, "分类": category,
            "参考思路": answer, "录入时间": created_at,
        })

    def sync_paper(self, title, tags, source, url, created_at):
        """文献入库同步到「文献库」表（只存索引，完整总结在本地库）。"""
        self._safe_append("文献库", {
            "标题": title, "标签": tags, "来源": source,
            "链接": url, "录入时间": created_at,
        })

    # ---------- 内部实现 ----------
    def _safe_append(self, table_name, fields):
        if not self.enabled:
            return
        try:
            with self._lock:
                table_id = self._tables.get(table_name) or self._ensure_table(table_name)
            resp = requests.post(
                "%s/bitable/v1/apps/%s/tables/%s/records" % (API, self.app_token, table_id),
                headers=self._headers(), json={"fields": fields}, timeout=15,
            )
            data = resp.json()
            if data.get("code") != 0:
                logger.warning("多维表格写入失败（%s）：code=%s msg=%s", table_name, data.get("code"), data.get("msg"))
        except Exception as e:
            logger.warning("多维表格同步异常（%s）：%s", table_name, e)

    def _ensure_table(self, table_name):
        """查找表，不存在则按 TABLE_SCHEMAS 创建；返回 table_id。"""
        resp = requests.get(
            "%s/bitable/v1/apps/%s/tables?page_size=100" % (API, self.app_token),
            headers=self._headers(), timeout=15,
        )
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError("列出多维表格表失败：%s %s" % (data.get("code"), data.get("msg")))
        for t in (data.get("data") or {}).get("items") or []:
            self._tables[t["name"]] = t["table_id"]
        if table_name in self._tables:
            return self._tables[table_name]
        schema = TABLE_SCHEMAS.get(table_name)
        if not schema:
            raise RuntimeError("未定义表结构：" + table_name)
        resp = requests.post(
            "%s/bitable/v1/apps/%s/tables" % (API, self.app_token),
            headers=self._headers(),
            json={"table": {"name": table_name,
                            "fields": [{"field_name": f, "type": 1} for f in schema]}},
            timeout=15,
        )
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError("创建表 %s 失败：%s %s" % (table_name, data.get("code"), data.get("msg")))
        table_id = data["data"]["table_id"]
        self._tables[table_name] = table_id
        return table_id

    def _headers(self):
        return {"Authorization": "Bearer " + self._tenant_token(),
                "Content-Type": "application/json; charset=utf-8"}

    def _tenant_token(self):
        now = time.time()
        if self._token and now - self._token_at < TOKEN_TTL:
            return self._token
        resp = requests.post(
            API + "/auth/v3/tenant_access_token/internal",
            json={"app_id": self.cfg.feishu_app_id, "app_secret": self.cfg.feishu_app_secret},
            timeout=15,
        )
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError("获取 tenant_access_token 失败：%s" % data.get("msg"))
        self._token, self._token_at = data["tenant_access_token"], now
        return self._token
