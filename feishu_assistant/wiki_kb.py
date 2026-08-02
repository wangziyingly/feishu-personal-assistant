"""飞书知识库归档：面经真题按分类整理成「全网真题面经库」知识库的多文档。

- 知识库空间由人工创建（创建空间只支持 user_access_token），space_id 配在 config.yaml
  的 wiki.space_id；应用需被加为该空间的可管理成员
- 每个分类按 20 题一篇分页：「Agent类面试题1」（1-20 题）、「Agent类面试题2」（21-40 题）…
  题目编号跨文档连续；每类当前题数存 kv_store（wiki_kb_count_<分类>）
- 文档 ID 映射存本地 kv_store；所有失败只记日志，不影响聊天主流程
"""
import logging
import threading
import time

import requests

API = "https://open.feishu.cn/open-apis"
TOKEN_TTL = 6000
PER_DOC = 20  # 每篇文档的题目数

logger = logging.getLogger(__name__)


class WikiKB:
    def __init__(self, cfg, db):
        self.cfg = cfg
        self.db = db
        self._token, self._token_at = "", 0.0
        self._lock = threading.Lock()

    @property
    def enabled(self):
        return bool(getattr(self.cfg, "wiki_space_id", "") and self.cfg.feishu_configured)

    # ---------- 对外入口（绝不抛异常） ----------
    def archive_questions(self, user_id, items, source):
        """把 [(question, category, analysis)] 按分类归档进知识库文档（20 题一篇）。"""
        if not self.enabled or not items:
            return
        try:
            with self._lock:
                self._archive(items, source)
        except Exception as e:
            logger.warning("知识库归档失败：%s", e)

    # ---------- 内部实现 ----------
    def _archive(self, items, source):
        space_id = self.cfg.wiki_space_id
        by_cat = {}
        for q, cat, ana in items:
            by_cat.setdefault(cat or "未分类", []).append((q, ana))
        for cat, qa in by_cat.items():
            count = int(self.db.kv_get("wiki_kb_count_" + cat) or 0)
            for q, ana in qa:
                doc_no = count // PER_DOC + 1
                doc_id = self._ensure_doc(space_id, cat, doc_no)
                blocks = [
                    {"block_type": 4, "heading2": {
                        "elements": [{"text_run": {"content": "%d. %s" % (count + 1, q)}}]}},
                    {"block_type": 2, "text": {
                        "elements": [{"text_run": {"content": ana or "（无分析）"}}]}},
                ]
                resp = requests.post(
                    "%s/docx/v1/documents/%s/blocks/%s/children" % (API, doc_id, doc_id),
                    headers=self._headers(), json={"children": blocks}, timeout=20,
                )
                data = resp.json()
                if data.get("code") != 0:
                    raise RuntimeError("写入文档块失败：%s %s" % (data.get("code"), data.get("msg")))
                count += 1
            self.db.kv_set("wiki_kb_count_" + cat, str(count))

    def _ensure_doc(self, space_id, category, doc_no):
        """确保「<分类>面试题<doc_no>」文档存在，返回 document_id。"""
        key = "wiki_kb_doc_%s_%d" % (category, doc_no)
        doc_id = self.db.kv_get(key)
        if doc_id:
            return doc_id
        resp = requests.post(
            API + "/docx/v1/documents", headers=self._headers(),
            json={"title": "%s面试题%d" % (category, doc_no)}, timeout=20,
        )
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError("创建文档失败：%s %s" % (data.get("code"), data.get("msg")))
        doc_id = data["data"]["document"]["document_id"]
        # 挂进知识库空间（父节点：空间「首页」节点，应用对该节点有可管理权限）
        mv = requests.post(
            "%s/wiki/v2/spaces/%s/nodes/move_docs_to_wiki" % (API, space_id),
            headers=self._headers(),
            json={"obj_type": "docx", "obj_token": doc_id,
                  "parent_wiki_token": getattr(self.cfg, "wiki_parent_node", "")},
            timeout=20,
        ).json()
        if mv.get("code") != 0:
            raise RuntimeError("文档挂入知识库失败：%s %s" % (mv.get("code"), mv.get("msg")))
        self.db.kv_set(key, doc_id)
        return doc_id

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
