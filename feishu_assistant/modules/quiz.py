"""题库模块：搜集的面试题/面经 → 分析答题思路 → 题库考试（抽题→作答→点评→记录掌握度）。

与错题本相互独立：错题本记实战踩坑，题库记主动搜集的备考题（含面经里提取的）。
题目固定分为 产品类/算法类/Agent类/其他类，用户也可指定分类。
考试会话数据由 router 持有并透传（session dict），本模块只负责业务逻辑。
"""
import re

import requests

from ..db import now_str

ADD_PROMPT = """你是面试备考助手。用户发来的可能是：一批搜集的面试题，或一篇面经（真实面试经历）。
请从中提取/整理出所有面试题，逐题分析，输出 JSON：
{"items": [{"question": "题目原文（精简，去掉序号）",
            "category": "四选一：产品类 / 算法类 / Agent类 / 其他类",
            "analysis": "第一行固定为「考察点：…」（一句话）；随后换行给出完整参考答案（300-500 字：答题框架并展开要点，可含恰当的例子或对比，像教科书答案一样可以直接照着学）"}]}

分类规则：
- 产品类：产品经理相关——需求分析、用户研究、商业/增长、产品设计、AI 产品落地
- 算法类：机器学习/深度学习基础、模型训练（SFT/RL 等）、数据处理、非 Agent 的系统设计
- Agent类：智能体、workflow/DAG、多智能体、工具调用、RAG 应用、Agent 框架与工程
- 其他类：以上都不适合的（如纯行为面试题），不要硬塞
只输出 JSON；面经中的非题目内容（薪资、流程描述等）忽略。输入：
%s"""

EVAL_PROMPT = """你是面试教练，正在用题库考用户。
题目：%s
参考答题思路：%s

用户的回答：%s

请评判，输出 JSON：
{"verdict": "good 或 bad（回答基本覆盖了思路要点为 good，明显跑偏/答不上来为 bad）",
 "feedback": "先一句总评，再指出答得好的地方和缺失的要点（120 字内）"}
只输出 JSON。"""


def fetch_url_text(url, max_chars=8000):
    """抓取公开网页正文（去标签），失败返回空串。JS 重/需登录的页面会抓不到。"""
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"},
                            timeout=15)
        resp.raise_for_status()
    except Exception:
        return ""
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", resp.text)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


class QuizModule:
    def __init__(self, cfg, db, llm, bitable=None, wiki=None):
        self.cfg = cfg
        self.db = db
        self.llm = llm
        self.bitable = bitable
        self.wiki = wiki

    # ---------- 录入 + 分析（含面经链接/文本） ----------
    CATEGORIES = ("产品类", "算法类", "Agent类", "其他类")

    @classmethod
    def _norm_category(cls, raw):
        """分类兜底：LLM 没按四类输出时，按关键词归拢，实在不像的进其他类。"""
        c = (raw or "").strip()
        if c in cls.CATEGORIES:
            return c
        if "agent" in c.lower() or "智能体" in c:
            return "Agent类"
        if "产品" in c:
            return "产品类"
        if any(k in c for k in ("算法", "训练", "模型", "数据", "RL", "SFT", "系统", "评估", "机器学习", "深度学习")):
            return "算法类"
        return "其他类"

    def handle_add(self, user_id, args, raw_text):
        text = (args.get("questions") or raw_text or "").strip()
        force_category = (args.get("category") or "").strip()
        if len(text) < 5:
            return ("把搜集到的面试题发给我（可以一次多条），或直接把**面经链接/面经文本**发给我，"
                    "我会提取其中的题目，逐题分析考察点和答题思路，按类型（产品类/算法类/Agent类）整理进题库，"
                    "并归档到知识库「全网真题面经库」。之后回复「考考我」或「用题库模拟面试」就能练。")
        # 面经链接：抓取正文；抓不到且原文几乎只有链接时给引导
        source = "搜集"
        urls = re.findall(r"https?://[^\s）)\]]+", text)
        if urls:
            fetched = fetch_url_text(urls[0])
            if fetched:
                text, source = fetched, "面经"
            elif len(re.sub(r"https?://\S+", "", text).strip()) < 10:
                return ("这个链接我抓不到内容（可能需要登录或有反爬）。"
                        "把面经的文字直接复制粘贴给我就行。")
        data = self.llm.chat_json([
            {"role": "user", "content": ADD_PROMPT % text[:8000]},
        ])
        items = data.get("items") or []
        if not items:
            return "没解析出有效的面试题，能把题目一条一条列清楚再发我吗？"
        saved = []
        for it in items[:20]:
            q = str(it.get("question") or "").strip()
            if not q:
                continue
            analysis = str(it.get("analysis") or "").strip()
            category = force_category or self._norm_category(str(it.get("category") or ""))
            self.db.add_question(user_id, q, analysis, category, source)
            saved.append((q, category, analysis))
            if self.bitable:
                # 多维表格只做索引：题目/分类/考察点/录入时间，完整答案在知识库文档
                self.bitable.sync_question(q, analysis.split("\n")[0], category, now_str())
        head = "已从面经提取 %d 道题" % len(saved) if source == "面经" else "已把 %d 道题分析并存入题库" % len(saved)
        lines = [head + "：\n"]
        for i, (q, cat, analysis) in enumerate(saved, 1):
            lines.append("%d. 【%s】%s" % (i, cat, q))
            if analysis:
                lines.append("   %s" % analysis)
        if self.wiki:
            self.wiki.archive_questions(user_id, saved, source)
            lines.append("\n（已同步归档到知识库「全网真题面经库」，按分类分文档连续编号）")
        lines.append("\n回复「考考我」抽题考试，或「用题库模拟面试」来场实战演练。")
        return "\n".join(lines)

    # ---------- 浏览 ----------
    def handle_query(self, user_id, args):
        category = (args.get("category") or "").strip() or None
        rows = self.db.list_questions(user_id, category=category)
        if not rows:
            return "题库还是空的。把搜集到的面试题或面经发给我，我帮你分析入库。"
        if not category:
            # 分类汇总视图
            counts = {}
            for r in self.db.list_questions(user_id, limit=200):
                counts[r["category"] or "未分类"] = counts.get(r["category"] or "未分类", 0) + 1
            summary = "、".join("%s %d" % (c, n) for c, n in sorted(counts.items(), key=lambda x: -x[1]))
            lines = ["你的题库（共 %d 条）：%s\n" % (sum(counts.values()), summary)]
        else:
            lines = ["你的题库「%s」类（共 %d 条）：\n" % (category, len(rows))]
        for i, r in enumerate(rows, 1):
            stat = "考过%d次" % r["asked_count"] if r["asked_count"] else "未考过"
            src = "·面经" if r["source"] == "面经" else ""
            lines.append("%d. 【%s】%s（%s%s）" % (i, r["category"] or "未分类", r["question"], stat, src))
        lines.append("\n回复「考我xx类的题」或「看看xx类的题」按类型筛选。")
        lines.append("也可以在飞书云文档查看：多维表格「面试小助理」（题库/错题本两张表）、"
                     "知识库「全网真题面经库」（按分类分文档）。")
        return "\n".join(lines)

    # ---------- 考试状态机 ----------
    def start_quiz(self, user_id, args, ctx):
        category = (args.get("category") or "").strip() or None
        row = self.db.pick_quiz_question(user_id, category=category)
        if not row:
            if category:
                return "题库里没有「%s」分类的题。可以先发题给我入库，或回复「看看题库」查现有分类。" % category
            return "题库还是空的。把搜集到的面试题发给我，我帮你分析入库后再考你。"
        session = {"module": "quiz", "category": category,
                   "qid": row["id"], "question": row["question"], "analysis": row["analysis"],
                   "qcat": row["category"], "count": 0, "good": 0}
        reply = self._ask_text(session, row)
        return session, reply

    def continue_quiz(self, user_id, text, session):
        # 评判当前题
        data = self.llm.chat_json([
            {"role": "user", "content": EVAL_PROMPT % (
                session["question"], session.get("analysis") or "（无）", text[:2000])},
        ])
        good = str(data.get("verdict") or "").lower() == "good"
        feedback = str(data.get("feedback") or "").strip() or "（评判失败，跳过点评）"
        self.db.record_quiz_result(session["qid"], good)
        session["count"] += 1
        session["good"] += 1 if good else 0
        mistake_note = ""
        if not good:
            # 答不上来的题自动记入错题本（同题去重）——错题本的来源之一
            if not any(m["question"] == session["question"]
                       for m in self.db.list_mistakes(user_id, limit=200)):
                self.db.add_mistake(user_id, session["question"],
                                    session.get("analysis") or "", session.get("qcat") or "未分类")
                if self.bitable:
                    self.bitable.sync_mistake(session["question"], session.get("analysis") or "",
                                              session.get("qcat") or "未分类", now_str())
            mistake_note = "（已记入错题本）"
        # 抽下一题
        row = self.db.pick_quiz_question(
            user_id, category=session.get("category"), exclude_id=session["qid"])
        if not row:
            session["done"] = True  # 通知 router 清除会话
            return feedback + mistake_note + "\n\n" + self.end_quiz(user_id, session, exhausted=True)
        session["qid"], session["question"], session["analysis"] = row["id"], row["question"], row["analysis"]
        session["qcat"] = row["category"]
        mark = "√ 答得不错" if good else "× 还差点意思"
        return "%s%s\n%s\n\n%s" % (mark, mistake_note, feedback, self._ask_text(session, row))

    def end_quiz(self, user_id, session, exhausted=False):
        count, good = session.get("count", 0), session.get("good", 0)
        head = "题库的题都考过一轮了。" if exhausted else "考试结束。"
        if not count:
            return head
        return ("%s本次共考 %d 题，答得不错 %d 题、还需加强 %d 题。"
                "答砸的题之后会被优先抽中，随时回复「考考我」再来一轮。" % (head, count, good, count - good))

    def _ask_text(self, session, row):
        return "【题库考试】（%s）\n%s\n\n请作答；回复「结束」停止考试。" % (
            row["category"] or "未分类", row["question"])
