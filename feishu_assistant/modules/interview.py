"""面试助手模块：模拟面试（多轮状态机）、JD 题集、简历深挖、复盘错题本。

模拟面试的会话数据由 router 持有并透传（session dict），本模块只负责业务逻辑。
"""
import re

from ..db import now_str
from .profile import profile_prompt_text

INTERVIEWER_PROMPT = """你是一位资深面试官，正在对候选人进行一场正式面试。
面试方向/岗位信息：
{topic}

要求：
1. 一次只问一个问题，不要一次抛出多个问题；
2. 候选人回答后，先用一两句话点评或追问其回答中的疑点，再抛出下一个问题；
3. 由浅入深，逐步覆盖：专业基础、项目经历深挖、场景设计/开放问题、行为问题（STAR 法则）；
4. 语言专业、简洁、全程使用中文；
5. 这是你的开场：先简短欢迎并说明面试流程，然后立即问第一个问题。"""

EVALUATOR_PROMPT = """你是一位资深面试教练。以下是一场模拟面试的完整记录（面试方向：{topic}）。
请按 STAR 法则进行复盘点评：

1. 逐题点评：候选人每个回答的亮点、不足（情境/任务/行动/结果哪些没讲清）、改进建议；
2. 总体评价：表达逻辑、专业深度、应变表现三方面打分（10 分制）并说明理由；
3. 备战建议：给出 3 条最有针对性的改进建议。

用中文输出，结构清晰，直接给出内容，不要客套。"""

WEAKNESS_PROMPT = """你是面试教练。基于用户的错题记录、题库考试弱项和历次面试复盘，找出 TA 的**薄弱模式**——
不是单个知识点，而是反复出现的问题模式（例如"回答缺量化结果""系统设计题缺乏权衡意识""项目细节经不起追问"）。

输出结构：
1. 薄弱模式 TOP3（每个：模式一句话 + 证据（来自哪些记录）+ 具体改进方法）；
2. 一周针对性提升计划（每天 30-60 分钟，可执行）；
3. 建议优先复习的错题/题库类别。

一针见血，中文，不客套。数据如下：
%s"""

EXTRACT_FAIL_PROMPT = """以下是一场模拟面试的记录。找出候选人答得不好、答不上来或明显薄弱的问题（最多 5 个），
给出每个问题的更好回答要点，输出 JSON：
{"items": [{"question": "面试官的问题", "answer": "更好的回答要点（100 字内）",
            "category": "分类（如 技术基础/项目经验/算法/系统设计/行为）"}]}
如果候选人整体答得都不错，输出 {"items": []}。只输出 JSON。面试记录：
%s"""


class InterviewModule:
    def __init__(self, cfg, db, llm, bitable=None):
        self.cfg = cfg
        self.db = db
        self.llm = llm
        self.bitable = bitable

    # ---------- 模拟面试 ----------
    def start_mock(self, user_id, args, ctx):
        topic = (args.get("topic") or "").strip()
        if not topic:
            return None, "想模拟什么方向的面试？比如：模拟一场后端开发的面试，或直接把 JD 粘贴给我。"
        # 题库模式：「用题库模拟面试」「用题库里Agent的题考我」——优先从用户题库选题
        if "题库" in topic:
            rows = self._bank_questions(user_id, topic)
            if not rows:
                return None, ("题库还是空的。先把搜集的面试题或面经发给我入库，"
                              "再来「用题库模拟面试」。")
            bank_text = "\n".join("%d. 【%s】%s" % (i, r["category"] or "未分类", r["question"])
                                  for i, r in enumerate(rows, 1))
            topic = ("基于用户自选题库的定向面试。优先从以下题目中挑选提问，"
                     "并对回答追问深挖；题目问完后可自由延伸相关题目：\n" + bank_text)
        # 指代解析：刚生成过 JD 题集 / 刚分析过简历时，把对应内容并入面试上下文
        elif ("JD" in topic.upper() or "这个" in topic or "这份" in topic) and ctx.get("last_jd"):
            topic = "基于以下岗位 JD 的定向面试：\n" + ctx["last_jd"][:4000]
        elif "简历" in topic and ctx.get("last_resume_text"):
            topic = "基于候选人简历的定向面试。简历内容：\n" + ctx["last_resume_text"][:4000]
        # 个人画像注入：让面试官知道候选人背景
        bg = profile_prompt_text(self.db, user_id)
        if bg:
            topic += "\n\n候选人背景（提问时结合，但不要提及你知道这些信息）：\n" + bg
        session = {
            "module": "interview_mock",
            "topic": topic,
            "messages": [{"role": "system", "content": INTERVIEWER_PROMPT.format(topic=topic)}],
        }
        opening = self.llm.chat(session["messages"])
        session["messages"].append({"role": "assistant", "content": opening})
        return session, opening + "\n\n（面试已开始，随时回复「结束面试」获取复盘点评）"

    def _bank_questions(self, user_id, topic, limit=8):
        """按 topic 里的关键词从题库筛题（匹配分类或题干），筛不到就用最近的题。"""
        kws = [w for w in re.split(r"[\s，。的]+", topic)
               if len(w) >= 2 and w not in ("题库", "模拟", "面试", "考我", "出来", "一场")]
        rows = self.db.list_questions(user_id, limit=50)
        if kws:
            matched = [r for r in rows
                       if any(k in ((r["category"] or "") + r["question"]) for k in kws)]
            if matched:
                rows = matched
        return rows[:limit]

    def continue_mock(self, user_id, text, session):
        session["messages"].append({"role": "user", "content": text})
        reply = self.llm.chat(session["messages"])
        session["messages"].append({"role": "assistant", "content": reply})
        return reply

    def end_mock(self, user_id, session):
        messages = session.get("messages") or []
        # 提炼问答记录（去掉 system prompt）
        transcript = [m for m in messages if m.get("role") in ("user", "assistant")]
        if len(transcript) <= 1:
            return "面试还没正式开始就结束了。想练的时候随时说「模拟面试」。"
        record_text = "\n".join(
            ("面试官：" if m["role"] == "assistant" else "候选人：") + m["content"]
            for m in transcript
        )
        feedback = self.llm.chat([
            {"role": "system", "content": EVALUATOR_PROMPT.format(topic=session.get("topic", ""))},
            {"role": "user", "content": record_text[:20000]},
        ])
        self.db.add_interview_session(
            user_id, "mock",
            {"topic": session.get("topic", ""), "record": record_text},
            feedback,
        )
        reply = "面试结束，以下是复盘点评：\n\n" + feedback
        # 提取答得不好的题，自动记入错题本（错题本的来源之一：模拟面试）
        n = self._harvest_mistakes(user_id, record_text)
        if n:
            reply += "\n\n（%d 道答得不好的题已记入错题本，回复「看看错题本」随时复习）" % n
        return reply + "\n\n（记录已归档，回复「面试记录」可查看历史）"

    def _harvest_mistakes(self, user_id, record_text):
        """从面试记录中提取弱项题入错题本；返回新录入条数。失败静默返回 0。"""
        try:
            data = self.llm.chat_json([
                {"role": "user", "content": EXTRACT_FAIL_PROMPT % record_text[:15000]},
            ])
        except LLMError:
            return 0
        existing = {m["question"] for m in self.db.list_mistakes(user_id, limit=200)}
        n = 0
        for it in (data.get("items") or [])[:5]:
            q = str(it.get("question") or "").strip()
            if not q or q in existing:
                continue
            answer = str(it.get("answer") or "").strip()
            category = str(it.get("category") or "未分类").strip()
            self.db.add_mistake(user_id, q, answer, category)
            if self.bitable:
                self.bitable.sync_mistake(q, answer, category, now_str())
            existing.add(q)
            n += 1
        return n

    # ---------- JD 生成题集 ----------
    def handle_jd(self, user_id, args, ctx):
        jd = (args.get("jd") or args.get("topic") or "").strip()
        if not jd:
            return "把岗位 JD 粘贴给我，我来生成针对性的面试题集。"
        ctx["last_jd"] = jd[:6000]
        result = self.llm.chat([
            {"role": "system", "content":
                "你是资深面试官。根据用户给出的岗位 JD，生成一份面试题集，包含：\n"
                "1. 专业技术题（8-10 题，紧扣 JD 要求的技术栈，标注考察点）；\n"
                "2. 项目/经验题（3-5 题）；\n"
                "3. 行为题（3 题，注明 STAR 考察维度）。\n"
                "用中文，分节清晰，只出题不附答案。"},
            {"role": "user", "content": "岗位 JD：\n" + jd[:6000]},
        ])
        self.db.add_interview_session(user_id, "jd", {"jd": jd[:2000]}, result)
        return result + "\n\n想实战演练的话，回复「用这个 JD 模拟面试」即可开始。"

    # ---------- 简历深挖 ----------
    def handle_resume(self, user_id, resume_text, ctx):
        resume_text = (resume_text or "").strip()
        if len(resume_text) < 50:
            return "简历内容太短了。可以把简历文字粘贴给我，或直接发 PDF 文件。"
        ctx["last_resume_text"] = resume_text[:8000]
        result = self.llm.chat([
            {"role": "system", "content":
                "你是资深面试官。针对候选人简历做深挖式提问，要求：\n"
                "1. 按项目/经历分组，每组 2-4 个追问，问细节、问权衡、问结果数据；\n"
                "2. 标出你认为简历中表述模糊、面试时容易被挑战的点；\n"
                "3. 用中文，问题要具体，不要泛泛的「介绍一下你的项目」。"},
            {"role": "user", "content": "简历内容：\n" + resume_text[:8000]},
        ])
        self.db.add_interview_session(user_id, "resume", {"resume": resume_text[:2000]}, result)
        return result + "\n\n回复「用这份简历模拟面试」，我可以直接扮演面试官开始实战演练。"

    # ---------- 复盘错题本 ----------
    def handle_review(self, user_id, args, raw_text):
        action = (args.get("action") or "").strip() or "add"
        if action == "query":
            return self._query_mistakes(user_id, args)

        questions_text = (args.get("questions") or raw_text or "").strip()
        if len(questions_text) < 5:
            return ("把面试中被问到的问题发给我（可以一次多条），我会整理进错题本并附参考思路。\n"
                    "攒几轮后回复「分析我的薄弱点」，我帮你找出反复栽跟头的问题模式。")
        data = self.llm.chat_json([
            {"role": "system", "content": "你是面试复盘助手，输出结构化 JSON。"},
            {"role": "user", "content":
                "以下是面试中实际被问到的问题，请逐条整理，输出 JSON：\n"
                "{\"items\": [{\"question\": \"原问题\", \"category\": \"分类（如 技术基础/项目经验/算法/行为/系统设计）\", "
                "\"answer\": \"参考回答要点（100 字内）\"}]}\n\n问题原文：\n%s" % questions_text[:4000]},
        ])
        items = data.get("items") or []
        if not items:
            return "没解析出有效问题，能把问题一条一条列清楚再发我吗？"
        saved = []
        for it in items[:20]:
            q = str(it.get("question") or "").strip()
            if not q:
                continue
            self.db.add_mistake(
                user_id, q,
                str(it.get("answer") or "").strip(),
                str(it.get("category") or "未分类").strip(),
            )
            saved.append((q, str(it.get("category") or "未分类"), str(it.get("answer") or "")))
            if self.bitable:
                self.bitable.sync_mistake(q, str(it.get("answer") or "").strip(),
                                          str(it.get("category") or "未分类").strip(), now_str())
        lines = ["已把 %d 道题存入错题本：\n" % len(saved)]
        for i, (q, cat, ans) in enumerate(saved, 1):
            lines.append("%d. 【%s】%s" % (i, cat, q))
            if ans:
                lines.append("   参考思路：%s" % ans)
        lines.append("\n回复「看看错题本」或「看看技术类的错题」可随时复习。")
        return "\n".join(lines)

    def _query_mistakes(self, user_id, args):
        category = (args.get("category") or "").strip() or None
        rows = self.db.list_mistakes(user_id, category=category)
        if not rows:
            return "错题本还是空的。面试后把被问倒的问题发给我，我帮你整理。"
        lines = ["你的错题本（共 %d 条%s）：\n" % (len(rows), "，分类：" + category if category else "")]
        for i, r in enumerate(rows, 1):
            lines.append("%d. 【%s】%s" % (i, r["category"] or "未分类", r["question"]))
            if r["answer"]:
                lines.append("   参考思路：%s" % r["answer"])
        return "\n".join(lines)

    # ---------- 薄弱模式分析（跨错题/题库/面试记录找规律） ----------
    def weakness_analysis(self, user_id):
        mistakes = self.db.list_mistakes(user_id, limit=50)
        bank = self.db.list_questions(user_id, limit=100)
        sessions = self.db.list_interview_sessions(user_id, limit=5)
        weak_bank = [r for r in bank if r["asked_count"] >= 1 and r["good_count"] * 2 <= r["asked_count"]]
        if not mistakes and not weak_bank and not sessions:
            return ("还没有可分析的面试数据。先练起来：「模拟面试」「考考我」，"
                    "真实面试后把被问倒的问题发给我，攒几轮后我再帮你找薄弱模式。")
        parts = []
        if mistakes:
            parts.append("【错题本（实战踩坑）】\n" + "\n".join(
                "· 【%s】%s" % (r["category"] or "未分类", r["question"]) for r in mistakes[:30]))
        if weak_bank:
            parts.append("【题库考试弱项（答错率≥50%）】\n" + "\n".join(
                "· 【%s】%s（考%d次错%d次）" % (r["category"] or "未分类", r["question"],
                                             r["asked_count"], r["asked_count"] - r["good_count"])
                for r in weak_bank[:20]))
        if sessions:
            parts.append("【近期面试复盘摘要】\n" + "\n".join(
                "· [%s] %s" % (r["type"], (r["feedback"] or "")[:300]) for r in sessions))
        reply = self.llm.chat([
            {"role": "user", "content": WEAKNESS_PROMPT % "\n\n".join(parts)[:8000]},
        ])
        reply = "【薄弱模式分析】\n\n" + reply + "\n\n（建议配合：「看看错题本」复习、「考考我」验证提升）"
        self.db.add_interview_session(user_id, "weakness", {}, reply)
        return reply

    # ---------- 历史面试记录 ----------
    def list_sessions(self, user_id):
        rows = self.db.list_interview_sessions(user_id)
        if not rows:
            return "还没有面试记录。回复「模拟面试」开始第一场练习吧。"
        type_name = {"mock": "模拟面试", "jd": "JD题集", "resume": "简历深挖", "review": "复盘",
                     "jd_match": "岗位匹配", "prep_resume": "定制简历",
                     "prep_letter": "自我介绍/求职信", "clinic": "简历问诊", "weakness": "薄弱分析"}
        lines = ["最近的面试相关记录："]
        for r in rows:
            lines.append("· [%s] %s（%s）" % (r["created_at"][:16], type_name.get(r["type"], r["type"]), (r["feedback"] or "")[:30] + "..."))
        lines.append("\n（历史详情暂存于本地数据库，后续可按需展开查看）")
        return "\n".join(lines)
