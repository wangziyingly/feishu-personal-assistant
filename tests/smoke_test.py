"""冒烟测试：不依赖飞书凭证和真实大模型，验证核心链路。

运行：.venv/bin/python tests/smoke_test.py
"""
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feishu_assistant.config import Config          # noqa: E402
from feishu_assistant.db import Database            # noqa: E402
from feishu_assistant.llm import parse_json, LLMError  # noqa: E402
from feishu_assistant.router import Router          # noqa: E402
from feishu_assistant.modules import paper as paper_mod  # noqa: E402

PASSED, FAILED = [], []


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print("PASS  %s" % name)
    except Exception as e:
        FAILED.append((name, e))
        print("FAIL  %s: %s" % (name, e))


# ---------- 假 LLM / 假 Bot ----------
TOMORROW = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")


class FakeLLM:
    """按提示词特征返回预设结果，模拟真实模型的路由与生成。"""

    def __init__(self):
        self.seen = []  # 记录每次 chat 收到的 messages，验证历史注入

    def chat(self, messages, temperature=None, max_tokens=None, on_delta=None):
        self.seen.append(messages)
        sys_msg = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        user_msg = messages[-1]["content"] if messages else ""
        if "资深简历顾问" in user_msg:
            return "改动说明：突出分布式项目、对齐 Go/K8s 关键词。\n\n定制简历正文：核心优势……"
        if "求职表达教练" in user_msg and "求职信" in user_msg:
            return "求职信正文：理解岗位……三点理由……"
        if "求职表达教练" in user_msg:
            return "自我介绍口述稿：开场定位……亮点……"
        sys_msg = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        if "面试教练" in sys_msg:
            return "复盘点评：STAR 分析……总体 7 分。"
        if "资深面试官" in sys_msg:
            return "面试官：请做一下自我介绍。"
        if "文献汇报" in sys_msg or "组会" in sys_msg:
            return "【一句话总结】测试汇报\n【方法详解】方法细节\n【局限与批判性思考】局限"
        return "好的，我在。"

    def chat_vision(self, prompt, image_path):
        return "这篇面经：1. 什么是CAP？2. 如何设计一个Agent系统？"

    def chat_json(self, messages, temperature=0.1):
        sys_msg = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        user_msg = messages[-1]["content"] if messages else ""
        if "意图分类器" in sys_msg:
            return self._route(user_msg)
        if "检索式" in user_msg:
            return {"queries": ["multi agent credit assignment", "LLM agents"]}
        if "相关性" in user_msg:
            return {"picked": [{"i": 1, "reason": "最相关"}]}
        if "判断以下文档开头" in user_msg:
            return {"doc_type": "paper"}
        if "逐题分析" in user_msg:
            return {"items": [{"question": "什么是CAP？",
                               "category": "Agent",
                               "analysis": "考察分布式基础；思路：定义+取舍+实例"}]}
        if "请评判" in user_msg:
            return {"verdict": "good", "feedback": "基本覆盖要点，缺实例"}
        if "从以下简历中提炼" in user_msg:
            return {"highlights": "熟悉Python/Go；做过分布式调度系统；211硕士"}
        if "解析以下岗位 JD" in user_msg:
            return {"summary": "后端开发工程师",
                    "hard_requirements": ["3年经验", "熟悉Go"],
                    "plus_items": ["熟悉K8s"],
                    "hidden_preferences": ["偏好大厂背景（推断）"]}
        if "匹配分析" in user_msg:
            return {"score": 75, "verdict": "需要定制简历",
                    "reasons": ["技术栈基本匹配"], "resume_edits": ["突出分布式项目"],
                    "skills_to_learn": ["K8s"]}
        if "挑剔眼光" in user_msg:
            return {"fatal": ["缺少量化结果"],
                    "issues": [{"problem": "项目描述太笼统", "fix": "补 STAR 结构和指标"}],
                    "overall": "有潜力但需大改"}
        if "答得不好、答不上来或明显薄弱" in user_msg:
            return {"items": [{"question": "什么是RESTful？",
                               "answer": "要点：资源定位+统一接口+无状态",
                               "category": "技术基础"}]}
        if "的正文" in user_msg:
            return {"title": "测试论文", "summary": "【研究问题】x\n【方法】y", "tags": ["DL"]}
        if "技术标签" in user_msg:
            return {"tags": ["深度学习", "RAG"]}
        return {}

    def _route(self, text):
        if "提醒我" in text:
            return {"intent": "todo_create",
                    "args": {"task": "交周报", "remind_at": TOMORROW + " 15:00"}}
        if "什么事" in text:
            return {"intent": "todo_manage", "args": {"action": "query"}}
        if "完成" in text:
            return {"intent": "todo_manage", "args": {"action": "complete", "index": 1}}
        if "汇报" in text or "精读" in text or ("这篇" in text and "面经" not in text):
            m = re.search(r"第\s*(\d+)\s*篇", text)
            return {"intent": "paper_report",
                    "args": {"index": int(m.group(1)) if m else None,
                             "title": "Reflexion" if ("这篇" in text and not m) else None}}
        if "论文" in text and ("搜" in text or "找" in text):
            return {"intent": "paper_search", "args": {"query": text}}
        if "感兴趣" in text:
            return {"intent": "paper_summarize_index", "args": {"index": 1}}
        if "结束面试" in text or "模拟" in text and "面试" in text:
            if "题库" in text:
                return {"intent": "interview_mock", "args": {"topic": "题库 Agent"}}
            return {"intent": "interview_mock", "args": {"topic": "后端开发"}}
        if "面经" in text:
            return {"intent": "quiz_add", "args": {"questions": text}}
        if "考考我" in text:
            return {"intent": "quiz_start", "args": {"category": None}}
        if "看看题库" in text:
            return {"intent": "quiz_query", "args": {}}
        if "面试题" in text and "分析" in text:
            return {"intent": "quiz_add", "args": {"questions": text}}
        if "取消" in text and "订阅" in text:
            return {"intent": "paper_subscribe", "args": {"enable": False}}
        if "订阅" in text:
            return {"intent": "paper_subscribe",
                    "args": {"keywords": "智能体", "enable": True, "time": "09:00", "top_n": 5}}
        if text.startswith("新建对话"):
            return {"intent": "conv_manage",
                    "args": {"action": "new", "name": text[4:].strip() or None}}
        if "切换" in text:
            return {"intent": "conv_manage", "args": {"action": "switch", "name": "默认对话"}}
        if "记住" in text:
            return {"intent": "profile_update",
                    "args": {"research_direction": "多智能体强化学习"}}
        if "定制" in text and "简历" in text:
            return {"intent": "prep_resume", "args": {}}
        if "求职信" in text:
            return {"intent": "prep_letter", "args": {"kind": "求职信"}}
        if "自我介绍" in text:
            return {"intent": "prep_letter", "args": {"kind": "自我介绍"}}
        if "问诊" in text:
            return {"intent": "resume_clinic", "args": {}}
        if "简历" in text:
            return {"intent": "interview_resume", "args": {"resume_text": text}}
        if "适不适合" in text or "能投" in text:
            return {"intent": "jd_match", "args": {"jd": text}}
        return {"intent": "chat", "args": {"text": text}}


FAKE_PAPERS = [
    {"title": "Paper A", "authors": "Alice", "summary": "multi agent credit assignment method",
     "url": "http://arxiv.org/abs/2601.00001", "published": "2026-07-30", "source": "arxiv"},
    {"title": "Paper B", "authors": "Bob", "summary": "unrelated topic",
     "url": "http://arxiv.org/abs/2601.00002", "published": "2026-07-29", "source": "arxiv"},
]


class FakeBot:
    def __init__(self):
        self.replies = []
        self.pushes = []
        self.patches = []       # 流式回复的定稿内容
        self.download_path = None

    def reply(self, message_id, text):
        self.replies.append(text)

    def reply_get_id(self, message_id, text):
        self.replies.append(text)
        return "stream_mid"

    def patch_message(self, mid, text):
        self.patches.append(text)

    def push(self, chat_id, text):
        self.pushes.append((chat_id, text))

    def download_file(self, message_id, file_key, save_dir, file_name):
        return self.download_path

    def download_image(self, message_id, image_key, save_dir):
        return self.download_path


def make_env():
    tmp = tempfile.mkdtemp()
    cfg = Config({"llm": {"api_key": "sk-test"}})
    db = Database(os.path.join(tmp, "test.db"))
    bot = FakeBot()
    router = Router(cfg, db, FakeLLM(), bot)
    return cfg, db, bot, router


def say(router, bot, user, text):
    bot.replies.clear()
    bot.patches.clear()
    router.handle_message(user, "chat_1", "mid_%d" % len(bot.replies), "text", {"text": text})
    if bot.patches:  # 流式回复：定稿在 patches
        return bot.patches[-1]
    assert bot.replies, "没有回复"
    return bot.replies[-1]


# ---------- 测试 ----------
def test_config_defaults():
    cfg = Config({})
    assert not cfg.feishu_configured and not cfg.llm_configured
    cfg2 = Config({"llm": {"api_key": "sk-abc"}})
    assert cfg2.llm_configured


def test_parse_json():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json('前言 {"a": 3} 后记') == {"a": 3}
    try:
        parse_json("不是JSON")
        raise AssertionError("应当抛 LLMError")
    except LLMError:
        pass


def test_db_cycle():
    tmp = tempfile.mkdtemp()
    db = Database(os.path.join(tmp, "t.db"))
    db.upsert_user("u1", "c1")
    assert db.get_chat_id("u1") == "c1"
    tid = db.add_todo("u1", "写报告", "2000-01-01 09:00")
    assert len(db.pending_todos("u1")) == 1
    due = db.due_todos(datetime.now().strftime("%Y-%m-%d %H:%M"))
    assert len(due) == 1 and due[0]["content"] == "写报告"
    db.mark_reminded(tid)
    assert not db.due_todos(datetime.now().strftime("%Y-%m-%d %H:%M"))
    db.complete_todo(tid, "u1")
    assert not db.pending_todos("u1")
    pid = db.add_paper("u1", "标题", "arxiv", "http://x", "总结", "RAG")
    assert db.get_paper(pid, "u1")["title"] == "标题"
    assert db.search_papers("u1", "RAG")
    db.set_subscription("u1", "agent")
    assert db.get_subscription("u1")["keywords"] == "agent"
    assert len(db.active_subscriptions()) == 1
    db.add_mistake("u1", "什么是CAP", "参考思路", "技术基础")
    assert db.list_mistakes("u1", category="技术")
    db.add_interview_session("u1", "mock", {"record": "..."}, "点评")
    assert db.list_interview_sessions("u1")
    db.close()


def test_todo_flow():
    cfg, db, bot, router = make_env()
    r = say(router, bot, "u1", "明天下午3点提醒我交周报")
    assert "已记下" in r and "交周报" in r, r
    r = say(router, bot, "u1", "我今天有什么事")
    assert "交周报" in r, r
    r = say(router, bot, "u1", "完成了第1条")
    assert "已完成" in r, r
    r = say(router, bot, "u1", "我今天有什么事")
    assert "没有任何待办" in r, r


def test_mock_interview_flow():
    cfg, db, bot, router = make_env()
    r = say(router, bot, "u1", "模拟一场后端开发面试")
    assert "自我介绍" in r, r
    assert router.sessions.get("u1"), "会话应已建立"
    r = say(router, bot, "u1", "我叫张三，做了三年后端")
    assert "面试官" in r, r
    r = say(router, bot, "u1", "结束面试")
    assert "复盘点评" in r, r
    assert not router.sessions.get("u1"), "会话应已清除"
    assert db.list_interview_sessions("u1"), "面试记录应已归档"
    # 答得不好的题应自动记入错题本
    mistakes = db.list_mistakes("u1")
    assert "已记入错题本" in r, r
    assert mistakes and mistakes[0]["question"] == "什么是RESTful？", mistakes


def test_inbox_queue():
    import tempfile
    db = Database(os.path.join(tempfile.mkdtemp(), "q.db"))
    assert db.inbox_enqueue("m1", {"a": 1})
    assert not db.inbox_enqueue("m1", {"a": 1}), "重复 message_id 应去重"
    rows = db.inbox_pending()
    assert len(rows) == 1 and rows[0]["status"] == "pending"
    db.inbox_mark(rows[0]["id"], "done")
    assert not db.inbox_pending(), "done 不再重放"
    db.inbox_enqueue("m2", {})
    db.inbox_mark(db.inbox_pending()[0]["id"], "failed")
    assert len(db.inbox_pending()) == 1, "failed 且 attempts<3 仍应重放"
    db.close()


def test_conversation_manage():
    cfg, db, bot, router = make_env()
    say(router, bot, "u1", "你好")
    r = say(router, bot, "u1", "新建对话 求职准备")
    assert "求职准备" in r, r
    r = say(router, bot, "u1", "对话列表")
    assert "默认对话" in r and "求职准备" in r and "← 当前" in r, r
    r = say(router, bot, "u1", "切换到默认对话")
    assert "已切换" in r, r
    assert any(m["content"] == "你好" for m in router.histories["u1"]), "切换后应恢复历史"


def test_digest_followup():
    cfg, db, bot, router = make_env()
    # 模拟日报推送后的登记
    router.note_digest_push("u1", "【论文日报】……", list(FAKE_PAPERS))
    assert router._ctx("u1")["last_papers"], "日报论文应进指代点"
    assert any("论文日报" in m["content"] for m in router.histories["u1"]), "推送应入历史"
    # 追问「我对第一篇比较感兴趣」→ 走摘要级解读
    orig = paper_mod.search_papers
    paper_mod.search_papers = lambda q, n: list(FAKE_PAPERS)
    try:
        r = say(router, bot, "u1", "我对第一篇比较感兴趣")
        assert "解读" in r and "已存入你的文献库" in r, r
    finally:
        paper_mod.search_papers = orig


def test_report_by_title():
    cfg, db, bot, router = make_env()
    db.add_paper("u1", "Reflexion: Language Agents with Verbal Reinforcement Learning",
                 "arxiv", "http://arxiv.org/abs/2303.11366",
                 "【一句话总结】已有汇报内容\n【汇报问答准备】Q&A", "Agent")
    r = say(router, bot, "u1", "就刚刚你说的Reflexion这篇")
    assert "文献汇报" in r and "已有汇报内容" in r, r


def test_quiz_bad_goes_to_mistakes():
    cfg, db, bot, router = make_env()
    db.add_question("u1", "什么是CAP？", "思路：定义+取舍+实例", "Agent")
    llm = router.quiz.llm
    orig = llm.chat_json

    def bad_verdict(messages, temperature=0.1):
        sys_msg = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        if "意图分类器" in sys_msg:
            return orig(messages)
        return {"verdict": "bad", "feedback": "没答到点上"}

    llm.chat_json = bad_verdict
    try:
        say(router, bot, "u1", "考考我")
        r = say(router, bot, "u1", "不知道")
        assert "已记入错题本" in r, r
        mistakes = db.list_mistakes("u1")
        assert mistakes and mistakes[0]["question"] == "什么是CAP？", mistakes
        assert len(mistakes) == 1, "同题不应重复记入"
    finally:
        llm.chat_json = orig


def test_pdf_flow_blank():
    cfg, db, bot, router = make_env()
    from pypdf import PdfWriter
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "blank.pdf")
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        w.write(f)
    bot.download_path = path
    bot.replies.clear()
    router.handle_message("u1", "c1", "mid_f", "file", {"file_name": "blank.pdf", "file_key": "k"})
    assert bot.replies and "提取不出" in bot.replies[-1], bot.replies


def test_pdf_flow_paper():
    cfg, db, bot, router = make_env()
    orig = paper_mod.extract_pdf_text
    paper_mod.extract_pdf_text = lambda p, **kw: ("Deep learning paper " * 50, False)
    try:
        bot.download_path = "/fake/paper.pdf"
        bot.replies.clear()
        router.handle_message("u1", "c1", "mid_f2", "file", {"file_name": "paper.pdf", "file_key": "k"})
        assert bot.replies and "已存入你的文献库" in bot.replies[-1], bot.replies
        assert db.list_papers("u1"), "文献库应有记录"
        assert router._ctx("u1").get("last_pdf_path") == "/fake/paper.pdf", "应记录 PDF 路径供文献汇报"
    finally:
        paper_mod.extract_pdf_text = orig


def test_need_driven_search():
    cfg, db, bot, router = make_env()
    orig = paper_mod.search_papers
    paper_mod.search_papers = lambda q, n: list(FAKE_PAPERS)
    try:
        r = say(router, bot, "u1", "我在做多智能体信用分配，帮我找最新的论文")
        assert "推荐理由：最相关" in r and "Paper A" in r, r
        ctx = router._ctx("u1")
        assert ctx.get("last_papers"), "搜索结果应存入上下文"
        assert ctx["last_papers"][0]["title"] == "Paper A", "排序后第一篇应是 Paper A"
    finally:
        paper_mod.search_papers = orig


def test_report_flow():
    cfg, db, bot, router = make_env()
    tmp = tempfile.mkdtemp()
    pdf_path = os.path.join(tmp, "report.pdf")
    with open(pdf_path, "wb") as f:
        f.write(b"%PDF-fake")
    ctx = router._ctx("u1")
    ctx["last_pdf_path"] = pdf_path
    orig = paper_mod.extract_pdf_text
    paper_mod.extract_pdf_text = lambda p, **kw: ("Long paper text " * 200, False)
    try:
        r = say(router, bot, "u1", "生成文献汇报")
        assert "文献汇报" in r and "一句话总结" in r, r
        assert db.list_papers("u1"), "汇报应已存入文献库"
    finally:
        paper_mod.extract_pdf_text = orig


def test_hard_filter():
    cfg, db, bot, router = make_env()
    pm = router.paper
    queries = ["multi agent credit assignment"]
    lib_paper = {"title": "Paper C", "authors": "", "summary": "multi agent stuff",
                 "url": "http://arxiv.org/abs/2601.00003", "published": "2026-07-30", "source": "arxiv"}
    db.add_paper("u1", "Paper C", "arxiv", lib_paper["url"], "已有总结", "")
    cands = list(FAKE_PAPERS) + [lib_paper]
    pool = pm._hard_filter("u1", cands, queries, 2)
    urls = [p["url"] for p in pool]
    assert lib_paper["url"] not in urls, "文献库已有的应被过滤"
    assert urls[0] == FAKE_PAPERS[0]["url"], "关键词匹配的应排前面"
    assert FAKE_PAPERS[1]["url"] in urls, "匹配不足 need_n 时应按原顺序补足"
    pool2 = pm._hard_filter("u1", cands, queries, 1)
    assert [p["url"] for p in pool2] == [FAKE_PAPERS[0]["url"]], "匹配够时只留匹配项"


def test_summarize_cache_reuse():
    cfg, db, bot, router = make_env()
    p = FAKE_PAPERS[0]
    db.add_paper("u1", p["title"], p["source"], p["url"], "库存总结内容", "RAG")
    ctx = {"last_papers": list(FAKE_PAPERS)}
    r = router.paper.handle_summarize_index("u1", {"index": 1}, ctx)
    assert "复用" in r and "库存总结内容" in r, r


def test_digest_skip_library():
    cfg, db, bot, router = make_env()
    for p in FAKE_PAPERS:
        db.add_paper("u1", p["title"], p["source"], p["url"], "已有", "")
    orig = paper_mod.search_papers
    paper_mod.search_papers = lambda q, n: list(FAKE_PAPERS)
    try:
        r, _ = router.paper.daily_digest("u1", "多智能体信用分配", 3)
        assert r is None, "候选全部已入库时不应推送"
    finally:
        paper_mod.search_papers = orig


def test_bare_commands():
    cfg, db, bot, router = make_env()
    r = say(router, bot, "u1", "搜文献")
    assert "想搜什么方向" in r, r
    r = say(router, bot, "u1", "新增一个待办")
    assert "没听清" in r, r
    r = say(router, bot, "u1", "模拟面试")
    assert "想模拟什么方向" in r, r
    r = say(router, bot, "u1", "我今天有什么事")
    assert "没有任何待办" in r, r


def test_quiz_flow():
    cfg, db, bot, router = make_env()
    r = say(router, bot, "u1", "这几道面试题帮我分析：1. 什么是CAP？")
    assert "存入题库" in r and "CAP" in r, r
    assert db.list_questions("u1"), "题库应有记录"
    r = say(router, bot, "u1", "看看题库")
    assert "CAP" in r and "未考过" in r, r
    r = say(router, bot, "u1", "考考我")
    assert "题库考试" in r and "CAP" in r, r
    assert router.sessions.get("u1"), "考试会话应建立"
    r = say(router, bot, "u1", "CAP是一致性、可用性、分区容忍性")
    assert "考过一轮" in r and "1 题" in r, r
    assert not router.sessions.get("u1"), "考完后会话应清除"
    q = db.list_questions("u1")[0]
    assert q["asked_count"] == 1 and q["good_count"] == 1, dict(q)


def test_mianjing_add():
    cfg, db, bot, router = make_env()
    from feishu_assistant.modules import quiz as quiz_mod
    orig = quiz_mod.fetch_url_text
    quiz_mod.fetch_url_text = lambda url: "面经：面试官问了什么是CAP，还问了如何设计一个Agent系统。"
    try:
        r = say(router, bot, "u1", "这篇面经帮我整理：https://example.com/mianjing/123")
        assert "已从面经提取" in r, r
        row = db.list_questions("u1")[0]
        assert row["source"] == "面经" and row["category"] == "Agent类", dict(row)
    finally:
        quiz_mod.fetch_url_text = orig


def test_mock_with_bank():
    cfg, db, bot, router = make_env()
    db.add_question("u1", "什么是CAP？", "思路：定义+取舍", "Agent")
    r = say(router, bot, "u1", "用题库模拟面试")
    session = router.sessions.get("u1")
    assert session, "应建立面试会话"
    assert "什么是CAP" in session["topic"], "题库题目应注入面试官 prompt"
    assert "自我介绍" in r, r


def test_resume_reference_after_upload():
    cfg, db, bot, router = make_env()
    # 简历已分析过时，再说"这是我的简历"应引用已有分析，而不是说"太短"
    router._ctx("u1")["last_resume_text"] = "张三的简历内容……" * 20
    r = say(router, bot, "u1", "这是我的简历")
    assert "已经分析过了" in r, r
    # 有 PDF 正在处理时，应提示稍等而不是说"太短"
    router._ctx("u1").pop("last_resume_text")
    router._ctx("u1")["pdf_pending"] = "wzy简历.pdf"
    r = say(router, bot, "u1", "这是我的简历")
    assert "正在分析" in r, r


def test_subscribe_with_time():
    cfg, db, bot, router = make_env()
    r = say(router, bot, "u1", "订阅智能体方向的论文，每天早上九点推五篇")
    assert "09:00" in r and "5 篇" in r, r
    sub = db.get_subscription("u1")
    assert sub["digest_time"] == "09:00" and sub["top_n"] == 5, dict(sub)
    from feishu_assistant.scheduler import AssistantScheduler
    from datetime import datetime as dt
    sched = AssistantScheduler(cfg, db, bot, FakeLLM())
    assert not sched._should_push(sub, dt(2026, 8, 2, 8, 55)), "未到订阅时间不推"
    assert sched._should_push(sub, dt(2026, 8, 2, 9, 5))
    assert sched._should_push(sub, dt(2026, 8, 2, 11, 30)), "睡过窗口醒来应补推"
    db.mark_pushed("u1", "2026-08-02")
    sub2 = db.get_subscription("u1")
    assert not sched._should_push(sub2, dt(2026, 8, 2, 9, 5)), "当日已推不应重复"


def test_image_mianjing():
    cfg, db, bot, router = make_env()
    bot.download_path = "/fake/mianjing.jpg"
    bot.replies.clear()
    router.handle_message("u1", "c1", "mid_img", "image", {"image_key": "k"})
    assert len(bot.replies) >= 2, "应先回执再出结果"
    assert "收到图片" in bot.replies[0], bot.replies
    last = bot.replies[-1]
    assert "提取" in last or "存入题库" in last, last
    row = db.list_questions("u1")[0]
    assert row["source"] == "搜集", dict(row)  # 无链接，按搜集入库


def test_job_prep():
    cfg, db, bot, router = make_env()
    ctx = router._ctx("u1")
    ctx["last_resume_text"] = "张三，3年后端，做过分布式调度系统……" * 10
    ctx["last_jd"] = "字节后端岗，要求Go、K8s"
    r = say(router, bot, "u1", "帮我定制简历")
    assert "改动说明" in r and "定制简历正文" in r, r
    r = say(router, bot, "u1", "写个自我介绍")
    assert "自我介绍口述稿" in r, r
    r = say(router, bot, "u1", "帮我写封求职信")
    assert "求职信正文" in r, r
    r = say(router, bot, "u1", "简历问诊")
    assert "硬伤" in r and "总评" in r, r
    rows = db.list_interview_sessions("u1")
    types = {row["type"] for row in rows}
    assert {"prep_resume", "prep_letter", "clinic"} <= types, types
    # 缺前置时的引导
    router._ctx("u2")
    r = say(router, bot, "u2", "简历问诊")
    assert "简历" in r and ("发" in r or "粘贴" in r), r


def test_jd_match():
    cfg, db, bot, router = make_env()
    r = say(router, bot, "u1", "帮我看看这个 JD 适不适合我：字节后端岗，要求3年经验、熟悉Go和K8s")
    assert "岗位解析" in r and "匹配度" in r and "需要定制简历" in r, r
    assert router._ctx("u1").get("last_jd"), "JD 应记入上下文"
    rows = db.list_interview_sessions("u1")
    assert rows and rows[0]["type"] == "jd_match", "匹配记录应归档"
    r = say(router, bot, "u1", "岗位发现与匹配")
    assert "把岗位 JD 发给我" in r, r


def test_rolling_context_and_reset():
    cfg, db, bot, router = make_env()
    llm = router.llm
    say(router, bot, "u1", "你好")
    hist = router.histories.get("u1")
    assert hist and len(hist) == 2 and hist[0]["role"] == "user", hist
    say(router, bot, "u1", "我刚才说了什么")
    # 第二次闲聊的 messages 里应带上第一轮历史
    last_call = llm.seen[-1]
    assert any(m["content"] == "你好" for m in last_call), "闲聊应注入滚动历史"
    assert len(router.histories["u1"]) == 4
    r = say(router, bot, "u1", "新建对话")
    assert "已开启新对话" in r, r
    assert len(router.histories["u1"]) == 2, "重置后只保留重置这一轮"
    assert not router.sessions.get("u1") and not router.contexts.get("u1")


def test_profile_flow():
    cfg, db, bot, router = make_env()
    r = say(router, bot, "u1", "记住我的研究方向是多智能体强化学习")
    assert "已记住" in r and "多智能体强化学习" in r, r
    r = say(router, bot, "u1", "个人画像")
    assert "研究方向：多智能体强化学习" in r, r
    from feishu_assistant.modules.profile import profile_prompt_text
    assert "多智能体强化学习" in profile_prompt_text(db, "u1"), "画像应能注入 prompt"
    # 简历要点自动提炼
    h = router.profile.extract_resume_highlights("u1", "张三，熟悉Python/Go，做过分布式调度系统……" * 10)
    assert "分布式调度" in h, h
    assert "简历要点" in profile_prompt_text(db, "u1")


def test_scheduler_init():
    from feishu_assistant.scheduler import AssistantScheduler
    cfg, db, bot, router = make_env()
    sched = AssistantScheduler(cfg, db, bot, FakeLLM())
    job_ids = {j.id for j in sched._sched.get_jobs()}
    assert {"due_todos", "morning_brief", "paper_digest"} <= job_ids, job_ids


def test_split_text():
    try:
        from feishu_assistant.bot import _split_text
    except ImportError as e:
        print("  (跳过 _split_text：lark_oapi 未安装 %s)" % e)
        return
    text = "\n".join("第%d行 " % i + "x" * 100 for i in range(200))
    chunks = _split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 3000 for c in chunks)


def test_arxiv_network():
    try:
        results = paper_mod.search_papers("retrieval augmented generation", 2)
        if not results:
            print("  (arXiv 无结果，可能网络受限，跳过断言)")
            return
        assert results[0]["title"] and results[0]["url"]
    except Exception as e:
        print("  (网络不可用，跳过 arXiv 实测：%s)" % e)


if __name__ == "__main__":
    tests = [
        ("配置加载", test_config_defaults),
        ("JSON 解析", test_parse_json),
        ("数据库增删改查", test_db_cycle),
        ("待办全流程", test_todo_flow),
        ("模拟面试全流程", test_mock_interview_flow),
        ("空白PDF处理", test_pdf_flow_blank),
        ("论文PDF总结入库", test_pdf_flow_paper),
        ("需求驱动搜索", test_need_driven_search),
        ("文献汇报流程", test_report_flow),
        ("第0层硬过滤", test_hard_filter),
        ("总结缓存复用", test_summarize_cache_reuse),
        ("日报跳过已入库", test_digest_skip_library),
        ("菜单裸指令", test_bare_commands),
        ("题库录入与考试", test_quiz_flow),
        ("个人画像", test_profile_flow),
        ("面经提取入库", test_mianjing_add),
        ("题库模拟面试", test_mock_with_bank),
        ("简历上传后引用", test_resume_reference_after_upload),
        ("滚动上下文与新建对话", test_rolling_context_and_reset),
        ("岗位匹配分析", test_jd_match),
        ("面试准备三件套", test_job_prep),
        ("图片面经识别", test_image_mianjing),
        ("考试答砸入错题本", test_quiz_bad_goes_to_mistakes),
        ("订阅时间篇数", test_subscribe_with_time),
        ("按标题生成汇报", test_report_by_title),
        ("多对话管理", test_conversation_manage),
        ("入站事件队列", test_inbox_queue),
        ("日报追问链路", test_digest_followup),
        ("定时任务注册", test_scheduler_init),
        ("长消息拆分", test_split_text),
        ("arXiv 联网搜索", test_arxiv_network),
    ]
    print("=" * 40)
    for name, fn in tests:
        check(name, fn)
    print("=" * 40)
    print("通过 %d / %d" % (len(PASSED), len(tests)))
    if FAILED:
        sys.exit(1)
    print("全部冒烟测试通过")
