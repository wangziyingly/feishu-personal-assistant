"""消息路由：LLM 意图识别 + 会话状态管理 + 分发到三大模块。

三类状态：
- histories：滚动对话历史（每用户最近若干轮，截断存储），意图分类和闲聊时注入，
  让"展开讲讲""再改一版"这类追问/改进意见能看懂上下文；「新建对话」可一键清空
- sessions：阻塞式多轮会话（模拟面试/题库考试），进行中时用户消息直接续聊，不再走意图分类
- contexts：被动上下文（最近待办列表/搜索结果/刚上传的PDF/简历/JD），供「第2条」「第1篇」这类指代解析
"""
import threading
from datetime import datetime

from .config import LLM_GUIDE, PDF_DIR
from .llm import LLMError
from .bitable import BitableSync
from .wiki_kb import WikiKB
from .modules.interview import InterviewModule
from .modules.jobmatch import JobMatchModule
from .modules.paper import PaperModule
from .modules.profile import ProfileModule
from .modules.quiz import QuizModule
from .modules.todo import TodoModule

HISTORY_MAX_PAIRS = 10     # 滚动历史保留轮数（超出丢弃最旧的）
HISTORY_MSG_CHARS = 500    # 每条消息存入历史的截断长度（控制 token）
CLASSIFY_HISTORY = 6       # 意图分类时带入的最近消息条数（3 轮）

INTENT_PROMPT = """你是飞书个人助手的意图分类器。当前时间：{now}（{weekday}）。
把用户消息分类为以下意图之一，并抽取参数，输出 JSON：{{"intent": "...", "args": {{...}}}}

意图列表：
1. todo_create —— 创建待办/提醒。args: {{"task": "事项", "remind_at": "YYYY-MM-DD HH:MM 或 null"}}
   例：「明天下午3点提醒我交周报」→ {{"intent":"todo_create","args":{{"task":"交周报","remind_at":"{tomorrow} 15:00"}}}}
   例：「记一下要买牛奶」→ {{"intent":"todo_create","args":{{"task":"买牛奶","remind_at":null}}}}
2. todo_manage —— 查询/完成/删除待办。args: {{"action":"query|complete|delete","index":序号或null,"keyword":关键词或null}}
   例：「我今天有什么事」→ action=query；「完成了第2条」→ action=complete,index=2；「删掉买牛奶」→ action=delete,keyword=买牛奶
3. paper_search —— 搜索文献（支持描述研究需求）。args: {{"query":"用户的需求原文或关键词"}}
   例：「帮我搜一下 RAG 最新的论文」→ {{"intent":"paper_search","args":{{"query":"RAG 最新论文"}}}}
   例：「我在做多智能体系统的信用分配，帮我找相关最新方法」→ {{"intent":"paper_search","args":{{"query":"多智能体系统的信用分配 最新方法"}}}}
4. paper_summarize_index —— 快速总结上次搜索结果中的第 N 篇（摘要级解读）。args: {{"index":1}}
   例：「总结第1篇」
5. paper_report —— 对论文做深度文献汇报/精读（组会汇报级）。args: {{"index": 序号或null, "title": "论文标题或null"}}
   例：「汇报第1篇」「精读第2篇」→ index 为序号；「生成文献汇报」「汇报这篇」（针对刚上传的 PDF）→ index=null；
   「汇报 Reflexion 这篇」「就刚才说的那篇 XX」→ {{"intent":"paper_report","args":{{"index":null,"title":"Reflexion"}}}}
6. paper_library —— 查询/提问自己的文献库。args: {{"query":"问题或关键词"}}
   例：「我文献库里有哪些 RAG 的论文」「我之前那篇讲 agent 的论文用了什么方法」
7. paper_subscribe —— 订阅/取消每日论文推送。args: {{"keywords":"研究需求或方向","enable":true或false,"time":"HH:MM 或 null","top_n":数字或null}}
   例：「订阅 RAG 和 agent 方向的论文」→ enable=true；「取消论文订阅」→ enable=false；
   「订阅智能体方向的论文，每天早上九点推五篇」→ time="09:00", top_n=5
8. interview_mock —— 开始模拟面试。args: {{"topic":"岗位/方向/JD 原文"}}
   例：「模拟一场后端开发面试」「用这个 JD 考考我」「用这份简历模拟面试」
   例：「用题库模拟面试」→ {{"intent":"interview_mock","args":{{"topic":"题库"}}}}；「用题库里Agent的题考我」→ topic="题库 Agent"
9. interview_jd —— 根据 JD 生成面试题集。args: {{"jd":"JD 原文"}}
   例：「根据这个 JD 出点面试题：……」
10. interview_resume —— 简历深挖提问。args: {{"resume_text":"简历文本"}}
    例：「这是我的简历，帮我看看面试官会怎么问：……」
11. interview_review —— 面试错题本。路由规则：**仅当用户主动描述自己亲身经历的真实面试**中被问倒的问题（如"我今天面试被问到…"）才走此意图。args: {{"action":"add|query","questions":"问题原文","category":"分类或null"}}
    例：「今天面试被问到：什么是CAP？缓存穿透怎么解决？」→ action=add；「看看错题本」「看看技术类的错题」→ action=query
    注意1：网上/社交媒体看到的面经、别人分享的面试题、图片截图里的题目，**都不是**错题本，归 intent 13
    注意2：模拟面试/题库考试中答不上来的题由系统在练习结束时自动记入错题本，用户说"把模拟面试的错题记下来"时，回复它已自动记录即可，不必重复录入
12. interview_history —— 查看历史面试记录。args: {{}}
    例：「面试记录」
13. quiz_add —— 录入面试题或面经（搜集的题目、网上看到的面经、图片截图转录的题目、面经文章/链接）。args: {{"questions":"原文或链接","category":"用户指定的分类或null"}}
    例：「这几道题帮我分析一下怎么回答：……」「这篇面经帮我整理一下：……」「这几道Agent的题存进题库」→ category="Agent"
    区分：只有用户亲口说是**自己面试被问的**（如"我今天面试被问到"），才归 intent 11
14. quiz_query —— 浏览题库。args: {{"category":"分类或null"}}
    例：「看看题库」「题库里有哪些系统设计的题」
15. quiz_start —— 从题库抽题考用户。args: {{"category":"分类或null"}}
    例：「考考我」「从题库里考我」「考我几道算法题」
    注意区分：「用这份 JD 考考我」「模拟面试考考我」是 intent 8（模拟面试），不是从题库抽题
16. profile_update —— 记住/更新用户的个人背景。args: {{"research_direction":"研究方向或null","target_companies":"目标公司/岗位或null","extra":"其他备注或null"}}
    例：「记住我的研究方向是多智能体」→ research_direction；「我的目标公司是字节」→ target_companies；「目标公司改成腾讯」→ target_companies（覆盖）
17. profile_query —— 查看用户画像（记住了用户的哪些信息）。args: {{}}
    例：「个人画像」「看看我的画像」「你记得我的研究方向吗」
18. jd_match —— 岗位发现与匹配：解析 JD 并与候选人背景匹配打分。args: {{"jd":"JD 原文或链接"}}
    例：「帮我看看这个 JD 适不适合我：……」「这个岗位我能投吗」「分析一下这个 JD：……」
19. prep_resume —— 针对已发过的 JD 定制一版简历（ATS 对齐）。args: {{}}
    例：「帮我定制简历」「针对这个 JD 改一版简历」「定制版简历」
20. prep_letter —— 写自我介绍或求职信。args: {{"kind":"自我介绍 或 求职信"}}
    例：「写个一分钟自我介绍」→ kind=自我介绍；「帮我写封求职信」→ kind=求职信
21. resume_clinic —— 简历问诊：HR 视角挑毛病。args: {{}}
    例：「简历问诊」「帮我挑挑简历的毛病」「HR 会怎么看我的简历」
22. weakness_analysis —— 跨错题/题库/面试记录分析用户的薄弱模式。args: {{}}
    例：「分析我的薄弱点」「我的面试薄弱模式」「错题规律分析」「我最近老挂在什么地方」
23. chat —— 以上都不匹配的闲聊/通用问答。args: {{"text":"用户原文"}}

注意：只输出 JSON；拿不准意图时选 chat；时间一律基于当前时间推算为具体时刻。
分类对象是对话中最后一条用户消息，之前的消息只是帮助理解指代的上下文。"""

WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# 菜单裸指令：与飞书机器人自定义菜单的菜单项一一对应（菜单点击后发送的就是菜单名本身）。
# 精确命中时跳过 LLM 意图分类（省一次调用，也避免模型把「搜文献」当成搜索关键词），
# 直接以"无参数"意图分发，由各模块反问用户具体要做什么。
BARE_COMMANDS = {
    "搜文献": ("paper_search", {"query": ""}),
    "我要搜索文献": ("paper_search", {"query": ""}),
    "生成文献汇报": ("paper_report", {"index": None}),
    "看看我的文献库": ("paper_library", {"query": ""}),
    "订阅论文日报": ("paper_subscribe", {"keywords": "", "enable": True}),
    "取消论文订阅": ("paper_subscribe", {"keywords": "", "enable": False}),
    "模拟面试": ("interview_mock", {"topic": ""}),
    "开始模拟面试": ("interview_mock", {"topic": ""}),
    "根据JD生成面试题": ("interview_jd", {"jd": ""}),
    "分析我的简历": ("interview_resume", {"resume_text": ""}),
    "看看错题本": ("interview_review", {"action": "query"}),
    "面试记录": ("interview_history", {}),
    "我今天有什么事": ("todo_manage", {"action": "query"}),
    "新增一个待办": ("todo_create", {"task": "", "remind_at": None}),
    "个人画像": ("profile_query", {}),
    "看看我的画像": ("profile_query", {}),
    "新建对话": ("reset_context", {}),
    "清空上下文": ("reset_context", {}),
    "岗位发现与匹配": ("jd_match", {"jd": ""}),
    "搜集的全网面经": ("quiz_add", {"questions": ""}),
    "面试准备": ("job_prep", {}),
    "复盘与迭代": ("interview_review", {"action": "add", "questions": ""}),
}


class Router:
    def __init__(self, cfg, db, llm, bot):
        self.cfg = cfg
        self.db = db
        self.llm = llm
        self.bot = bot
        self.todo = TodoModule(cfg, db, llm)
        paper_sync = BitableSync(cfg, app_token=cfg.paper_bitable_token)
        self.paper = PaperModule(cfg, db, llm, paper_sync)
        bitable = BitableSync(cfg)
        wiki = WikiKB(cfg, db)
        self.interview = InterviewModule(cfg, db, llm, bitable)
        self.quiz = QuizModule(cfg, db, llm, bitable, wiki)
        self.profile = ProfileModule(cfg, db, llm)
        self.jobmatch = JobMatchModule(cfg, db, llm)
        self.sessions = {}   # user_id -> 阻塞式会话
        self.contexts = {}   # user_id -> 被动上下文
        self.histories = {}  # user_id -> 滚动对话历史
        self._lock = threading.Lock()

    # ---------- 入口（bot 回调） ----------
    def handle_message(self, user_id, chat_id, message_id, msg_type, content):
        self.db.upsert_user(user_id, chat_id)
        if not self.cfg.llm_configured:
            self.bot.reply(message_id, LLM_GUIDE)
            return
        try:
            if msg_type == "file":
                self.bot.reply(message_id, "收到文件（%s），正在解析分析，内容多的话可能要一两分钟，好了马上发你。"
                               % (content.get("file_name") or ""))
                reply = self._handle_file(user_id, message_id, content)
                self._record(user_id, "[PDF文件] %s" % (content.get("file_name") or ""), reply)
            elif msg_type == "text":
                reply = self._handle_text(user_id, content.get("text", ""))
                self._record(user_id, content.get("text", ""), reply)
            elif msg_type == "image":
                self.bot.reply(message_id, "收到图片，正在识别图中内容（面经/JD/简历截图都可以直接发）…")
                reply = self._handle_image(user_id, message_id, content)
                self._record(user_id, "[图片]", reply)
            else:
                reply = "我目前只理解文字、图片和 PDF 文件消息，其他类型（语音等）暂时处理不了。"
        except LLMError as e:
            reply = "大模型调用出了点问题：%s\n请稍后再试，或检查 config.yaml 里的 llm 配置。" % e
        except Exception as e:
            reply = "处理时出错了：%s" % e
        self.bot.reply(message_id, reply)

    # ---------- 文本消息 ----------
    def _handle_text(self, user_id, text, note=None):
        text = (text or "").strip()
        if not text:
            return "没收到内容，再说一遍？"
        ctx = self._ctx(user_id)
        session = self._session(user_id)

        # 阻塞式会话（模拟面试/题库考试）优先
        if session and session.get("module") == "interview_mock":
            if "结束面试" in text or text in ("结束", "结束吧"):
                self._clear_session(user_id)
                return self.interview.end_mock(user_id, session)
            if "退出面试" in text or text == "取消面试":
                self._clear_session(user_id)
                return "已退出本场模拟面试（未保存记录）。"
            return self.interview.continue_mock(user_id, text, session)
        if session and session.get("module") == "quiz":
            if text in ("结束", "结束吧", "结束考试", "不考了", "退出"):
                self._clear_session(user_id)
                return self.quiz.end_quiz(user_id, session)
            reply = self.quiz.continue_quiz(user_id, text, session)
            if session.get("done"):  # 题库已考完，自动清除会话
                self._clear_session(user_id)
            return reply

        bare = BARE_COMMANDS.get(text)
        if bare:
            intent, args = bare
        else:
            # note：给分类器的来源提示（如"图片转录"），帮助区分面经/错题，不进入分发参数
            cls_input = ("%s\n（背景说明：%s）" % (text, note)) if note else text
            intent_data = self._classify(user_id, cls_input)
            intent = intent_data.get("intent", "chat")
            args = intent_data.get("args") or {}

        if intent == "reset_context":
            self._clear_session(user_id)
            with self._lock:
                self.contexts.pop(user_id, None)
                self.histories.pop(user_id, None)
            return ("已开启新对话：对话历史、指代点（上次搜索/简历/JD）、"
                    "进行中的会话都已清空。我们从这里重新开始。")
        if intent == "todo_create":
            return self.todo.handle_create(user_id, args)
        if intent == "todo_manage":
            return self.todo.handle_manage(user_id, args, ctx)
        if intent == "paper_search":
            return self.paper.handle_search(user_id, args, ctx)
        if intent == "paper_summarize_index":
            return self.paper.handle_summarize_index(user_id, args, ctx)
        if intent == "paper_report":
            return self.paper.handle_report(user_id, args, ctx)
        if intent == "paper_library":
            return self.paper.handle_library(user_id, args, text)
        if intent == "paper_subscribe":
            return self.paper.handle_subscribe(user_id, args)
        if intent == "interview_mock":
            session, reply = self.interview.start_mock(user_id, args, ctx)
            if session:
                self._set_session(user_id, session)
            return reply
        if intent == "interview_jd":
            return self.interview.handle_jd(user_id, args, ctx)
        if intent == "interview_resume":
            resume_text = args.get("resume_text") or ""
            if len(resume_text.strip()) < 50:
                if ctx.get("pdf_pending"):
                    return "你的 PDF（%s）我正在分析，马上就好，稍等片刻～" % ctx["pdf_pending"]
                if ctx.get("last_resume_text"):
                    return ("你的简历我已经分析过了。回复「用这份简历模拟面试」开始实战演练，"
                            "或把新的简历发给我。")
            reply = self.interview.handle_resume(user_id, resume_text, ctx)
            if "简历内容太短" not in reply:
                self.profile.extract_resume_highlights(user_id, resume_text)
            return reply
        if intent == "profile_update":
            return self.profile.handle_update(user_id, args)
        if intent == "profile_query":
            return self.profile.handle_query(user_id)
        if intent == "jd_match":
            return self.jobmatch.handle_match(user_id, args, ctx)
        if intent == "prep_resume":
            return self.jobmatch.tailor_resume(user_id, ctx)
        if intent == "prep_letter":
            return self.jobmatch.write_letter(user_id, args, ctx)
        if intent == "resume_clinic":
            return self.jobmatch.clinic(user_id, ctx)
        if intent == "weakness_analysis":
            return self.interview.weakness_analysis(user_id)
        if intent == "job_prep":
            return ("面试准备区，直接回复对应指令：\n"
                    "· 「定制简历」——针对你发过的 JD 改一版 ATS 友好的简历\n"
                    "· 「写个自我介绍」/「写封求职信」——基于简历+JD 生成\n"
                    "· 「简历问诊」——HR 视角挑毛病，可反复迭代\n"
                    "· 「用题库模拟面试」「用这份简历模拟面试」——实战演练\n\n"
                    "前置：发过简历 + 发过 JD（或先做一次「岗位发现与匹配」）效果最佳。")
        if intent == "interview_review":
            return self.interview.handle_review(user_id, args, text)
        if intent == "interview_history":
            return self.interview.list_sessions(user_id)
        if intent == "quiz_add":
            return self.quiz.handle_add(user_id, args, text)
        if intent == "quiz_query":
            return self.quiz.handle_query(user_id, args)
        if intent == "quiz_start":
            session, reply = self.quiz.start_quiz(user_id, args, ctx)
            if session:
                self._set_session(user_id, session)
            return reply
        # chat 兜底
        return self._chat(user_id, text)

    # ---------- 图片消息（面经/JD/简历截图等） ----------
    def _handle_image(self, user_id, message_id, content):
        image_key = content.get("image_key") or ""
        path = self.bot.download_image(message_id, image_key, PDF_DIR)
        if not path:
            return "图片下载失败了，重发一次试试。"
        try:
            text = self.llm.chat_vision(
                "请完整转录这张图片里的文字内容（保持题目编号和结构），不要评论、不要补充。"
                "如果是面试题/面经/JD/简历的截图，原样转录即可。", path)
        except LLMError as e:
            return "图片识别失败：%s。可以把图中文字复制发我。" % e
        text = (text or "").strip()
        if len(text) < 5:
            return ("这张图里没识别出有效文字。面经/JD/简历截图可以发清晰一点的图，"
                    "或直接把文字粘贴给我。")
        # 转录文本走正常意图管线：面经截图 → 自动入库分析，JD 截图 → 岗位匹配等
        return self._handle_text(user_id, text,
                                 note="这段文字是用户发送图片的转录内容；若内容是面试题/面经截图，应归 quiz_add（题库/面经），不是用户自己的错题")

    # ---------- 文件消息（PDF） ----------
    def _handle_file(self, user_id, message_id, content):
        file_name = content.get("file_name") or ""
        ctx = self._ctx(user_id)
        ctx["pdf_pending"] = file_name  # 标记"有 PDF 正在处理"，供随后的文字消息引用
        try:
            return self._do_handle_file(user_id, message_id, content, file_name)
        finally:
            ctx.pop("pdf_pending", None)

    def _do_handle_file(self, user_id, message_id, content, file_name):
        file_key = content.get("file_key") or ""
        if not file_name.lower().endswith(".pdf"):
            return "目前只支持 PDF 文件。论文或简历的 PDF 直接发我即可。"
        path = self.bot.download_file(message_id, file_key, PDF_DIR, file_name)
        if not path:
            return "文件下载失败了，请重发一次试试。"
        from .modules.paper import extract_pdf_text

        try:
            head, _ = extract_pdf_text(path, max_pages=3, max_chars=3000)
        except Exception:
            return "这个 PDF 打不开或已损坏，换个文件试试。"
        doc_type = self.paper.classify_doc(head)
        if doc_type == "resume":
            full_text, _ = extract_pdf_text(path)
            reply = "收到你的简历，先做一轮深挖分析：\n\n" + self.interview.handle_resume(
                user_id, full_text, self._ctx(user_id))
            if self.profile.extract_resume_highlights(user_id, full_text):
                reply += "\n\n（已把简历要点记入你的个人画像，回复「个人画像」可查看）"
            return reply
        # paper / other 都走论文式结构化总结
        reply = self.paper.summarize_pdf(user_id, path, file_name)
        if "已存入你的文献库" in reply:
            self._ctx(user_id)["last_pdf_path"] = path  # 供「生成文献汇报」指代
        return reply

    # ---------- 意图分类 ----------
    def _classify(self, user_id, text):
        now = datetime.now()
        from datetime import timedelta
        prompt = INTENT_PROMPT.format(
            now=now.strftime("%Y-%m-%d %H:%M"),
            weekday=WEEKDAYS[now.weekday()],
            tomorrow=(now + timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        try:
            # 带入最近几轮历史，让"展开讲讲""换个方向"这类指代能被正确分类
            messages = [{"role": "system", "content": prompt}]
            messages.extend(self._history(user_id)[-CLASSIFY_HISTORY:])
            messages.append({"role": "user", "content": text})
            data = self.llm.chat_json(messages)
            if "intent" not in data:
                return {"intent": "chat", "args": {"text": text}}
            return data
        except LLMError:
            return {"intent": "chat", "args": {"text": text}}

    # ---------- 闲聊 ----------
    def _chat(self, user_id, text):
        # 带完整滚动历史：对输出的改进意见（"太简略了""再改一版"）能看懂在改什么
        messages = [{"role": "system", "content":
            "你是用户的飞书个人助手，擅长文献、面试、待办管理，也能正常聊天和回答问题。"
            "回答简洁、中文。若用户的问题适合用助手的功能解决，顺带提示一句用法。"
            "铁律：你只是对话接口，不能执行任何实际动作（入库、订阅、建待办、发推送、修改数据）。"
            "绝不声称「已入库/已订阅/已记下/已设置」这类已完成状态——这些只能由功能模块真正执行。"
            "用户提出这类请求时，告诉他对应的指令说法（如「汇报第1篇」「订阅xx方向的论文」），"
            "或如实说明当前缺少前置条件（如需要先发 PDF）。"}]
        messages.extend(self._history(user_id))
        messages.append({"role": "user", "content": text})
        return self.llm.chat(messages)

    # ---------- 滚动历史 ----------
    def _history(self, user_id):
        with self._lock:
            return self.histories.setdefault(user_id, [])

    def _record(self, user_id, user_text, reply):
        hist = self._history(user_id)
        hist.append({"role": "user", "content": (user_text or "")[:HISTORY_MSG_CHARS]})
        hist.append({"role": "assistant", "content": (reply or "")[:HISTORY_MSG_CHARS]})
        if len(hist) > HISTORY_MAX_PAIRS * 2:
            del hist[:-HISTORY_MAX_PAIRS * 2]

    # ---------- 状态管理 ----------
    def _session(self, user_id):
        with self._lock:
            return self.sessions.get(user_id)

    def _set_session(self, user_id, session):
        with self._lock:
            self.sessions[user_id] = session

    def _clear_session(self, user_id):
        with self._lock:
            self.sessions.pop(user_id, None)

    def _ctx(self, user_id):
        with self._lock:
            return self.contexts.setdefault(user_id, {})
