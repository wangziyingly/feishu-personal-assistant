"""岗位发现与匹配：JD 解析（硬性要求/加分项/隐含偏好）+ 简历匹配度打分。

流程：用户发来 JD（粘贴或链接）→ 解析岗位要求 → 结合个人画像/简历打分 →
给出「投/不投/需要定制简历」建议，以及简历该改哪、该补什么技能。
JD 会记入 ctx（last_jd），可直接接「用这个 JD 模拟面试」「根据JD生成面试题」。
"""
import re

from ..llm import LLMError
from .profile import profile_prompt_text
from .quiz import fetch_url_text

JD_PARSE_PROMPT = """你是资深招聘专家。解析以下岗位 JD，输出 JSON：
{"summary": "一句话岗位定位",
 "hard_requirements": ["硬性要求（学历/年限/核心技术栈）"],
 "plus_items": ["加分项"],
 "hidden_preferences": ["隐含偏好（从措辞/公司类型推断，标注'推断'）"]}
只输出 JSON。JD 原文：
%s"""

MATCH_PROMPT = """你是求职顾问。根据岗位要求和候选人背景做匹配分析，输出 JSON：
{"score": 0-100 的整数匹配分,
 "verdict": "投 / 不投 / 需要定制简历（三选一）",
 "reasons": ["判断依据（2-4 条，具体到要求与背景的对应关系）"],
 "resume_edits": ["简历需要添加/修改的部分（具体可操作，如'把xx项目提到第一位并补充xx指标'）"],
 "skills_to_learn": ["需要补的技能知识，按优先级排序"]}
只输出 JSON，不要客套，依据不足的项写'候选人背景不足，无法判断'。

岗位要求：
硬性：%s
加分：%s
隐含偏好：%s

候选人背景：
%s
"""

GUIDE = ("把岗位 JD 发给我（直接粘贴文字，或发招聘链接），我会：\n"
         "1. 解析出硬性要求、加分项、隐含偏好\n"
         "2. 结合你的简历和画像打匹配分，给出「投/不投/需要定制简历」的建议\n"
         "3. 告诉你简历要改哪里、需要补什么技能\n\n"
         "小提示：先发过简历或完善个人画像（菜单：助理→个人画像），匹配会打得更准。")

TAILOR_PROMPT = """你是资深简历顾问，精通 ATS（简历筛选系统）关键词优化。
基于候选人的简历原文和目标 JD，输出一版定制简历：
1. 开头给 3-5 条「本版改动说明」（突出了什么、对齐了哪些 JD 关键词、删减了什么及原因）；
2. 然后给出完整定制简历正文：核心优势摘要 → 技能关键词（对齐 ATS）→ 项目与经历
   （按 JD 相关性重排，每条用 STAR+量化结果改写）→ 教育背景；
3. 绝不编造候选人没有的经历和数字；需要补数据的地方用【待补充：…】标注；
4. 中文输出。

简历原文：
%s

目标 JD：
%s"""

INTRO_PROMPT = """你是求职表达教练。基于候选人简历和目标 JD，写一段 60-90 秒的中文自我介绍口述稿：
结构 = 开场定位一句话 → 2-3 个与 JD 最匹配的经历亮点（含量化结果）→ 为什么是这家公司这个岗 → 简短收尾。
口语化、有记忆点，不要书面腔和形容词堆砌。只输出口述稿本身。

简历：%s

JD：%s"""

LETTER_PROMPT = """你是求职表达教练。基于候选人简历和目标 JD，写一封 300-400 字的中文求职信：
结构 = 对岗位/业务的理解 → 匹配的三点理由（具体经历支撑）→ 能为团队带来什么 → 专业收尾。
不堆砌形容词，不出现'贵公司'这类陈旧措辞。只输出求职信本身。

简历：%s

JD：%s"""

CLINIC_PROMPT = """你是阅简历无数的资深 HR，以挑剔眼光审这份简历%s。输出 JSON：
{"fatal": ["可能直接被刷掉的硬伤"],
 "issues": [{"problem": "问题描述", "fix": "具体改法"}],
 "overall": "总评（2-3 句，给出投递胜算的直白判断）"}
只输出 JSON，一针见血，不客套。简历原文：
%s"""


class JobMatchModule:
    def __init__(self, cfg, db, llm, bot=None):
        self.cfg = cfg
        self.db = db
        self.llm = llm
        self.bot = bot

    def handle_match(self, user_id, args, ctx):
        jd = (args.get("jd") or "").strip()
        if not jd:
            return GUIDE
        urls = re.findall(r"https?://[^\s）)\]]+", jd)
        if urls:
            fetched = fetch_url_text(urls[0])
            if fetched:
                jd = fetched
            elif len(re.sub(r"https?://\S+", "", jd).strip()) < 10:
                return "这个链接我抓不到内容（可能需要登录）。把 JD 文字直接粘贴给我就行。"
        ctx["last_jd"] = jd[:6000]

        try:
            parsed = self.llm.chat_json([{"role": "user", "content": JD_PARSE_PROMPT % jd[:5000]}])
        except LLMError:
            raise
        hard = "、".join(str(x) for x in (parsed.get("hard_requirements") or []))
        plus = "、".join(str(x) for x in (parsed.get("plus_items") or []))
        hidden = "、".join(str(x) for x in (parsed.get("hidden_preferences") or []))

        bg_parts = [profile_prompt_text(self.db, user_id)]
        resume = ctx.get("last_resume_text") or ""
        if resume:
            bg_parts.append("简历全文（节选）：" + resume[:3000])
        bg = "\n".join(p for p in bg_parts if p) or "（候选人背景为空：未提供简历/画像，以下判断仅基于 JD 常识）"

        match = self.llm.chat_json([
            {"role": "user", "content": MATCH_PROMPT % (hard, plus, hidden, bg)},
        ])

        lines = ["【岗位解析】%s\n" % (parsed.get("summary") or "")]
        lines.append("硬性要求：%s" % (hard or "未识别"))
        lines.append("加分项：%s" % (plus or "无"))
        if hidden:
            lines.append("隐含偏好：%s" % hidden)
        lines.append("\n【匹配度：%s 分】建议：%s" % (match.get("score", "?"), match.get("verdict") or "?"))
        reasons = match.get("reasons") or []
        if reasons:
            lines.append("依据：" + "；".join(str(r) for r in reasons))
        edits = match.get("resume_edits") or []
        if edits:
            lines.append("\n简历要改：")
            lines.extend("· " + str(e) for e in edits)
        skills = match.get("skills_to_learn") or []
        if skills:
            lines.append("\n要补的技能（按优先级）：")
            lines.extend("· " + str(s) for s in skills)
        lines.append("\nJD 已记录。回复「用这个 JD 模拟面试」实战演练，或「根据JD生成面试题」出题集。")
        reply = "\n".join(lines)
        self.db.add_interview_session(user_id, "jd_match", {"jd": jd[:2000]}, reply)
        return reply

    # ---------- 面试准备：定制简历 / 自我介绍·求职信 / 简历问诊 ----------
    def _check_prereq(self, ctx, need_jd=True):
        """返回 (resume, jd, 缺失引导语 or None)。"""
        resume = (ctx.get("last_resume_text") or "").strip()
        jd = (ctx.get("last_jd") or "").strip()
        missing = []
        if not resume:
            missing.append("简历（发 PDF 或粘贴文字给我）")
        if need_jd and not jd:
            missing.append("岗位 JD（粘贴给我，或先做「岗位发现与匹配」）")
        if missing:
            return resume, jd, "还缺%s。准备好后再发指令。" % "和".join(missing)
        return resume, jd, None

    def tailor_resume(self, user_id, ctx):
        """定制版简历：突出相关项目、对齐 JD 关键词（过 ATS）。"""
        resume, jd, err = self._check_prereq(ctx)
        if err:
            return err
        from ..bot import make_stream
        stream = make_stream(self.bot, ctx)
        try:
            reply = self.llm.chat([
                {"role": "user", "content": TAILOR_PROMPT % (resume[:7000], jd[:4000])},
            ], on_delta=stream.update if stream else None)
        except Exception as e:
            if stream:
                stream.close("（生成中断：%s，请稍后再试）" % e)
            raise
        reply += "\n\n（可继续说「项目部分再突出一下 xxx」让我迭代；说「新建对话」可清空重来）"
        self.db.add_interview_session(user_id, "prep_resume", {"jd": jd[:1000]}, reply)
        if stream:
            stream.close(reply)
            return None
        return reply

    def write_letter(self, user_id, args, ctx):
        """自我介绍口述稿 / 求职信。"""
        kind = (args.get("kind") or "").strip()
        resume, jd, err = self._check_prereq(ctx)
        if err:
            return err
        if "求职信" in kind:
            prompt, tag = LETTER_PROMPT, "求职信"
        else:
            prompt, tag = INTRO_PROMPT, "自我介绍"
        reply = self.llm.chat([
            {"role": "user", "content": prompt % (resume[:6000], jd[:4000])},
        ])
        reply = "【%s】\n\n%s\n\n（想换风格直接说，比如「再正式一点」「缩短到 30 秒」）" % (tag, reply)
        self.db.add_interview_session(user_id, "prep_letter", {"kind": tag}, reply)
        return reply

    def clinic(self, user_id, ctx):
        """简历问诊：HR 视角挑毛病。有 JD 时按 JD 针对性审，没有则按通用标准。"""
        resume, jd, err = self._check_prereq(ctx, need_jd=False)
        if err:
            return err
        jd_note = "，结合以下目标 JD 审：\n" + jd[:3000] if jd else "（无特定目标岗位，按通用标准审）"
        data = self.llm.chat_json([
            {"role": "user", "content": CLINIC_PROMPT % (jd_note, resume[:6000])},
        ])
        lines = ["【简历问诊】\n"]
        fatal = data.get("fatal") or []
        if fatal:
            lines.append("可能直接被刷的硬伤：")
            lines.extend("· " + str(f) for f in fatal)
        issues = data.get("issues") or []
        if issues:
            lines.append("\n问题与改法：")
            for i, it in enumerate(issues[:6], 1):
                lines.append("%d. %s" % (i, it.get("problem") or ""))
                if it.get("fix"):
                    lines.append("   改法：%s" % it["fix"])
        lines.append("\n总评：%s" % (data.get("overall") or ""))
        lines.append("\n（改完后把新简历发我，可以再诊一轮）")
        reply = "\n".join(lines)
        self.db.add_interview_session(user_id, "clinic", {"jd": jd[:1000]}, reply)
        return reply
