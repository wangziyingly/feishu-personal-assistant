"""个人画像模块：记住用户的研究方向、目标公司、简历要点，供各模块个性化。

- 用户可以显式说「记住我的研究方向是xxx」来更新
- 发简历（PDF 或文本）时自动提炼简历要点存入画像
- profile_prompt_text() 是给其他模块的注入点：论文检索排序、模拟面试等
  在 prompt 里附带一句"你是谁"，越用越懂你
"""
from ..llm import LLMError

FIELDS = [
    ("research_direction", "研究方向"),
    ("target_companies", "目标公司/岗位"),
    ("resume_highlights", "简历要点"),
    ("extra", "其他备注"),
]


def profile_prompt_text(db, user_id):
    """生成供 prompt 注入的用户背景描述；无画像返回空串。"""
    row = db.get_profile(user_id)
    if not row:
        return ""
    parts = ["%s：%s" % (label, row[key]) for key, label in FIELDS if row[key]]
    return "用户背景——" + "；".join(parts) + "。" if parts else ""


class ProfileModule:
    def __init__(self, cfg, db, llm):
        self.cfg = cfg
        self.db = db
        self.llm = llm

    # ---------- 查看 ----------
    def handle_query(self, user_id):
        row = self.db.get_profile(user_id)
        if not row or not any(row[k] for k, _ in FIELDS):
            return ("我还没有记住你的信息。可以直接告诉我，比如：\n"
                    "· 记住我的研究方向是多智能体强化学习\n"
                    "· 我的目标公司是字节跳动和小红书\n"
                    "· 发一份简历 PDF 给我，我会自动提炼要点\n"
                    "之后论文推荐、模拟面试都会基于你的背景来做。")
        lines = ["这是我记住的关于你的信息：\n"]
        for key, label in FIELDS:
            if row[key]:
                lines.append("· %s：%s" % (label, row[key]))
        lines.append("\n想修改的话直接说，比如「我的目标公司改成xxx」。")
        return "\n".join(lines)

    # ---------- 更新 ----------
    def handle_update(self, user_id, args):
        fields = {k: (args.get(k) or "").strip() if isinstance(args.get(k), str) else ""
                  for k, _ in FIELDS}
        fields = {k: v for k, v in fields.items() if v}
        if not fields:
            return ("想让我记住什么？比如：\n"
                    "· 记住我的研究方向是 RAG 检索增强\n"
                    "· 我的目标公司是字节跳动的后端岗")
        self.db.upsert_profile(user_id, **fields)
        labels = "、".join(dict(FIELDS)[k] for k in fields)
        return ("已记住（更新了 %s）。\n\n%s" % (labels, self.handle_query(user_id)))

    # ---------- 简历要点自动提炼（发简历时由 router 调用） ----------
    def extract_resume_highlights(self, user_id, resume_text):
        """从简历文本提炼要点存入画像；失败静默跳过。返回提炼出的要点或空串。"""
        try:
            data = self.llm.chat_json([
                {"role": "user", "content":
                    "从以下简历中提炼 3-5 条要点（核心技能、最有分量的项目/经历、教育背景），"
                    "每条一句话，输出 JSON：{\"highlights\": \"要点1；要点2；要点3\"}。简历：\n%s"
                    % resume_text[:6000]},
            ])
            highlights = str(data.get("highlights") or "").strip()
        except LLMError:
            return ""
        if highlights:
            self.db.upsert_profile(user_id, resume_highlights=highlights)
        return highlights
