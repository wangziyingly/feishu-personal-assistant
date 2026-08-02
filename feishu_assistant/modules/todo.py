"""待办事项模块：自然语言创建、查询、完成、删除 + 提醒与早报。

时间解析不在这里做——router 的意图分类已经带着当前时间把
"明天下午3点"解析成了 ISO 时间，这里只负责校验、存库和格式化。
"""
from datetime import datetime

TIME_FMT = "%Y-%m-%d %H:%M"


def _fmt_remind(remind_at):
    """把 '2026-07-31 15:00' 显示成 '今天 15:00' / '明天 15:00' / '08-02 15:00'。"""
    try:
        dt = datetime.strptime(remind_at, TIME_FMT)
    except (ValueError, TypeError):
        return remind_at or ""
    today = datetime.now().date()
    delta = (dt.date() - today).days
    hm = dt.strftime("%H:%M")
    if delta == 0:
        return "今天 " + hm
    if delta == 1:
        return "明天 " + hm
    if delta == 2:
        return "后天 " + hm
    return dt.strftime("%m-%d ") + hm


class TodoModule:
    def __init__(self, cfg, db, llm):
        self.cfg = cfg
        self.db = db
        self.llm = llm

    # ---------- 创建 ----------
    def handle_create(self, user_id, args):
        task = (args.get("task") or "").strip()
        remind_at = (args.get("remind_at") or "").strip() or None
        if not task:
            return "没听清要提醒你做什么，能再说一遍吗？比如：明天下午3点提醒我交周报。"
        if remind_at:
            try:
                datetime.strptime(remind_at, TIME_FMT)
            except ValueError:
                remind_at = None
        self.db.add_todo(user_id, task, remind_at)
        if remind_at:
            return "已记下：%s\n提醒时间：%s" % (task, _fmt_remind(remind_at))
        return "已记下：%s（未设提醒时间，可在查询后说「提醒我x点做」重建）" % task

    # ---------- 查询 / 完成 / 删除 ----------
    def handle_manage(self, user_id, args, ctx):
        action = (args.get("action") or "query").strip()
        if action == "query":
            return self._query(user_id, ctx)
        rows = self.db.pending_todos(user_id)
        if not rows:
            return "你现在没有待办事项。"
        target = self._resolve_target(rows, args, ctx)
        if target is None:
            return ("没找到你说的那条待办。你当前的待办是：\n" + self._format_rows(rows, ctx))
        if action == "complete":
            self.db.complete_todo(target["id"], user_id)
            return "已完成：%s，继续加油！" % target["content"]
        if action == "delete":
            self.db.delete_todo(target["id"], user_id)
            return "已删除：%s" % target["content"]
        return self._query(user_id, ctx)

    def _query(self, user_id, ctx):
        rows = self.db.pending_todos(user_id)
        if not rows:
            return "太棒了，你现在没有任何待办事项。"
        return "你当前的待办（共 %d 项）：\n%s" % (len(rows), self._format_rows(rows, ctx))

    def _format_rows(self, rows, ctx):
        ctx["last_todo_ids"] = [r["id"] for r in rows]
        lines = []
        for i, r in enumerate(rows, 1):
            if r["remind_at"]:
                lines.append("%d. [%s] %s" % (i, _fmt_remind(r["remind_at"]), r["content"]))
            else:
                lines.append("%d. [未设提醒] %s" % (i, r["content"]))
        return "\n".join(lines)

    def _resolve_target(self, rows, args, ctx):
        """按序号（最近一次查询列表）或关键词定位一条待办。"""
        index = args.get("index")
        if index:
            try:
                idx = int(index) - 1
            except (TypeError, ValueError):
                idx = -1
            last_ids = ctx.get("last_todo_ids") or []
            if 0 <= idx < len(last_ids):
                for r in rows:
                    if r["id"] == last_ids[idx]:
                        return r
            if 0 <= idx < len(rows):  # 没查过列表时按当前排序兜底
                return rows[idx]
        keyword = (args.get("keyword") or "").strip()
        if keyword:
            for r in rows:
                if keyword in r["content"]:
                    return r
        return None

    # ---------- 定时任务用 ----------
    def reminder_text(self, row):
        return "【待办提醒】%s\n设定时间：%s\n\n回复「完成了」即可标记完成。" % (
            row["content"],
            _fmt_remind(row["remind_at"]),
        )

    def morning_brief_text(self, user_id):
        rows = self.db.pending_todos(user_id)
        if not rows:
            return None
        today_hms = []
        others = []
        for r in rows:
            if r["remind_at"] and _fmt_remind(r["remind_at"]).startswith("今天"):
                today_hms.append(r)
            else:
                others.append(r)
        lines = ["早上好！这是你今天的待办清单："]
        if today_hms:
            lines.append("\n今日到期：")
            for r in today_hms:
                lines.append("· [%s] %s" % (_fmt_remind(r["remind_at"]), r["content"]))
        if others:
            lines.append("\n其他待办：")
            for r in others:
                if r["remind_at"]:
                    lines.append("· [%s] %s" % (_fmt_remind(r["remind_at"]), r["content"]))
                else:
                    lines.append("· %s" % r["content"])
        return "\n".join(lines)
