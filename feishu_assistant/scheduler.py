"""定时任务：待办到点提醒、每日待办早报、每日论文订阅推送。

所有 job 内部捕获异常，单个任务失败不影响调度器本身。
"""
import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .modules.todo import TodoModule
from .modules.paper import PaperModule
from .modules.ghwatch import GhWatchModule

logger = logging.getLogger(__name__)


def _parse_hhmm(s, default=(8, 0)):
    try:
        h, m = s.strip().split(":")
        return int(h), int(m)
    except (ValueError, AttributeError):
        return default


class AssistantScheduler:
    def __init__(self, cfg, db, bot, llm, router=None):
        self.cfg = cfg
        self.db = db
        self.bot = bot
        self.router = router  # 可选：日报推送后登记指代点/历史，供用户追问
        self.todo = TodoModule(cfg, db, llm)
        self.paper = PaperModule(cfg, db, llm)
        self.ghwatch = GhWatchModule(cfg, db, llm)
        self._sched = BackgroundScheduler()

        # 每分钟检查到点待办
        self._sched.add_job(self._job_due_todos, "interval", minutes=1, id="due_todos")
        # 每日早报（misfire 宽限 8 小时：机器睡眠醒来后补发）
        h, m = _parse_hhmm(cfg.morning_brief)
        self._sched.add_job(self._job_morning_brief, CronTrigger(hour=h, minute=m),
                            id="morning_brief", misfire_grace_time=8 * 3600, coalesce=True)
        # 每日论文订阅推送：每 15 分钟扫一次，按每个订阅自己的时间推送（防当日重复）
        self._sched.add_job(self._job_paper_digest, "interval", minutes=15, id="paper_digest")
        # GitHub release 雷达：按配置的间隔轮询（默认每小时）
        self._sched.add_job(self._job_github_watch, "interval",
                            minutes=cfg.github_poll_minutes, id="github_watch")

    def start(self):
        self._sched.start()
        logger.info("定时任务已启动")

    # ---------- 待办提醒 ----------
    def _job_due_todos(self):
        try:
            now_hm = datetime.now().strftime("%Y-%m-%d %H:%M")
            for row in self.db.due_todos(now_hm):
                chat_id = self.db.get_chat_id(row["user_id"])
                if not chat_id:
                    continue
                try:
                    self.bot.push(chat_id, self.todo.reminder_text(row))
                    self.db.mark_reminded(row["id"])
                except Exception as e:
                    logger.error("推送待办提醒失败(id=%s): %s", row["id"], e)
        except Exception as e:
            logger.error("检查到点待办失败: %s", e)

    # ---------- 每日早报 ----------
    def _job_morning_brief(self):
        try:
            for u in self.db.all_users():
                try:
                    text = self.todo.morning_brief_text(u["user_id"])
                    if text:
                        self.bot.push(u["chat_id"], text)
                except Exception as e:
                    logger.error("推送早报失败(user=%s): %s", u["user_id"], e)
        except Exception as e:
            logger.error("早报任务失败: %s", e)

    # ---------- 论文订阅推送 ----------
    def _should_push(self, sub, now):
        """当天未推且已过订阅时间即推——机器睡眠错过窗口后，醒来能补推。"""
        if sub["last_pushed"] == now.strftime("%Y-%m-%d"):
            return False
        hhmm = sub["digest_time"] or self.cfg.paper_digest
        h, m = _parse_hhmm(hhmm, default=(8, 30))
        return now.hour * 60 + now.minute >= h * 60 + m

    def _job_paper_digest(self):
        try:
            now = datetime.now()
            for sub in self.db.active_subscriptions():
                if not self._should_push(sub, now):
                    continue
                chat_id = self.db.get_chat_id(sub["user_id"])
                if not chat_id:
                    continue
                try:
                    top_n = sub["top_n"] or self.cfg.paper_digest_top_n
                    text, papers, searched_ok = self.paper.daily_digest(
                        sub["user_id"], sub["keywords"], top_n)
                    if text:
                        self.bot.push(chat_id, text)
                        if self.router:
                            self.router.note_digest_push(sub["user_id"], text, papers)
                    # 检索成功（含干净的空结果）才标记；检索失败（限流等）不标记，下轮重试
                    if searched_ok:
                        self.db.mark_pushed(sub["user_id"], now.strftime("%Y-%m-%d"))
                    else:
                        logger.warning("论文日报检索失败（可能限流），下轮重试(user=%s)", sub["user_id"])
                except Exception as e:
                    logger.error("推送论文日报失败(user=%s): %s", sub["user_id"], e)
        except Exception as e:
            logger.error("论文日报任务失败: %s", e)

    # ---------- GitHub release 雷达 ----------
    def _job_github_watch(self):
        try:
            # 按用户归组：同一轮多个 repo 出新版本时，一次 LLM 调用批量解读（漏斗第 1 层）
            by_user = {}
            for w in self.db.all_github_watches():
                by_user.setdefault(w["user_id"], []).append(w)
            for user_id, watches in by_user.items():
                chat_id = self.db.get_chat_id(user_id)
                if not chat_id:
                    continue
                pending = []  # [(watch, release)]
                for w in watches:
                    new = self.ghwatch.check_new_releases(w)
                    if new is None:  # 拉取失败（限流/断网）：游标不动，下轮重试
                        continue
                    if not w["last_release_id"] and not new:
                        # 游标未初始化（订阅时拉取失败）：本轮补上，避免误推历史版本
                        self.db.mark_github_release(
                            w["id"], self.ghwatch.latest_release_id(w["repo"]))
                    pending.extend((w, r) for r in new)
                if not pending:
                    continue
                text = self.ghwatch.digest_text(user_id, [(w["repo"], r) for w, r in pending])
                if not text:  # LLM 解读失败：游标不动，下轮重试
                    continue
                try:
                    self.bot.push(chat_id, text)
                    for w, r in pending:
                        self.db.mark_github_release(w["id"], r["id"])
                except Exception as e:
                    logger.error("推送 GitHub release 失败(user=%s): %s", user_id, e)
        except Exception as e:
            logger.error("GitHub 订阅任务失败: %s", e)
