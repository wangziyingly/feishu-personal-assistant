"""定时任务：待办到点提醒、每日待办早报、每日论文订阅推送。

所有 job 内部捕获异常，单个任务失败不影响调度器本身。
"""
import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .modules.todo import TodoModule
from .modules.paper import PaperModule

logger = logging.getLogger(__name__)


def _parse_hhmm(s, default=(8, 0)):
    try:
        h, m = s.strip().split(":")
        return int(h), int(m)
    except (ValueError, AttributeError):
        return default


class AssistantScheduler:
    def __init__(self, cfg, db, bot, llm):
        self.cfg = cfg
        self.db = db
        self.bot = bot
        self.todo = TodoModule(cfg, db, llm)
        self.paper = PaperModule(cfg, db, llm)
        self._sched = BackgroundScheduler()

        # 每分钟检查到点待办
        self._sched.add_job(self._job_due_todos, "interval", minutes=1, id="due_todos")
        # 每日早报
        h, m = _parse_hhmm(cfg.morning_brief)
        self._sched.add_job(self._job_morning_brief, CronTrigger(hour=h, minute=m), id="morning_brief")
        # 每日论文订阅推送：每 15 分钟扫一次，按每个订阅自己的时间推送（防当日重复）
        self._sched.add_job(self._job_paper_digest, "interval", minutes=15, id="paper_digest")

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
        """到点且今日未推。订阅时间窗 15 分钟（配合轮询间隔）。"""
        if sub["last_pushed"] == now.strftime("%Y-%m-%d"):
            return False
        hhmm = sub["digest_time"] or self.cfg.paper_digest
        h, m = _parse_hhmm(hhmm, default=(8, 30))
        target = h * 60 + m
        now_min = now.hour * 60 + now.minute
        return target <= now_min < target + 15

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
                    text = self.paper.daily_digest(sub["user_id"], sub["keywords"], top_n)
                    if text:
                        self.bot.push(chat_id, text)
                    self.db.mark_pushed(sub["user_id"], now.strftime("%Y-%m-%d"))
                except Exception as e:
                    logger.error("推送论文日报失败(user=%s): %s", sub["user_id"], e)
        except Exception as e:
            logger.error("论文日报任务失败: %s", e)
