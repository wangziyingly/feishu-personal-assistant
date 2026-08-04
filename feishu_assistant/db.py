"""SQLite 本地存储层。

存六类数据：用户(用于主动推送)、待办、文献库、论文订阅、面试记录、面试错题本。
时间一律存本地时间字符串 "YYYY-MM-DD HH:MM:SS"（待办提醒用 "YYYY-MM-DD HH:MM"）。
"""
import json
import sqlite3
import threading
from datetime import datetime

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id    TEXT PRIMARY KEY,
    chat_id    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS todos (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    content    TEXT NOT NULL,
    remind_at  TEXT,               -- "YYYY-MM-DD HH:MM"，NULL 表示不设提醒
    status     TEXT NOT NULL DEFAULT 'pending',  -- pending/done/deleted
    reminded   INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS papers (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    title      TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT '',   -- arxiv / pdf_upload / semantic_scholar
    url        TEXT NOT NULL DEFAULT '',
    summary    TEXT NOT NULL DEFAULT '',
    tags       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS subscriptions (
    user_id      TEXT PRIMARY KEY,
    keywords     TEXT NOT NULL DEFAULT '',
    enabled      INTEGER NOT NULL DEFAULT 1,
    digest_time  TEXT NOT NULL DEFAULT '',   -- 推送时间 "HH:MM"，空=用全局配置
    top_n        INTEGER NOT NULL DEFAULT 0, -- 推送篇数，0=用全局配置
    last_pushed  TEXT NOT NULL DEFAULT '',   -- 最近推送日期 "YYYY-MM-DD"，防重复
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS interview_sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    type       TEXT NOT NULL,              -- mock / jd / resume / review
    status     TEXT NOT NULL DEFAULT 'done',
    transcript TEXT NOT NULL DEFAULT '',   -- JSON 文本
    feedback   TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mistakes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL DEFAULT '',
    category   TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS question_bank (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL,
    question      TEXT NOT NULL,
    analysis      TEXT NOT NULL DEFAULT '',  -- 考察点 + 答题思路
    category      TEXT NOT NULL DEFAULT '',  -- 自由分类：AI产品经理/算法/Agent/系统设计…
    source        TEXT NOT NULL DEFAULT '搜集',  -- 搜集 / 面经
    asked_count   INTEGER NOT NULL DEFAULT 0,  -- 被考次数
    good_count    INTEGER NOT NULL DEFAULT 0,  -- 答得好的次数
    last_result   TEXT NOT NULL DEFAULT '',    -- 最近一次结果：good / bad
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS profiles (
    user_id            TEXT PRIMARY KEY,
    research_direction TEXT NOT NULL DEFAULT '',  -- 研究方向
    target_companies   TEXT NOT NULL DEFAULT '',  -- 目标公司/岗位
    resume_highlights  TEXT NOT NULL DEFAULT '',  -- 简历要点（发简历时自动提炼）
    extra              TEXT NOT NULL DEFAULT '',  -- 其他备注
    updated_at         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv_store (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conv_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id    INTEGER NOT NULL,
    role       TEXT NOT NULL,          -- user / assistant
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS github_watch (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,
    repo            TEXT NOT NULL,               -- owner/name
    last_release_id TEXT NOT NULL DEFAULT '',    -- 最近已推送的 release id，去重兼重试游标
    created_at      TEXT NOT NULL,
    UNIQUE(user_id, repo)
);
CREATE TABLE IF NOT EXISTS inbox_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL UNIQUE,   -- 飞书消息 id，兼作去重键
    payload    TEXT NOT NULL,          -- JSON：user_id/chat_id/msg_type/content
    status     TEXT NOT NULL DEFAULT 'pending',  -- pending / done / failed
    attempts   INTEGER NOT NULL DEFAULT 0,
    received_at TEXT NOT NULL
);
"""


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Database:
    def __init__(self, path):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            # 老库迁移：question_bank 补 source 列；subscriptions 补推送配置列
            cols = [r["name"] for r in self._conn.execute("PRAGMA table_info(question_bank)")]
            if cols and "source" not in cols:
                self._conn.execute(
                    "ALTER TABLE question_bank ADD COLUMN source TEXT NOT NULL DEFAULT '搜集'")
            sub_cols = [r["name"] for r in self._conn.execute("PRAGMA table_info(subscriptions)")]
            if sub_cols and "digest_time" not in sub_cols:
                self._conn.execute("ALTER TABLE subscriptions ADD COLUMN digest_time TEXT NOT NULL DEFAULT ''")
                self._conn.execute("ALTER TABLE subscriptions ADD COLUMN top_n INTEGER NOT NULL DEFAULT 0")
                self._conn.execute("ALTER TABLE subscriptions ADD COLUMN last_pushed TEXT NOT NULL DEFAULT ''")
            self._conn.commit()

    def _execute(self, sql, params=()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql, params=()):
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ---------- 用户 ----------
    def upsert_user(self, user_id, chat_id):
        self._execute(
            "INSERT INTO users(user_id, chat_id, created_at) VALUES(?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET chat_id=excluded.chat_id",
            (user_id, chat_id, now_str()),
        )

    def get_chat_id(self, user_id):
        rows = self._query("SELECT chat_id FROM users WHERE user_id=?", (user_id,))
        return rows[0]["chat_id"] if rows else None

    def all_users(self):
        return self._query("SELECT user_id, chat_id FROM users")

    # ---------- 待办 ----------
    def add_todo(self, user_id, content, remind_at):
        cur = self._execute(
            "INSERT INTO todos(user_id, content, remind_at, created_at) VALUES(?,?,?,?)",
            (user_id, content, remind_at, now_str()),
        )
        return cur.lastrowid

    def pending_todos(self, user_id):
        return self._query(
            "SELECT * FROM todos WHERE user_id=? AND status='pending' "
            "ORDER BY CASE WHEN remind_at IS NULL THEN 1 ELSE 0 END, remind_at",
            (user_id,),
        )

    def due_todos(self, now_hm):
        """到点且未提醒的待办。now_hm 形如 'YYYY-MM-DD HH:MM'。"""
        return self._query(
            "SELECT * FROM todos WHERE status='pending' AND reminded=0 "
            "AND remind_at IS NOT NULL AND remind_at <= ?",
            (now_hm,),
        )

    def mark_reminded(self, todo_id):
        self._execute("UPDATE todos SET reminded=1 WHERE id=?", (todo_id,))

    def complete_todo(self, todo_id, user_id):
        self._execute(
            "UPDATE todos SET status='done' WHERE id=? AND user_id=? AND status='pending'",
            (todo_id, user_id),
        )

    def delete_todo(self, todo_id, user_id):
        self._execute(
            "UPDATE todos SET status='deleted' WHERE id=? AND user_id=? AND status='pending'",
            (todo_id, user_id),
        )

    # ---------- 文献库 ----------
    def add_paper(self, user_id, title, source, url, summary, tags):
        cur = self._execute(
            "INSERT INTO papers(user_id, title, source, url, summary, tags, created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (user_id, title, source, url, summary, tags, now_str()),
        )
        return cur.lastrowid

    def list_papers(self, user_id, limit=20):
        return self._query(
            "SELECT * FROM papers WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )

    def search_papers(self, user_id, keyword):
        like = "%" + keyword + "%"
        return self._query(
            "SELECT * FROM papers WHERE user_id=? AND (title LIKE ? OR tags LIKE ? OR summary LIKE ?) "
            "ORDER BY id DESC LIMIT 20",
            (user_id, like, like, like),
        )

    def get_paper(self, paper_id, user_id):
        rows = self._query(
            "SELECT * FROM papers WHERE id=? AND user_id=?", (paper_id, user_id)
        )
        return rows[0] if rows else None

    # ---------- 论文订阅 ----------
    def set_subscription(self, user_id, keywords, enabled=True, digest_time=None, top_n=None):
        row = self.get_subscription(user_id)
        new_time = digest_time if digest_time is not None else (row["digest_time"] if row else "")
        new_topn = top_n if top_n is not None else (row["top_n"] if row else 0)
        self._execute(
            "INSERT INTO subscriptions(user_id, keywords, enabled, digest_time, top_n, created_at) "
            "VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET keywords=excluded.keywords, enabled=excluded.enabled, "
            "digest_time=excluded.digest_time, top_n=excluded.top_n",
            (user_id, keywords, 1 if enabled else 0, new_time, new_topn, now_str()),
        )

    def mark_pushed(self, user_id, date_str):
        self._execute("UPDATE subscriptions SET last_pushed=? WHERE user_id=?", (date_str, user_id))

    def get_subscription(self, user_id):
        rows = self._query("SELECT * FROM subscriptions WHERE user_id=?", (user_id,))
        return rows[0] if rows else None

    def active_subscriptions(self):
        return self._query("SELECT * FROM subscriptions WHERE enabled=1")

    # ---------- GitHub 仓库订阅（release 雷达） ----------
    def add_github_watch(self, user_id, repo):
        """订阅 repo；已存在返回 False。"""
        try:
            self._execute(
                "INSERT INTO github_watch(user_id, repo, created_at) VALUES(?,?,?)",
                (user_id, repo, now_str()),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    def remove_github_watch(self, user_id, repo):
        cur = self._execute(
            "DELETE FROM github_watch WHERE user_id=? AND repo=?", (user_id, repo))
        return cur.rowcount > 0

    def list_github_watch(self, user_id):
        return self._query(
            "SELECT * FROM github_watch WHERE user_id=? ORDER BY id", (user_id,))

    def all_github_watches(self):
        return self._query("SELECT * FROM github_watch ORDER BY id")

    def mark_github_release(self, watch_id, release_id):
        self._execute(
            "UPDATE github_watch SET last_release_id=? WHERE id=?",
            (str(release_id), watch_id),
        )

    # ---------- 面试记录 ----------
    def add_interview_session(self, user_id, type_, transcript, feedback=""):
        cur = self._execute(
            "INSERT INTO interview_sessions(user_id, type, transcript, feedback, created_at) "
            "VALUES(?,?,?,?,?)",
            (user_id, type_, json.dumps(transcript, ensure_ascii=False), feedback, now_str()),
        )
        return cur.lastrowid

    def list_interview_sessions(self, user_id, limit=10):
        return self._query(
            "SELECT * FROM interview_sessions WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )

    # ---------- 错题本 ----------
    def add_mistake(self, user_id, question, answer, category):
        cur = self._execute(
            "INSERT INTO mistakes(user_id, question, answer, category, created_at) VALUES(?,?,?,?,?)",
            (user_id, question, answer, category, now_str()),
        )
        return cur.lastrowid

    def list_mistakes(self, user_id, category=None, limit=50):
        if category:
            return self._query(
                "SELECT * FROM mistakes WHERE user_id=? AND category LIKE ? ORDER BY id DESC LIMIT ?",
                (user_id, "%" + category + "%", limit),
            )
        return self._query(
            "SELECT * FROM mistakes WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )

    # ---------- 题库（搜集的面试题/面经） ----------
    def add_question(self, user_id, question, analysis, category, source="搜集"):
        cur = self._execute(
            "INSERT INTO question_bank(user_id, question, analysis, category, source, created_at) "
            "VALUES(?,?,?,?,?,?)",
            (user_id, question, analysis, category, source, now_str()),
        )
        return cur.lastrowid

    def list_questions(self, user_id, category=None, limit=50):
        if category:
            return self._query(
                "SELECT * FROM question_bank WHERE user_id=? AND category LIKE ? ORDER BY id DESC LIMIT ?",
                (user_id, "%" + category + "%", limit),
            )
        return self._query(
            "SELECT * FROM question_bank WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )

    def pick_quiz_question(self, user_id, category=None, exclude_id=None):
        """抽题：优先没被考过的，其次最近一次答砸的，同等条件下随机。"""
        sql = ("SELECT * FROM question_bank WHERE user_id=? AND id != ? "
               + ("AND category LIKE ? " if category else "")
               + "ORDER BY asked_count, CASE last_result WHEN 'bad' THEN 0 ELSE 1 END, RANDOM() LIMIT 1")
        params = (user_id, exclude_id or -1) + (("%" + category + "%",) if category else ())
        rows = self._query(sql, params)
        return rows[0] if rows else None

    def record_quiz_result(self, qid, good):
        self._execute(
            "UPDATE question_bank SET asked_count=asked_count+1, "
            "good_count=good_count+?, last_result=? WHERE id=?",
            (1 if good else 0, "good" if good else "bad", qid),
        )

    # ---------- 个人画像 ----------
    PROFILE_FIELDS = ("research_direction", "target_companies", "resume_highlights", "extra")

    def get_profile(self, user_id):
        rows = self._query("SELECT * FROM profiles WHERE user_id=?", (user_id,))
        return rows[0] if rows else None

    def upsert_profile(self, user_id, **fields):
        """只更新传入的非空字段；没有任何画像字段时忽略。"""
        sets = {k: v for k, v in fields.items() if k in self.PROFILE_FIELDS and v}
        if not sets:
            return
        cols = ", ".join(sets)
        updates = ", ".join("%s=excluded.%s" % (k, k) for k in sets)
        self._execute(
            "INSERT INTO profiles(user_id, %s, updated_at) VALUES(?, %s, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET %s, updated_at=excluded.updated_at"
            % (cols, ", ".join("?" for _ in sets), updates),
            (user_id, *sets.values(), now_str()),
        )

    # ---------- KV 小存储（知识库空间/文档映射等） ----------
    def kv_get(self, key):
        rows = self._query("SELECT value FROM kv_store WHERE key=?", (key,))
        return rows[0]["value"] if rows else ""

    def kv_set(self, key, value):
        self._execute(
            "INSERT INTO kv_store(key, value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    # ---------- 入站事件队列（持久化：重启/崩溃后重放） ----------
    def inbox_enqueue(self, message_id, payload):
        """落库即去重：重复 message_id 返回 False（飞书重投只处理一次）。"""
        try:
            self._execute(
                "INSERT INTO inbox_events(message_id, payload, received_at) VALUES(?,?,?)",
                (message_id, json.dumps(payload, ensure_ascii=False), now_str()),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    def inbox_pending(self, max_attempts=3, max_age_hours=24):
        return self._query(
            "SELECT * FROM inbox_events WHERE status != 'done' AND attempts < ? "
            "AND received_at >= datetime('now', 'localtime', ?)",
            (max_attempts, "-%d hours" % max_age_hours),
        )

    def inbox_mark(self, event_id, status):
        self._execute(
            "UPDATE inbox_events SET status=?, attempts=attempts+1 WHERE id=?",
            (status, event_id),
        )

    # ---------- 多对话（命名会话 + 历史持久化） ----------
    def create_conversation(self, user_id, name):
        cur = self._execute(
            "INSERT INTO conversations(user_id, name, created_at) VALUES(?,?,?)",
            (user_id, name, now_str()),
        )
        return cur.lastrowid

    def list_conversations(self, user_id):
        return self._query(
            "SELECT c.*, (SELECT COUNT(*) FROM conv_messages m WHERE m.conv_id=c.id) AS msg_count "
            "FROM conversations c WHERE c.user_id=? ORDER BY c.id",
            (user_id,),
        )

    def delete_conversation(self, conv_id, user_id):
        self._execute("DELETE FROM conv_messages WHERE conv_id=?", (conv_id,))
        self._execute("DELETE FROM conversations WHERE id=? AND user_id=?", (conv_id, user_id))

    def add_conv_message(self, conv_id, role, content):
        self._execute(
            "INSERT INTO conv_messages(conv_id, role, content, created_at) VALUES(?,?,?,?)",
            (conv_id, role, content, now_str()),
        )

    def get_conv_messages(self, conv_id, limit):
        """按时间正序返回最近 limit 条。"""
        rows = self._query(
            "SELECT role, content FROM conv_messages WHERE conv_id=? ORDER BY id DESC LIMIT ?",
            (conv_id, limit),
        )
        return rows[::-1]

    def close(self):
        with self._lock:
            self._conn.close()
