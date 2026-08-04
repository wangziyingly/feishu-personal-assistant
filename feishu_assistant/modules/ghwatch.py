"""GitHub 仓库订阅模块：盯 repo 的新 release，LLM 解读后推送。

轮询 GitHub REST API（无需公网服务器，与本地长连接架构契合）。
未认证 60 次/小时限额，盯十几个 repo 每小时一轮足够；config.yaml 配
github.token 可到 5000 次/小时。

成本沿用漏斗设计：
- 第 0 层（零成本）：release id 与库中游标比对，无更新直接跳过；draft/prerelease 硬过滤；
- 第 1 层（便宜）：同一轮的多个新 release 压缩进一次 LLM 调用批量解读；
- 第 2 层（防重复）：推送成功后 last_release_id 落库，同一版本永不重复解读/推送。
拉取失败时不推进游标，下轮自动重试（at-least-once，宁可重复不漏推）。
"""
import logging
import re

import requests

from ..llm import LLMError
from .profile import profile_prompt_text

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
RELEASES_PER_REPO = 3        # 每轮每 repo 最多回看几个 release
NOTES_MAX_CHARS = 3000       # 单个 release notes 送入 LLM 的截断长度
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def normalize_repo(text):
    """从用户输入提取 owner/name；支持完整 GitHub URL。非法输入返回 None。"""
    text = (text or "").strip().rstrip("/")
    m = re.search(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", text)
    if m:
        text = m.group(1)
    for suffix in (".git", "/releases", "/issues", "/pulls"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text if REPO_RE.match(text) else None


class GhWatchModule:
    def __init__(self, cfg, db, llm):
        self.cfg = cfg
        self.db = db
        self.llm = llm

    # ---------- GitHub API ----------
    def _fetch_releases(self, repo):
        """返回 [{id, tag, name, url, body, draft, prerelease}]，按时间倒序；失败抛异常。"""
        headers = {"Accept": "application/vnd.github+json"}
        token = getattr(self.cfg, "github_token", "")
        if token:
            headers["Authorization"] = "Bearer %s" % token
        resp = requests.get(
            "%s/repos/%s/releases" % (GITHUB_API, repo),
            params={"per_page": RELEASES_PER_REPO},
            headers=headers, timeout=20,
        )
        resp.raise_for_status()
        return [
            {"id": str(r.get("id")), "tag": r.get("tag_name") or "",
             "name": r.get("name") or r.get("tag_name") or "",
             "url": r.get("html_url") or "",
             "body": (r.get("body") or "").strip(),
             "draft": bool(r.get("draft")), "prerelease": bool(r.get("prerelease"))}
            for r in resp.json()
        ]

    def check_new_releases(self, watch):
        """返回该订阅的新 release 列表（正序，旧→新）；拉取失败返回 None（下轮重试）。"""
        try:
            releases = self._fetch_releases(watch["repo"])
        except Exception as e:
            logger.warning("拉取 %s releases 失败: %s", watch["repo"], e)
            return None
        # 第 0 层硬过滤：draft / prerelease 不推
        releases = [r for r in releases if not r["draft"] and not r["prerelease"]]
        last = watch["last_release_id"] or ""
        if not last:
            # 首次订阅：只盯之后的新版本，不回溯推送历史 release
            return []
        new = []
        for r in releases:  # 倒序遍历，撞到游标即停
            if r["id"] == last:
                break
            new.append(r)
        return new[::-1]

    def latest_release_id(self, repo):
        """订阅时初始化游标为当前最新 release（失败返回空串，下轮再初始化）。"""
        try:
            releases = self._fetch_releases(repo)
        except Exception:
            return ""
        releases = [r for r in releases if not r["draft"] and not r["prerelease"]]
        return releases[0]["id"] if releases else ""

    # ---------- LLM 批量解读 ----------
    def digest_text(self, user_id, items):
        """items: [(repo, release)]；一次 LLM 调用批量解读，返回推送文本。失败返回 None。"""
        bg = profile_prompt_text(self.db, user_id)
        briefs = "\n\n".join(
            "%d. 仓库 %s 发布 %s（%s）\nRelease notes：\n%s"
            % (i, repo, rel["name"] or rel["tag"], rel["tag"],
               (rel["body"] or "（无 release notes）")[:NOTES_MAX_CHARS])
            for i, (repo, rel) in enumerate(items, 1)
        )
        try:
            text = self.llm.chat([
                {"role": "system", "content":
                    "你是用户的技术雷达助手。用中文解读 GitHub 新版本，面向准备 Agent 开发岗求职的开发者。"
                    "每个仓库按以下结构输出：\n"
                    "📦 仓库名 版本号\n【更新了什么】3 条以内要点，说人话\n"
                    "【为什么值得关注】结合用户背景指出与 Agent 开发/求职/其个人项目的关联，"
                    "没有关联就如实说「例行维护，可不关注」\n"
                    "不要编造 notes 里没有的内容。"},
                {"role": "user", "content":
                    "%s以下是用户订阅的仓库新发布的版本：\n\n%s" % (
                        bg + "\n" if bg else "", briefs)},
            ])
        except LLMError as e:
            logger.error("GitHub release 解读失败: %s", e)
            return None
        links = "\n".join("· %s %s：%s" % (repo, rel["tag"], rel["url"])
                          for repo, rel in items)
        return "%s\n\n链接：\n%s" % (text, links)

    # ---------- 订阅管理（router 调用） ----------
    def handle_watch(self, user_id, args):
        action = (args.get("action") or "list").strip()
        if action == "add":
            repo = normalize_repo(args.get("repo"))
            if not repo:
                return ("没认出仓库名。给我 owner/repo 格式（如 langchain-ai/langgraph）"
                        "或直接发 GitHub 链接。")
            if not self.db.add_github_watch(user_id, repo):
                return "你已经订阅过 %s 了。回复「我订阅了哪些repo」可查看全部。" % repo
            # 游标初始化为当前最新 release：只推之后的新版本，不回溯刷屏
            self.db.mark_github_release(
                self.db.list_github_watch(user_id)[-1]["id"], self.latest_release_id(repo))
            return ("已订阅 %s 🎯 之后它每次发新版本，我会解读更新内容推送给你"
                    "（每小时检查一次，只推正式版）。" % repo)
        if action == "remove":
            repo = normalize_repo(args.get("repo"))
            if repo and self.db.remove_github_watch(user_id, repo):
                return "已取消订阅 %s。" % repo
            return "没找到这个订阅。回复「我订阅了哪些repo」确认一下名字。"
        rows = self.db.list_github_watch(user_id)
        if not rows:
            return ("你还没有订阅任何 GitHub 仓库。回复「订阅 langchain-ai/langgraph 的更新」"
                    "或直接发仓库链接即可订阅。")
        lines = ["你订阅的 GitHub 仓库（有新 release 会推送解读）："]
        for i, r in enumerate(rows, 1):
            lines.append("%d. %s（订阅于 %s）" % (i, r["repo"], r["created_at"][:10]))
        lines.append("\n回复「取消订阅 owner/repo」移除；发仓库链接可添加。")
        return "\n".join(lines)
