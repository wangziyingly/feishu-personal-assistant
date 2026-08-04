"""文献助手模块：需求驱动搜索、PDF 结构化总结、文献汇报（组会精读）、文献库、每日订阅推送。

搜索用 arXiv（主）+ Semantic Scholar（备），都是免费公开 API，无需 key。
"需求驱动"指：LLM 先把用户的研究需求扩展成多个英文检索式，检索合并后再按
需求相关性排序筛选，而不是简单关键词匹配。

成本上按"先便宜后贵"的漏斗设计：
- 第 0 层（零成本）：硬过滤——去掉文献库已有的、与检索式关键词零重叠的候选，
  并截断候选数量，尽量不把垃圾候选送进 LLM；
- 第 1 层（便宜）：一次 LLM 调用对候选批量打分排序（_rerank）；
- 第 2 层（贵）：精读/汇报前查文献库缓存，已有结果直接复用，不重复调用 LLM。
"""
import os
import re
from datetime import datetime, timedelta

import requests

from ..config import PDF_DIR
from ..db import now_str
from ..llm import LLMError
from .profile import profile_prompt_text

ABSTRACT_PREVIEW = 200      # 搜索结果里摘要预览长度
PDF_MAX_PAGES = 80          # PDF 最多解析页数
PDF_MAX_CHARS = 120000      # 送入大模型的正文字符上限（长上下文模型可读全文，精读质量依赖全文）
LIBRARY_CTX_CHARS = 400     # 文献库问答时每篇带入的摘要长度
ARXIV_PDF_MAX_BYTES = 40 * 1024 * 1024  # arXiv 全文下载大小上限
RERANK_CANDIDATES = 15      # 硬过滤后送进 LLM 粗筛的候选上限
REPORT_MARK = "【汇报问答准备】"  # 判断文献库存储内容是否为文献汇报的标记

# 检索式里的通用词，不参与关键词粗匹配
QUERY_STOPWORDS = set(
    ("the a an and or of in on for with to from by at as is are was were be been "
     "via using use based new recent latest study research paper papers review "
     "approach method methods technique techniques model models system systems").split())


def _norm_title(title):
    """标题归一化（小写、去非字母数字），用于和文献库比对。"""
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


def _query_terms(queries):
    """从英文检索式提取关键词（≥4 字符、非停用词），用于零成本粗匹配。"""
    terms = set()
    for q in queries:
        for w in re.findall(r"[a-z0-9][a-z0-9\-]*", (q or "").lower()):
            w = w.strip("-")
            if len(w) >= 4 and w not in QUERY_STOPWORDS:
                terms.add(w)
    return terms

REPORT_PROMPT = """你是一位擅长把论文讲明白的研究者，正在为一位聪明的初学者准备文献汇报。
读者不了解这个领域的术语，也不关心作者是谁。目标是让他读完真正懂这篇论文，而不是交一份学术八股。

基于给出的论文内容（可能有截断），输出一份中文文献汇报，严格按以下结构（保留小节标题）：

【一句话总结】用大白话说清这篇论文做了什么
【痛点与动机】它解决什么问题？之前的方法卡在哪？用具体场景说明
【核心思路】直觉上它是怎么做的？必须配一个类比或具体例子（"就像……"），不许直接堆术语
【方法详解】按"输入是什么 → 经过哪几步 → 输出是什么"的流水线讲清实现；
每个术语第一次出现时用一句话解释；关键设计说清为什么这样做
【实验与结果】数据集、基线、关键数字，这些数字说明什么
【局限与坑】至少 2 条：什么情况下会失效、哪些结论不能全信
【能学到什么】对做 Agent/AI 开发的读者有 1-2 条具体可借鉴的点
【汇报问答准备】组会上可能被问的 2-3 个问题及应对要点

铁律：
- 禁止罗列作者、禁止"xxx et al."式引用堆砌，只讲内容本身
- 禁止空洞形容词（"新颖的""强大的""有效的"），用具体事实和数字说话
- 只基于论文内容，不要编造数字；论文没提到的部分写「论文未提及」。"""


# ---------- 搜索 ----------
def search_arxiv(query, max_results=5):
    """返回 [{title, authors, summary, url, published, source}]，按提交时间倒序。"""
    import arxiv  # 延迟导入，未安装时只影响本功能

    client = arxiv.Client(page_size=10, delay_seconds=3, num_retries=2)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    results = []
    for r in client.results(search):
        results.append({
            "title": re.sub(r"\s+", " ", r.title).strip(),
            "authors": ", ".join(a.name for a in r.authors[:5]),
            "summary": re.sub(r"\s+", " ", r.summary or "").strip(),
            "url": r.entry_id,
            "published": r.published.strftime("%Y-%m-%d") if r.published else "",
            "source": "arxiv",
        })
    return results


def search_semantic_scholar(query, max_results=5):
    resp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": query, "limit": max_results,
                "fields": "title,authors,abstract,year,url"},
        timeout=20,
    )
    resp.raise_for_status()
    results = []
    for p in (resp.json().get("data") or []):
        results.append({
            "title": p.get("title") or "",
            "authors": ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:5]),
            "summary": p.get("abstract") or "",
            "url": p.get("url") or "",
            "published": str(p.get("year") or ""),
            "source": "semantic_scholar",
        })
    return [r for r in results if r["title"]]


def search_papers(query, max_results=5):
    """arXiv 优先，失败或无结果时回退 Semantic Scholar。"""
    try:
        results = search_arxiv(query, max_results)
        if results:
            return results
    except Exception:
        pass
    try:
        return search_semantic_scholar(query, max_results)
    except Exception:
        return []


# ---------- PDF 解析与下载 ----------
def extract_pdf_text(path, max_pages=PDF_MAX_PAGES, max_chars=PDF_MAX_CHARS):
    """提取 PDF 文本；返回 (文本, 是否被截断)。"""
    from pypdf import PdfReader

    reader = PdfReader(path)
    parts = []
    total = 0
    truncated = False
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            truncated = True
            break
        try:
            text = page.extract_text() or ""
        except Exception:
            continue
        parts.append(text)
        total += len(text)
        if total > max_chars:
            truncated = True
            break
    return "\n".join(parts)[:max_chars], truncated


def download_arxiv_pdf(entry_id, save_dir=PDF_DIR):
    """根据 arXiv abs 链接下载全文 PDF，返回本地路径；失败返回 None。"""
    m = re.search(r"abs/([^/?]+)", entry_id or "")
    if not m:
        return None
    arxiv_id = m.group(1)
    pdf_url = "https://arxiv.org/pdf/" + arxiv_id
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "arxiv_" + arxiv_id.replace("/", "_") + ".pdf")
    try:
        with requests.get(pdf_url, timeout=60, stream=True) as r:
            r.raise_for_status()
            size = 0
            with open(path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    size += len(chunk)
                    if size > ARXIV_PDF_MAX_BYTES:
                        f.close()
                        os.remove(path)
                        return None
                    f.write(chunk)
        return path
    except Exception:
        if os.path.exists(path):
            os.remove(path)
        return None


class PaperModule:
    def __init__(self, cfg, db, llm, bitable=None, bot=None):
        self.cfg = cfg
        self.db = db
        self.llm = llm
        self.bitable = bitable
        self.bot = bot  # 传入时，长生成（文献汇报）走流式回复

    def _save_paper(self, user_id, title, source, url, summary, tags):
        """入库：本地 SQLite（主存储）+ 多维表格「文献库记录」（索引）。"""
        pid = self.db.add_paper(user_id, title, source, url, summary, tags)
        if self.bitable:
            self.bitable.sync_paper(title, tags, source, url, now_str())
        return pid

    # ---------- 需求驱动：检索式扩展 + 相关性排序 ----------
    def _expand_queries(self, need, user_id=None):
        """把用户的研究需求扩展成 2-3 个英文检索式；失败时回退为原需求。"""
        bg = profile_prompt_text(self.db, user_id) if user_id else ""
        try:
            data = self.llm.chat_json([
                {"role": "user", "content":
                    "%s用户想找文献，需求是：%s\n"
                    "请生成 2-3 个适合 arXiv 检索的英文检索式（简短关键词组合，覆盖需求的不同侧面），"
                    "输出 JSON：{\"queries\": [\"q1\", \"q2\"]}" % (bg + "\n" if bg else "", need[:1000])},
            ])
            qs = [q.strip() for q in (data.get("queries") or [])
                  if isinstance(q, str) and q.strip()]
            return qs[:3] or [need]
        except LLMError:
            return [need]

    def _rerank(self, need, candidates, top_n, user_id=None):
        """LLM 按需求相关性排序，返回 [(paper, reason)]；失败时回退为原顺序。"""
        bg = profile_prompt_text(self.db, user_id) if user_id else ""
        briefs = "\n".join(
            "%d. %s | %s" % (i, p["title"], (p["summary"] or "")[:200])
            for i, p in enumerate(candidates, 1)
        )
        try:
            data = self.llm.chat_json([
                {"role": "user", "content":
                    "%s用户的文献需求：%s\n以下是候选论文（序号. 标题 | 摘要）：\n%s\n"
                    "请按与需求（及用户背景）的相关性挑选最相关的 %d 篇并排序，输出 JSON："
                    "{\"picked\": [{\"i\": 序号, \"reason\": \"一句中文推荐理由\"}]}"
                    % (bg + "\n" if bg else "", need[:800], briefs, top_n)},
            ])
            picked = []
            for item in (data.get("picked") or [])[:top_n]:
                try:
                    i = int(item.get("i")) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= i < len(candidates):
                    picked.append((candidates[i], str(item.get("reason") or "")))
            if picked:
                return picked
        except LLMError:
            pass
        return [(p, "") for p in candidates[:top_n]]

    # ---------- 第 0 层：零成本硬过滤 ----------
    def _hard_filter(self, user_id, candidates, queries, need_n):
        """去掉文献库已有的、与检索式关键词零重叠的候选，并截断候选数量。

        全程不调 LLM：入库比对靠 SQLite，相关性粗筛靠检索式关键词匹配。
        关键词匹配后不足 need_n 篇时，用被过滤掉的候选按原顺序补足，避免误伤。
        """
        rows = self.db.list_papers(user_id, limit=500)
        known_urls = {r["url"] for r in rows if r["url"]}
        known_titles = {_norm_title(r["title"]) for r in rows if r["title"]}
        fresh = [p for p in candidates
                 if (not p.get("url") or p["url"] not in known_urls)
                 and _norm_title(p.get("title")) not in known_titles]
        terms = _query_terms(queries)
        matched, rest = [], []
        for p in fresh:
            text = ((p.get("title") or "") + " " + (p.get("summary") or "")).lower()
            (matched if any(t in text for t in terms) else rest).append(p)
        pool = matched if len(matched) >= need_n else matched + rest
        return pool[:RERANK_CANDIDATES]

    def _find_in_library(self, user_id, paper):
        """按 URL 或归一化标题在文献库中查找已有条目，命中返回该行，否则 None。"""
        url = paper.get("url") or ""
        nt = _norm_title(paper.get("title"))
        for r in self.db.list_papers(user_id, limit=500):
            if url and r["url"] == url:
                return r
            if nt and _norm_title(r["title"]) == nt:
                return r
        return None

    # ---------- 关键词/需求搜索 ----------
    def handle_search(self, user_id, args, ctx):
        need = (args.get("query") or "").strip()
        if not need:
            return ("想搜什么方向的论文？可以直接描述你的需求，"
                    "比如：我在做 RAG 检索降噪，帮我找最新的方法。")
        queries = self._expand_queries(need, user_id)
        seen, candidates = set(), []
        for q in queries:
            for p in search_papers(q, self.cfg.paper_search_top_n):
                key = p["url"] or p["title"]
                if key not in seen:
                    seen.add(key)
                    candidates.append(p)
        if not candidates:
            return ("没有搜到与「%s」相关的论文，换个说法试试？"
                    "（arXiv 主要覆盖计算机/AI/数理领域）" % need)
        pool = self._hard_filter(user_id, candidates, queries, self.cfg.paper_search_top_n)
        if not pool:
            return ("搜到的论文都已经在你的文献库里了。换个需求描述试试，"
                    "或回复「看看我的文献库」直接回顾已有内容。")
        picked = self._rerank(need, pool, self.cfg.paper_search_top_n, user_id)
        ctx["last_papers"] = [p for p, _ in picked]
        lines = ["根据你的需求「%s」，为你筛选出最相关的 %d 篇：\n" % (need, len(picked))]
        for i, (p, reason) in enumerate(picked, 1):
            lines.append("%d. 《%s》(%s)" % (i, p["title"], p["published"] or "n.d."))
            if p["authors"]:
                lines.append("   作者：%s" % p["authors"])
            if reason:
                lines.append("   推荐理由：%s" % reason)
            if p["summary"]:
                lines.append("   摘要：%s..." % p["summary"][:ABSTRACT_PREVIEW])
            lines.append("   链接：%s\n" % p["url"])
        lines.append("回复「总结第1篇」看摘要级解读；回复「汇报第1篇」我会下载全文做组会级精读（稍慢）。")
        return "\n".join(lines)

    # ---------- 总结搜索结果中的第 N 篇（摘要级） ----------
    def handle_summarize_index(self, user_id, args, ctx):
        papers = ctx.get("last_papers") or []
        if not papers:
            return "你还没有搜索过论文，先告诉我你的需求，比如：帮我找 RAG 方向最新的论文。"
        try:
            idx = int(args.get("index")) - 1
        except (TypeError, ValueError):
            idx = -1
        if not (0 <= idx < len(papers)):
            return "序号超出范围了，上次搜索共 %d 篇，回复「总结第1篇」~「总结第%d篇」均可。" % (len(papers), len(papers))
        p = papers[idx]
        cached = self._find_in_library(user_id, p)
        if cached:
            return ("《%s》之前已经解读过，直接复用文献库结果（标签：%s）：\n\n%s"
                    % (cached["title"], cached["tags"] or "无", cached["summary"]))
        summary_text = self._summarize_from_abstract(p)
        tags = self._extract_tags(p["title"] + " " + p["summary"])
        self._save_paper(user_id, p["title"], p["source"], p["url"], summary_text, tags)
        return "《%s》解读：\n\n%s\n\n已存入你的文献库（标签：%s）。回复「汇报第%d篇」可获取全文精读版。" % (
            p["title"], summary_text, tags or "无", idx + 1)

    def _summarize_from_abstract(self, paper):
        messages = [
            {"role": "system", "content": "你是论文解读助手，擅长用简洁的中文结构化总结论文。"},
            {"role": "user", "content":
                "请根据以下论文信息输出结构化解读（基于摘要，不要编造摘要中没有的实验数据）：\n"
                "标题：%s\n作者：%s\n摘要：%s\n\n"
                "输出格式：\n【研究问题】...\n【方法】...\n【主要结论】...\n【局限与适用场景】（摘要未提则写「摘要未提及」）"
                % (paper["title"], paper["authors"], paper["summary"][:3000])},
        ]
        return self.llm.chat(messages)

    def _extract_tags(self, text):
        try:
            data = self.llm.chat_json([
                {"role": "user", "content":
                    "给以下内容提取 2-4 个中文技术标签（如 RAG、大模型、强化学习），"
                    "输出 JSON：{\"tags\": [\"标签1\", \"标签2\"]}。内容：%s" % text[:1500]}
            ])
            tags = data.get("tags") or []
            return "、".join(str(t) for t in tags[:4])
        except LLMError:
            return ""

    # ---------- 文献汇报（组会级精读） ----------
    def handle_report(self, user_id, args, ctx):
        idx = args.get("index")
        if idx is not None:
            papers = ctx.get("last_papers") or []
            try:
                i = int(idx) - 1
            except (TypeError, ValueError):
                i = -1
            if not (0 <= i < len(papers)):
                return "没有找到对应的论文。先搜索文献再回复「汇报第1篇」，或直接发 PDF 给我。"
            paper = papers[i]
            cached = self._find_in_library(user_id, paper)
            if cached and REPORT_MARK in (cached["summary"] or ""):
                return ("《%s》之前已生成过文献汇报，直接复用文献库结果（标签：%s）：\n\n%s"
                        % (cached["title"], cached["tags"] or "无", cached["summary"]))
            if paper.get("source") == "arxiv":
                pdf_path = download_arxiv_pdf(paper["url"])
            else:
                pdf_path = None
            if not pdf_path:
                return self._report_from_abstract(user_id, paper)
            return self._report_from_pdf(user_id, pdf_path, paper["title"],
                                         paper["source"], paper["url"], ctx)
        # 无序号：按标题 > 刚上传的 PDF
        title = (args.get("title") or "").strip()
        if title:
            return self._report_by_title(user_id, title, ctx)
        pdf_path = ctx.get("last_pdf_path")
        if not pdf_path or not os.path.exists(pdf_path):
            return ("先给我一篇论文：发 PDF 文件、直接告诉我论文标题（arXiv 上的），"
                    "或搜索后回复「汇报第1篇」。")
        return self._report_from_pdf(user_id, pdf_path, os.path.basename(pdf_path),
                                     "pdf_upload", "", ctx)

    def _report_by_title(self, user_id, title, ctx=None):
        """按标题出汇报：先查文献库（有汇报直接复用），再按库存/搜索到的 arXiv 链接下全文。"""
        rows = self.db.search_papers(user_id, title[:50])
        row = rows[0] if rows else None
        if row and REPORT_MARK in (row["summary"] or ""):
            return ("《%s》文献汇报（文献库已有，标签：%s）：\n\n%s"
                    % (row["title"], row["tags"] or "无", row["summary"]))
        paper = None
        if row:
            paper = {"title": row["title"], "authors": "", "summary": row["summary"],
                     "url": row["url"], "source": row["source"]}
        else:
            results = search_papers(title, 3)
            if not results:
                return ("文献库和 arXiv 上都没找到《%s》。标题可以给得再准一点，"
                        "或直接把 PDF 发给我。" % title)
            paper = results[0]
        pdf_path = download_arxiv_pdf(paper["url"]) if paper.get("url") else None
        if not pdf_path:
            return self._report_from_abstract(user_id, paper)
        return self._report_from_pdf(user_id, pdf_path, paper["title"],
                                     paper.get("source") or "arxiv", paper["url"], ctx)

    def _report_from_pdf(self, user_id, pdf_path, title, source, url, ctx=None):
        try:
            text, truncated = extract_pdf_text(pdf_path)
        except Exception:
            return "PDF 解析失败了，换个文件试试。"
        if len(text.strip()) < 500:
            return "这个 PDF 提取不出足够正文，无法做文献汇报（可能是扫描件）。"
        from ..bot import make_stream
        stream = make_stream(self.bot, ctx)
        try:
            report = self.llm.chat([
                {"role": "system", "content": REPORT_PROMPT},
                {"role": "user", "content": "论文标题：%s\n正文%s：\n%s" % (
                    title, "（内容有截断）" if truncated else "", text)},
            ], on_delta=stream.update if stream else None)
        except Exception as e:
            if stream:
                stream.close("（生成中断：%s，请稍后再试）" % e)
            raise
        tags = self._extract_tags(title + " " + text[:1000])
        self._save_paper(user_id, title, source, url, report, tags)
        final = "《%s》文献汇报：\n\n%s\n\n（已存入文献库，标签：%s；之后可在文献库中问答回顾）" % (
            title, report, tags or "无")
        if stream:
            stream.close(final)
            if ctx is not None:
                ctx["_streamed_text"] = final
            return None  # 已通过流式回复定稿，router 跳过重复回复
        return final

    def _report_from_abstract(self, user_id, paper):
        """拿不到全文时（如 Semantic Scholar 来源或下载失败），基于摘要出汇报并说明。"""
        report = self.llm.chat([
            {"role": "system", "content":
                "你是擅长把论文讲明白的解读助手，读者是不了解该领域术语的初学者。"
                "只有论文标题和摘要可用，请基于摘要输出中文文献汇报，"
                "结构参照：【一句话总结】（大白话）【痛点与动机】【核心思路】（配类比或例子）"
                "【方法（据摘要推断）】（术语第一次出现要一句话解释）【主要结论】"
                "【局限与坑】【能学到什么】。"
                "禁止罗列作者和引用堆砌；摘要未覆盖的部分明确写「摘要未提及」，不要编造。"},
            {"role": "user", "content": "标题：%s\n作者：%s\n摘要：%s" % (
                paper["title"], paper["authors"], paper["summary"][:3000])},
        ])
        tags = self._extract_tags(paper["title"] + " " + paper["summary"])
        self._save_paper(user_id, paper["title"], paper["source"], paper["url"], report, tags)
        return ("《%s》文献汇报（基于摘要，全文获取失败）：\n\n%s\n\n（已存入文献库，标签：%s）"
                % (paper["title"], report, tags or "无"))

    # ---------- PDF 总结 ----------
    def summarize_pdf(self, user_id, pdf_path, file_name):
        text, truncated = extract_pdf_text(pdf_path)
        if len(text.strip()) < 200:
            return "这个 PDF 提取不出多少文字（可能是扫描件），暂时无法总结。可以试试文字版 PDF。"
        data = self.llm.chat_json([
            {"role": "system", "content": "你是论文解读助手，输出严谨、简洁的中文结构化总结。"},
            {"role": "user", "content":
                "以下是论文《%s》的正文%s，请输出 JSON：\n"
                "{\"title\": \"论文标题\", \"summary\": \"结构化总结\", \"tags\": [\"标签1\", \"标签2\"]}\n\n"
                "summary 按此格式组织：\n【研究问题】...\n【方法】...\n【实验与结果】...\n【结论】...\n【局限与展望】...\n\n"
                "tags 给 2-4 个中文技术标签。正文如下：\n%s"
                % (file_name, "（内容有截断）" if truncated else "", text)},
        ])
        title = (data.get("title") or file_name).strip()
        summary = (data.get("summary") or "").strip()
        tags = "、".join(str(t) for t in (data.get("tags") or [])[:4])
        if not summary:
            return "总结失败了，请稍后再试。"
        self._save_paper(user_id, title, "pdf_upload", "", summary, tags)
        note = "（注：PDF 较长，总结基于前 %d 页/前 %d 字）\n\n" % (PDF_MAX_PAGES, PDF_MAX_CHARS) if truncated else ""
        return ("%s《%s》解读：\n\n%s\n\n已存入你的文献库（标签：%s）。\n"
                "回复「生成文献汇报」可获取组会汇报级精读。" % (note, title, summary, tags or "无"))

    def classify_doc(self, text_head):
        """根据 PDF 开头判断文档类型：paper / resume / other。"""
        try:
            data = self.llm.chat_json([
                {"role": "user", "content":
                    "判断以下文档开头属于哪一类：学术论文 paper / 个人简历 resume / 其他 other。"
                    "输出 JSON：{\"doc_type\": \"paper\"}。文档开头：\n%s" % text_head[:2000]}
            ])
            doc_type = (data.get("doc_type") or "other").strip()
            return doc_type if doc_type in ("paper", "resume", "other") else "other"
        except LLMError:
            return "other"

    # ---------- 文献库 ----------
    def handle_library(self, user_id, args, raw_text, ctx=None):
        rows = self.db.list_papers(user_id, limit=20)
        if not rows:
            return ("你的文献库还是空的。可以把论文 PDF 直接发给我，"
                    "或说「帮我找 xxx 方向的论文」再「汇报第1篇」入库。")
        query = (args.get("query") or raw_text or "").strip()
        entries = []
        for r in rows:
            entries.append("【%d】《%s》标签:%s\n摘要要点:%s" % (
                r["id"], r["title"], r["tags"] or "无", (r["summary"] or "")[:LIBRARY_CTX_CHARS]))
        messages = [
            {"role": "system", "content":
                "你是用户的文献库助手。根据文献库条目回答用户问题："
                "若用户想浏览，则分类列出；若用户提问，则结合条目内容回答并指明出处。"
                "不要编造条目之外的信息。回答用中文，简洁。"},
            {"role": "user", "content":
                "我的文献库条目：\n%s\n\n我的问题：%s" % ("\n\n".join(entries), query)},
        ]
        from ..bot import make_stream
        stream = make_stream(self.bot, ctx)
        if not stream:
            return self.llm.chat(messages)
        try:
            final = self.llm.chat(messages, on_delta=stream.update)
        except Exception as e:
            stream.close("（生成中断：%s，请稍后再试）" % e)
            raise
        stream.close(final)
        ctx["_streamed_text"] = final
        return None  # 已通过流式回复定稿，router 跳过重复回复

    # ---------- 订阅 ----------
    def handle_subscribe(self, user_id, args):
        keywords = (args.get("keywords") or "").strip()
        enable = args.get("enable", True)
        if not enable:
            self.db.set_subscription(user_id, keywords or "", enabled=False)
            return "已关闭每日论文订阅推送。"
        if not keywords:
            return ("想订阅什么方向？可以直接描述你的研究需求，"
                    "比如：订阅多智能体强化学习和信用分配方向的论文，每天早上九点推五篇。")
        digest_time = (args.get("time") or "").strip() or None
        top_n = args.get("top_n")
        try:
            top_n = int(top_n) if top_n else None
        except (TypeError, ValueError):
            top_n = None
        old = self.db.get_subscription(user_id)
        self.db.set_subscription(user_id, keywords, enabled=True,
                                 digest_time=digest_time, top_n=top_n)
        show_time = digest_time or self.cfg.paper_digest
        show_n = top_n or self.cfg.paper_digest_top_n
        note = ""
        if old and old["keywords"] and old["keywords"] != keywords:
            note = "\n（已替换原订阅「%s」；想换回来直接说）" % old["keywords"]
        return ("已订阅「%s」。\n每天 %s 从 arXiv 新论文中按你的需求筛选 %d 篇推送（附推荐理由）。%s\n"
                "回复「取消论文订阅」可关闭；想改方向/时间/篇数直接说。" % (keywords, show_time, show_n, note))

    def daily_digest(self, user_id, need, top_n):
        """定时任务调用：按用户需求筛选新论文。
        返回 (推送文本 or None, 入选论文列表, 检索是否成功)。
        检索成功但无新论文 → (None, [], True)，当天不再重试；
        检索失败（如 API 限流）→ (None, [], False)，下轮应重试。
        周一回看 3 天（arXiv 周末不更新，2 天窗口会漏掉周五的新论文）。"""
        days = 3 if datetime.now().weekday() == 0 else 2
        since = (datetime.now() - timedelta(days=days)).date()  # 按日期比较，不带时刻
        queries = self._expand_queries(need, user_id)
        seen, candidates, any_result = set(), [], False
        for q in queries:
            results = search_papers(q, 10)
            if results:
                any_result = True
            for p in results:
                key = p["url"] or p["title"]
                if key in seen:
                    continue
                try:
                    pub = datetime.strptime(p["published"], "%Y-%m-%d").date()
                except ValueError:
                    continue
                if pub >= since:
                    seen.add(key)
                    candidates.append(p)
        if not candidates:
            return None, [], any_result
        pool = self._hard_filter(user_id, candidates, queries, top_n)
        if not pool:
            return None, [], True
        picked = self._rerank(need, pool, top_n, user_id)
        lines = ["【论文日报】与你的方向「%s」最相关的新论文：\n" % need]
        for i, (p, reason) in enumerate(picked, 1):
            lines.append("%d. 《%s》(%s)" % (i, p["title"], p["published"]))
            if reason:
                lines.append("   推荐理由：%s" % reason)
            lines.append("   %s" % p["url"])
        lines.append("\n回复「总结第N篇」快速解读，「汇报第N篇」全文精读；也可以把 PDF 直接发给我。")
        return "\n".join(lines), [p for p, _ in picked], True
