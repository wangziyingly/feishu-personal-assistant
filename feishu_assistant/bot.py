"""飞书长连接机器人层：收发消息、下载文件、主动推送、流式回复、入站事件持久化。

只依赖 lark-oapi 官方 SDK 的稳定用法；真实收发需要 config.yaml 里的 app_id/app_secret。
入站事件先落库（db.inbox_events）再处理，重启/崩溃后重放，重复事件按 message_id 去重。
"""
import json
import os
import re
import threading
import time

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    GetMessageResourceRequest,
    P2ImMessageReceiveV1,
    PatchMessageRequest,
    PatchMessageRequestBody,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)

# 飞书单条文本消息的安全长度，超长自动拆分
MAX_CHUNK = 3000


def _split_text(text, size=MAX_CHUNK):
    """按换行优先切成多段，避免一条消息过长被拒。"""
    if len(text) <= size:
        return [text]
    chunks, buf = [], ""
    for line in text.split("\n"):
        if len(buf) + len(line) + 1 > size:
            if buf:
                chunks.append(buf)
            while len(line) > size:  # 单行也超长则硬切
                chunks.append(line[:size])
                line = line[size:]
            buf = line
        else:
            buf = buf + "\n" + line if buf else line
    if buf:
        chunks.append(buf)
    return chunks


def _card_json(text):
    """把纯文本包成飞书卡片 JSON。飞书的 PATCH 更新消息只支持卡片（interactive）消息，
    流式回复的占位和刷新都必须走卡片。"""
    return json.dumps({
        "config": {"wide_screen_mode": True},
        "elements": [{"tag": "div", "text": {"tag": "plain_text", "content": text}}],
    }, ensure_ascii=False)


class FeishuBot:
    def __init__(self, cfg, db=None):
        self.cfg = cfg
        self.db = db  # 入站事件持久化用；不传则退化为内存去重
        self.client = (
            lark.Client.builder()
            .app_id(cfg.feishu_app_id)
            .app_secret(cfg.feishu_app_secret)
            .log_level(lark.LogLevel.INFO)
            .build()
        )
        self._handler = None          # 由 router 注入：fn(user_id, chat_id, message_id, msg_type, content)
        self._seen_ids = set()        # 事件去重（无 db 时的内存兜底）
        self._seen_lock = threading.Lock()
        self._ws_client = None

    def on_message(self, handler):
        """注册消息处理回调：handler(user_id, chat_id, message_id, msg_type, content:dict)"""
        self._handler = handler

    # ---------- 事件入口 ----------
    def _do_message_receive(self, data: P2ImMessageReceiveV1):
        try:
            event = data.event
            msg = event.message
            message_id = msg.message_id

            chat_type = msg.chat_type  # p2p / group
            mentions = msg.mentions or []
            if chat_type != "p2p" and not mentions:
                return  # 群聊只在被 @ 时响应

            user_id = event.sender.sender_id.open_id
            chat_id = msg.chat_id
            msg_type = msg.message_type
            try:
                content = json.loads(msg.content or "{}")
            except json.JSONDecodeError:
                content = {}

            if msg_type == "text":
                text = content.get("text", "")
                for m in mentions:  # 去掉 @机器人 占位符
                    text = text.replace(m.key, "").strip()
                content["text"] = text.strip()
                if not content["text"]:
                    return

            # 先落库再处理：重启/崩溃后可在 replay_pending 中恢复；重复事件（飞书重投）去重
            if self.db:
                payload = {"user_id": user_id, "chat_id": chat_id, "message_id": message_id,
                           "msg_type": msg_type, "content": content}
                if not self.db.inbox_enqueue(message_id, payload):
                    return
                # 取刚插入那行的 id（payload 里没带，用 message_id 查）
                row = self.db._query("SELECT id FROM inbox_events WHERE message_id=?", (message_id,))
                event_id = row[0]["id"] if row else None
            else:
                with self._seen_lock:
                    if message_id in self._seen_ids:
                        return
                    self._seen_ids.add(message_id)
                event_id = None

            if self._handler:
                threading.Thread(
                    target=self._process_event,
                    args=(event_id, user_id, chat_id, message_id, msg_type, content),
                    daemon=True,
                ).start()
        except Exception as e:
            lark.logger.exception("处理消息事件出错: %s", e)

    def _process_event(self, event_id, user_id, chat_id, message_id, msg_type, content):
        try:
            self._handler(user_id, chat_id, message_id, msg_type, content)
            if self.db and event_id:
                self.db.inbox_mark(event_id, "done")
        except Exception as e:
            lark.logger.exception("消息处理异常: %s", e)
            if self.db and event_id:
                self.db.inbox_mark(event_id, "failed")
            try:
                self.reply(message_id, "抱歉，处理这条消息时出错了：%s" % e)
            except Exception:
                pass

    # ---------- 重启重放 ----------
    def replay_pending(self):
        """重放入站队列中未完成的事件（重启/睡眠恢复后调用）。"""
        if not self.db or not self._handler:
            return 0
        rows = self.db.inbox_pending()
        if rows:
            lark.logger.info("重放 %d 条未处理的入站事件", len(rows))
        for row in rows:
            try:
                p = json.loads(row["payload"])
                threading.Thread(
                    target=self._process_event,
                    args=(row["id"], p["user_id"], p["chat_id"], p["message_id"],
                          p["msg_type"], p["content"]),
                    daemon=True,
                ).start()
            except Exception as e:
                lark.logger.error("重放事件失败(id=%s): %s", row["id"], e)
        return len(rows)

    # ---------- 发消息 ----------
    def reply(self, message_id, text):
        """回复指定消息（自动拆长文）。"""
        for chunk in _split_text(text):
            req = (
                ReplyMessageRequest.builder()
                .message_id(message_id)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .content(json.dumps({"text": chunk}, ensure_ascii=False))
                    .msg_type("text")
                    .build()
                )
                .build()
            )
            resp = self.client.im.v1.message.reply(req)
            if not resp.success():
                lark.logger.error("回复消息失败: code=%s msg=%s", resp.code, resp.msg)
                break

    def reply_get_id(self, message_id, text):
        """以卡片形式回复并返回新消息的 message_id（流式回复占位用）；失败返回 None。
        必须用卡片：纯文本消息不支持 PATCH 更新（飞书限制）。"""
        req = (
            ReplyMessageRequest.builder()
            .message_id(message_id)
            .request_body(
                ReplyMessageRequestBody.builder()
                .content(_card_json(text))
                .msg_type("interactive")
                .build()
            )
            .build()
        )
        resp = self.client.im.v1.message.reply(req)
        if not resp.success():
            lark.logger.error("回复消息失败: code=%s msg=%s", resp.code, resp.msg)
            return None
        return resp.data.message_id

    def patch_message(self, mid, text):
        """更新已发送卡片消息的内容（流式回复的渐进刷新）。成功返回 True，失败返回 False。"""
        try:
            req = (
                PatchMessageRequest.builder()
                .message_id(mid)
                .request_body(
                    PatchMessageRequestBody.builder()
                    .content(_card_json(text))
                    .build()
                )
                .build()
            )
            resp = self.client.im.v1.message.patch(req)
        except Exception as e:
            lark.logger.error("更新消息异常: %s", e)
            return False
        if not resp.success():
            lark.logger.error("更新消息失败: code=%s msg=%s", resp.code, resp.msg)
            return False
        return True

    def push(self, chat_id, text):
        """主动向会话推送消息（提醒/早报/订阅推送用）。"""
        for chunk in _split_text(text):
            req = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type("text")
                    .content(json.dumps({"text": chunk}, ensure_ascii=False))
                    .build()
                )
                .build()
            )
            resp = self.client.im.v1.message.create(req)
            if not resp.success():
                lark.logger.error("推送消息失败: code=%s msg=%s", resp.code, resp.msg)
                break

    # ---------- 文件下载 ----------
    def download_file(self, message_id, file_key, save_dir, file_name):
        """下载消息中的文件到本地，返回保存路径；失败返回 None。"""
        req = (
            GetMessageResourceRequest.builder()
            .message_id(message_id)
            .file_key(file_key)
            .type("file")
            .build()
        )
        resp = self.client.im.v1.message_resource.get(req)
        if not resp.success():
            lark.logger.error("下载文件失败: code=%s msg=%s", resp.code, resp.msg)
            return None
        os.makedirs(save_dir, exist_ok=True)
        safe_name = os.path.basename(file_name or "download.bin")
        path = os.path.join(save_dir, safe_name)
        with open(path, "wb") as f:
            f.write(resp.file.read())
        return path

    def download_image(self, message_id, image_key, save_dir):
        """下载消息中的图片到本地，返回保存路径；失败返回 None。"""
        req = (
            GetMessageResourceRequest.builder()
            .message_id(message_id)
            .file_key(image_key)
            .type("image")
            .build()
        )
        resp = self.client.im.v1.message_resource.get(req)
        if not resp.success():
            lark.logger.error("下载图片失败: code=%s msg=%s", resp.code, resp.msg)
            return None
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "img_%s.jpg" % re.sub(r"[^\w]", "", image_key)[:20])
        with open(path, "wb") as f:
            f.write(resp.file.read())
        return path

    # ---------- 启动 ----------
    def start(self):
        """启动 WebSocket 长连接（阻塞）。"""
        handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._do_message_receive)
            .build()
        )
        self._ws_client = lark.ws.Client(
            self.cfg.feishu_app_id,
            self.cfg.feishu_app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.INFO,
        )
        self._ws_client.start()


class StreamReply:
    """流式回复：先发占位消息，LLM 边生成边 PATCH 刷新，close 定稿。

    用法：stream = StreamReply(bot, message_id)；llm.chat(..., on_delta=stream.update)；
    最后 stream.close(完整文本)。close 后模块应返回 None，由 router 跳过重复回复。
    """

    def __init__(self, bot, message_id, placeholder="▍正在生成，请稍候…"):
        self._bot = bot
        self._orig_mid = message_id
        self._last = 0.0
        self.mid = bot.reply_get_id(message_id, placeholder) if message_id else None

    def update(self, text):
        if not self.mid:
            return
        now = time.time()
        if now - self._last < 1.5:  # PATCH 节流，避免触发飞书频控
            return
        self._last = now
        self._bot.patch_message(self.mid, text + " ▍")

    def close(self, text):
        if self.mid and self._bot.patch_message(self.mid, text):
            return
        if self._orig_mid:  # 占位发送/定稿刷新失败时退化为普通回复，保证内容不丢
            self._bot.reply(self._orig_mid, text)


def make_stream(bot, ctx):
    """从 ctx 取当前消息 id 创建流式回复；无 bot/无消息 id 时返回 None（退化为普通回复）。"""
    if not bot or not ctx:
        return None
    mid = ctx.pop("current_message_id", None)
    return StreamReply(bot, mid) if mid else None
