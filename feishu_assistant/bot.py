"""飞书长连接机器人层：收发消息、下载文件、主动推送。

只依赖 lark-oapi 官方 SDK 的稳定用法；真实收发需要 config.yaml 里的 app_id/app_secret。
"""
import json
import os
import re
import threading

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    GetMessageResourceRequest,
    P2ImMessageReceiveV1,
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


class FeishuBot:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = (
            lark.Client.builder()
            .app_id(cfg.feishu_app_id)
            .app_secret(cfg.feishu_app_secret)
            .log_level(lark.LogLevel.INFO)
            .build()
        )
        self._handler = None          # 由 router 注入：fn(user_id, chat_id, message_id, msg_type, content)
        self._seen_ids = set()        # 事件去重（飞书可能重投）
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

            with self._seen_lock:
                if message_id in self._seen_ids:
                    return
                self._seen_ids.add(message_id)
                if len(self._seen_ids) > 5000:
                    self._seen_ids = set(list(self._seen_ids)[-2500:])

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

            if self._handler:
                threading.Thread(
                    target=self._safe_handle,
                    args=(user_id, chat_id, message_id, msg_type, content),
                    daemon=True,
                ).start()
        except Exception as e:
            lark.logger.exception("处理消息事件出错: %s", e)

    def _safe_handle(self, user_id, chat_id, message_id, msg_type, content):
        try:
            self._handler(user_id, chat_id, message_id, msg_type, content)
        except Exception as e:
            lark.logger.exception("消息处理异常: %s", e)
            try:
                self.reply(message_id, "抱歉，处理这条消息时出错了：%s" % e)
            except Exception:
                pass

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
