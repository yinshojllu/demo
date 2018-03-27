#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mns.queue import Message, QueueMeta, MNSExceptionBase
from mns.account import Account
from urllib.parse import urlparse
import json
from . import logger


class Aliyun_Account():
    """
    阿里云账号包装类

    :param host: 阿里云 mns url
    :param access_id: 阿里云 mns access_id
    :param access_key: 阿里云 mns access_key
    :param security_token: 阿里云 mns security_token
    :type host: str
    :type access_id: str
    :type access_key: str
    :type security_token: str
    """
    def __init__(self, host, access_id, access_key, security_token=""):
        self.acct = Account(host, access_id, access_key, security_token)

    def list_queue(self, prefix=""):
        """
        获取账户下的队列名列表

        :param prefix: 队列名前缀筛选
        :type prefix: str
        :returns: 队列名列表
        :rtype: list
        """
        marker = ""
        queues = []
        while(True):
            queue_url_list, marker = self.acct.list_queue(prefix, 1000, marker)
            for queue_url in queue_url_list:
                queue_name = urlparse(queue_url).path.split('/')[-1]
                queues.append(queue_name)
            if(marker == ""):
                break
        return queues

    def has_queue(self, name):
        """
        检查是否存在某队列

        :param name: 队列名
        :type name: str
        :returns: 是否存在
        :rtype: bool
        """
        return name in self.list_queue()

    def get_queue(self, name):
        """
        获取队列, 如不存在, 返回None

        :param name: 队列名
        :type name: str
        :returns: 队列实例
        :rtype: Queue, None
        """
        if self.has_queue(name):
            return Queue(self.acct.get_queue(name))
        else:
            return None

    def create_queue(self, name, timeout=3600):
        """
        新建队列

        :param name: 队列名
        :type name: str
        :returns: 队列实例
        :rtype: Queue
        """
        queue = self.acct.get_queue(name)
        queue_meta = QueueMeta()
        queue_meta.set_visibilitytimeout(timeout)
        queue_meta.set_maximum_message_size(10240)
        queue_meta.set_message_retention_period(timeout)
        queue_meta.set_delay_seconds(0)
        queue_meta.set_polling_wait_seconds(20)
        queue_meta.set_logging_enabled(True)
        queue.create(queue_meta)
        return Queue(queue)


class Queue():
    """
    阿里云队列包装类

    :param queue: 阿里云 mns queue 实例
    :type queue: mns.queue.Queue
    """
    def __init__(self, queue):
        self.name = queue.queue_name
        self.queue = queue

    def send_msg(self, msg, priority):
        """
        发送消息

        :param msg: 消息体
        :param priority: 优先级
        :type msg: dict
        :type priority: int
        :returns: 是否成功
        :rtype: bool
        """
        msg_body = json.dumps(msg)
        message = Message(msg_body)
        message.set_delayseconds(0)
        message.set_priority(priority)
        try:
            send_msg = self.queue.send_message(message)
            logger.info(
                "Send Message id: %s, body: %s" %
                (send_msg.message_id, msg_body))
            return True
        except MNSExceptionBase as e:
            logger.error("Send Message Fail!\nException:%s\n\n" % e)
            return False

    def receive_msg(self):
        """
        接收消息, long pull 阻塞直到消息返回

        :returns: 消息体
        :rtype: dict
        """
        while True:
            try:
                wait_seconds = 10
                recv_msg = self.queue.receive_message(wait_seconds)
                logger.info(
                    "Receive Message, id: %s, body: %s" %
                    (recv_msg.message_id, recv_msg.message_body))
                return json.loads(recv_msg.message_body)
            except MNSExceptionBase as e:
                logger.info("outtime, retry")
                continue
