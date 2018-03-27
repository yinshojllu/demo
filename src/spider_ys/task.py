#!/usr/bin/env python
# -*- coding: utf-8 -*-
from uuid import uuid4
from redis import StrictRedis
from pymongo import MongoClient


class Status_Queue():
    """
    任务状态队列

    :param name: 任务队列名
    :type name: str
    """
    def __init__(self, acct, name="status-queue"):
        self.status_queue = acct.get_queue(name)
        if self.status_queue is None:
            raise Exception("状态队列不存在")

    def change_status(self, task, status):
        """
        修改任务状态

        :param task: 任务实例
        :param status: 状态
        :type task: Task
        :type status: str
        :returns: True, 失败抛出异常
        :rtype: bool
        """
        task_status = {
            "task_id": task.task_id,
            "status": status,
        }
        self.status_queue.send_msg(task_status, 8)
        return True

    def _get_status(self):
        while True:
            msg = self.status_queue.receive_msg()
            task_id = msg["task_id"]
            status = msg["status"]
            yield (task_id, status)

    def _pump_status(self, redis_uri, expire=7200):
        r = StrictRedis.from_url(redis_uri)
        for task_id, status in self._get_status():
            r.set(task_id, status)
            r.expire(task_id, expire)


class Data_Store():
    """
    任务结果存储
    """
    def __init__(self, mongo_uri):
        client = MongoClient(mongo_uri, connectTimeoutMS=10000)
        db = client.tasks_database
        self.c = db.result

    def put_data(self, task, data, expire=None):
        """
        存储任务结果

        :param task: 任务实例
        :param data: 任务结果
        :param expire: 存储过期时间
        :type task: Task
        :type data: dict
        :type expire: datatime, None
        :returns: True or 抛出异常
        :rtype: bool
        """
        self.c.insert_one({
            "task_id": task.task_id,
            "data": data,
            "expire_time": expire,
        })
        return True

    def query_data(self, task):
        """
        检索任务结果

        :param task: 任务实例
        :type task: Task
        :returns: 查询结果
        :rtype: dict
        """
        query = {"task_id": task.task_id}
        data = self.c.find_one(query)
        if data is None:
            return None
        else:
            return data["data"]


class Task_Queue():
    """
    任务队列

    :param acct: 阿里云账户
    :param name: 任务名
    :type acct: Account
    :type name: str
    """
    def __init__(self, acct, name, status_queue):
        self.task_queue = acct.get_queue(name)
        if self.task_queue is None:
            raise Exception("任务队列不存在")
        self.status_queue = status_queue

    def push_task(self, data, priority):
        """
        推送任务

        :param data: 任务数据
        :param priority: 任务优先级
        :type data: dict
        :type priority: int
        :returns: 任务实例
        :rtype: Task
        """
        task_id = str(uuid4())
        pkg = {
            "task_id": task_id,
            "data": data,
        }
        self.task_queue.send_msg(pkg, priority)
        task = Task(task_id, data, self.status_queue)
        task.change_status("PENDING")
        return task

    def pop_task(self):
        """
        获取任务 generator

        :returns: 任务实例
        :rtype: Task
        """
        while True:
            msg = self.task_queue.receive_msg()
            task_id = msg["task_id"]
            data = msg["data"]
            task = Task(task_id, data)
            yield task

    def running_loop(self, runner, store):
        """
        轮询任务执行循环

        :param runner: 任务执行者
        :param store: 结果存储
        :type runner: function
        :type store: Data_Store
        """
        for task in self.pop_task():
            task.change_status("WORKING")
            try:
                result = task.run(runner)
                store.put_data(task, result)
            except Exception:
                task.change_status("ERROR")


class Task():
    """
    任务

    :param task_id: 任务id
    :param data: 任务数据
    :param status_queue: 任务状态队列
    :type task_id: str, uuid4
    :type data: dict
    :type status_queue: Status_Queue
    """
    def __init__(self, task_id, data, status_queue):
        self.task_id = task_id
        self.data = data
        self.status_queue = status_queue

    def change_status(self, status):
        """
        改变任务状态

        :param status: 状态
        :type status: str
        """
        self.status_queue.change_status(self, status)

    def run(self, runner):
        """
        执行任务

        :param runner: 任务执行函数
        :type runner: function
        """
        result = runner(self.data)
        return result
