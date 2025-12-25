# src/chatbot/main.py
from __future__ import annotations

import asyncio
import logging
import signal

from chatbot.bootstrap import build_container
from chatbot.settings import Settings


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    asyncio.run(_amain())


async def _amain() -> None:
    # 先初始化 settings + logging，避免 build_container 阶段日志丢失
    settings = Settings()
    _setup_logging(settings.LOG_LEVEL)

    container = build_container(settings=settings)

    log = logging.getLogger("chatbot.main")
    stop_evt = asyncio.Event()

    def _request_stop() -> None:
        if not stop_evt.is_set():
            stop_evt.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            pass

    consumer_task = asyncio.create_task(container.consumer.run_forever(), name="rocketmq-consumer")
    scheduler_task = asyncio.create_task(container.scheduler.run_forever(), name="send-scheduler")

    stop_wait_task = asyncio.create_task(stop_evt.wait(), name="stop-wait")

    log.info("service started")
    try:
        done, pending = await asyncio.wait(
            [consumer_task, scheduler_task, stop_wait_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # 任一工作任务异常退出也应触发停止
        for t in done:
            if t is stop_wait_task:
                continue
            exc = t.exception()
            if exc is not None:
                log.exception("worker exited with exception", exc_info=exc)
                stop_evt.set()

    finally:
        # 请求各组件停止（若实现了 stop）
        try:
            stop = getattr(container.consumer, "stop", None)
            if callable(stop):
                stop()
        except Exception:
            pass

        try:
            stop = getattr(container.scheduler, "stop", None)
            if callable(stop):
                stop()
        except Exception:
            pass

        # 取消任务并等待退出
        for t in (consumer_task, scheduler_task, stop_wait_task):
            if not t.done():
                t.cancel()

        await asyncio.gather(consumer_task, scheduler_task, stop_wait_task, return_exceptions=True)

        await container.aclose()
        log.info("service stopped")

if __name__ == "__main__":
    main()
