from __future__ import annotations

from newton_lab.tasks.builtin import register_builtin_tasks
from newton_lab.tasks.registry import TaskSpec, get_task_spec, list_tasks, register_task


_IMPORTED = False


def import_tasks() -> None:
    global _IMPORTED
    if _IMPORTED:
        return
    register_builtin_tasks()
    _IMPORTED = True


__all__ = ["TaskSpec", "get_task_spec", "import_tasks", "list_tasks", "register_task"]
