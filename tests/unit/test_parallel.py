"""Unit tests for auto_art.core.parallel."""
from auto_art.core.parallel import AttackTask, ParallelAttackRunner


def _ok_payload(x=0):
    return {"v": x}


def _boom():
    raise RuntimeError("task error")


def test_parallel_runner_basic():
    runner = ParallelAttackRunner(max_cpu_workers=2, max_gpu_workers=1, default_timeout=30.0)
    tasks = [
        AttackTask(name="a", callable=_ok_payload, kwargs={"x": 1}),
        AttackTask(name="b", callable=_ok_payload, kwargs={"x": 2}),
        AttackTask(name="c", callable=_ok_payload, kwargs={"x": 3}),
    ]
    results = runner.run(tasks)
    by_name = {r.name: r for r in results}
    assert len(by_name) == 3
    assert all(by_name[k].success for k in ("a", "b", "c"))
    assert by_name["b"].data["v"] == 2


def test_parallel_runner_error():
    runner = ParallelAttackRunner(max_cpu_workers=2, default_timeout=30.0)
    results = runner.run([AttackTask(name="bad", callable=_boom)])
    assert results[0].success is False
    assert "task error" in (results[0].error or "")


def test_parallel_runner_gpu_and_cpu():
    runner = ParallelAttackRunner(max_cpu_workers=2, max_gpu_workers=1)
    gpu_task = AttackTask(name="gpu", callable=_ok_payload, kwargs={"x": 10}, use_gpu=True)
    cpu_task = AttackTask(name="cpu", callable=_ok_payload, kwargs={"x": 20}, use_gpu=False)
    results = runner.run([gpu_task, cpu_task])
    by_name = {r.name: r for r in results}
    assert len(by_name) == 2
    assert by_name["gpu"].success and by_name["cpu"].success
    assert by_name["gpu"].data["v"] == 10
    assert by_name["cpu"].data["v"] == 20


def test_parallel_runner_empty():
    runner = ParallelAttackRunner()
    results = runner.run([])
    assert results == []
