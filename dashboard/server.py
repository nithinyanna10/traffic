"""
FastAPI dashboard server with WebSocket streaming.

Endpoints:
    GET  /                      - Serves the live dashboard
    WS   /ws/state              - Streams live environment state
    POST /api/start             - Starts training (steps, checkpoint, curriculum)
    POST /api/stop               - Stops training (saves PPO checkpoint on stop)
    GET  /api/status             - Returns current training status
    GET  /api/metrics            - Returns training metrics history (for charts)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="RL-Traffic-Control Dashboard")

# Static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global state
_state_lock = threading.Lock()
_current_state: dict = {}
_training_active = False
_training_thread: Optional[threading.Thread] = None
_connected_clients: list[WebSocket] = []
_training_metrics: list[dict] = []
_replay_buffer: list[dict] = []
_REPLAY_MAX = 500
_stress_until_step: Optional[int] = None


def update_state(state: dict) -> None:
    """Called by the training loop to push new state."""
    global _current_state
    with _state_lock:
        _current_state = state.copy()


def get_state() -> dict:
    with _state_lock:
        return _current_state.copy()


def add_metrics(metrics: dict) -> None:
    """Append episode metrics; use keys compatible with visualize.py (total_reward, success_rate, etc.)."""
    global _training_metrics
    # Normalize for visualize.py: episode, reward -> total_reward, plus success_rate, avg_latency, avg_cost
    m = {
        "episode": metrics.get("episode", len(_training_metrics)),
        "total_reward": metrics.get("reward", metrics.get("total_reward", 0)),
        "success_rate": metrics.get("success_rate", 0),
        "avg_latency": metrics.get("avg_latency", 0),
        "avg_cost": metrics.get("avg_cost", 0),
        "steps": metrics.get("steps", 0),
    }
    _training_metrics.append(m)
    if len(_training_metrics) > 10_000:
        _training_metrics = _training_metrics[-5000:]


_SERVICE_NAMES = ["Fast GPU", "Cheap CPU", "Flaky Spot", "Serverless", "Queue Cluster"]
_EVENT_LOG: list[dict] = []
_MAX_EVENTS = 20


def _append_last_event(state: dict, action: int, reward: float, info: dict) -> None:
    """Set state['last_event'] for dashboard ticker; optionally append to event log."""
    global _EVENT_LOG
    event_type = "reward" if reward >= 0 else ""
    if action <= 4:
        svc = _SERVICE_NAMES[action] if action < len(_SERVICE_NAMES) else f"Service {action}"
        state["last_event"] = {"text": f"→ {svc} (r: {reward:.2f})", "type": event_type}
    elif action == 5:
        state["last_event"] = {"text": "Delay request", "type": ""}
    elif action == 6:
        state["last_event"] = {"text": f"Retry (r: {reward:.2f})", "type": event_type}
    elif action == 7:
        state["last_event"] = {"text": "Drop request", "type": "fail"}
    elif action == 8:
        state["last_event"] = {"text": f"Degrade (r: {reward:.2f})", "type": event_type}
    else:
        state["last_event"] = None
    resp = info.get("response")
    if resp and resp.get("cascade"):
        state["last_event"] = {"text": "Cascade failure!", "type": "cascade"}
    if state.get("last_event"):
        _EVENT_LOG.append(state["last_event"])
        if len(_EVENT_LOG) > _MAX_EVENTS:
            _EVENT_LOG.pop(0)
        state["events"] = _EVENT_LOG[-8:]  # send last few so new clients get recent


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard HTML."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>Dashboard — static/index.html not found</h1>")


@app.get("/metrics", response_class=HTMLResponse)
async def metrics_page():
    """Serve the metrics chart page (full-session curves from /api/metrics)."""
    metrics_path = STATIC_DIR / "metrics.html"
    if metrics_path.exists():
        return HTMLResponse(content=metrics_path.read_text())
    return HTMLResponse(content="<h1>metrics.html not found</h1>")


@app.get("/replay", response_class=HTMLResponse)
async def replay_page():
    """Serve the replay viewer (last N state snapshots from /api/replay)."""
    replay_path = STATIC_DIR / "replay.html"
    if replay_path.exists():
        return HTMLResponse(content=replay_path.read_text())
    return HTMLResponse(content="<h1>replay.html not found</h1>")


@app.websocket("/ws/state")
async def ws_state(websocket: WebSocket):
    """Stream live environment + training state to clients."""
    await websocket.accept()
    _connected_clients.append(websocket)
    try:
        while True:
            state = get_state()
            if state:
                state["training_active"] = _training_active
                await websocket.send_json(state)
            await asyncio.sleep(0.1)  # 10 FPS
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _connected_clients:
            _connected_clients.remove(websocket)


@app.post("/api/start")
async def start_training(
    agent_type: str = "ppo",
    steps: int = 50_000,
    checkpoint: Optional[str] = None,
    curriculum: bool = True,
):
    """Start training in a background thread. Optional: checkpoint path to load, curriculum on/off."""
    global _training_active, _training_thread

    if _training_active:
        return {"status": "already_running"}

    global _EVENT_LOG
    _EVENT_LOG = []
    _training_active = True

    def _run():
        global _training_active
        try:
            _run_training(agent_type, total_steps=steps, checkpoint_path=checkpoint, use_curriculum=curriculum)
        finally:
            _training_active = False

    _training_thread = threading.Thread(target=_run, daemon=True)
    _training_thread.start()
    return {"status": "started", "agent": agent_type, "steps": steps, "checkpoint": checkpoint, "curriculum": curriculum}


@app.get("/api/metrics")
async def get_metrics(last: int = 500):
    """Return last N episode metrics (for full-session charts). Keys: episode, total_reward, success_rate, avg_latency, avg_cost."""
    return {"metrics": _training_metrics[-last:] if _training_metrics else []}


@app.post("/api/stop")
async def stop_training():
    """Stop training (sets flag; training loop should check)."""
    global _training_active
    _training_active = False
    return {"status": "stopped"}


@app.get("/api/status")
async def training_status():
    """Current training status and recent metrics."""
    return {
        "training_active": _training_active,
        "state": get_state(),
        "recent_metrics": _training_metrics[-20:] if _training_metrics else [],
    }


@app.get("/api/replay")
async def get_replay(last: int = 100):
    """Return last N state snapshots for episode replay (step, episode, services, etc.)."""
    global _replay_buffer
    return {"replay": _replay_buffer[-last:] if _replay_buffer else []}


@app.post("/api/stress")
async def start_stress(duration_steps: int = 100):
    """Enable temporary stress: high traffic for the next duration_steps (training loop must be running)."""
    global _stress_until_step
    import threading as T
    state = get_state()
    current_step = state.get("global_step", 0)
    _stress_until_step = current_step + duration_steps
    return {"status": "stress_scheduled", "until_step": _stress_until_step, "duration": duration_steps}


def _run_training(
    agent_type: str,
    total_steps: int,
    checkpoint_path: Optional[str] = None,
    use_curriculum: bool = True,
):
    """Run training in a thread, pushing state updates. Uses curriculum if use_curriculum."""
    import gymnasium as gym
    import env as _env_reg  # noqa

    environment = gym.make("MicroserviceRouting-v0")
    unwrapped = environment.unwrapped

    curriculum_scheduler = None
    if use_curriculum:
        from training.curriculum import CurriculumScheduler
        curriculum_scheduler = CurriculumScheduler()

    if agent_type == "ppo":
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(device="cpu")
        if checkpoint_path and os.path.isfile(checkpoint_path + ".zip"):
            try:
                agent.load(checkpoint_path)
            except Exception:
                pass
        obs, info = environment.reset()
        update_state(info)
        episode = 0
        episode_reward = 0.0

        for step in range(total_steps):
            if not _training_active:
                break
            if curriculum_scheduler and step % 50 == 0:
                progress = step / max(total_steps, 1)
                config = curriculum_scheduler.get_config(progress)
                rate = 3.0 if (_stress_until_step is not None and step < _stress_until_step) else config.traffic_rate
                unwrapped.set_traffic_rate(rate)
                unwrapped.set_burst_config(config.burst_probability, config.burst_size)
                unwrapped.set_failure_injection(config.failure_injection_prob)

            action = agent.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = environment.step(action)
            episode_reward += reward

            state = info.copy()
            state["episode"] = episode
            state["episode_reward"] = episode_reward
            state["global_step"] = step
            state["agent_type"] = agent_type
            state["current_action"] = action
            state["last_reward"] = reward
            _append_last_event(state, action, reward, info)
            update_state(state)
            _replay_buffer.append(state.copy())
            if len(_replay_buffer) > _REPLAY_MAX:
                _replay_buffer.pop(0)

            if terminated or truncated:
                add_metrics({
                    "episode": episode,
                    "reward": episode_reward,
                    "steps": info.get("step", 0),
                    "success_rate": info.get("success_rate", 0),
                    "avg_latency": info.get("avg_latency", 0),
                    "avg_cost": info.get("avg_cost", 0),
                })
                obs, info = environment.reset()
                episode_reward = 0.0
                episode += 1

        try:
            ckpt_dir = Path(__file__).parent.parent / "checkpoints" / "ppo"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            agent.save(str(ckpt_dir / "dashboard_latest"))
        except Exception:
            pass

    else:
        from agents.baselines import BASELINES
        agent_cls = BASELINES.get(agent_type)
        if agent_cls:
            agent = agent_cls()
        else:
            environment.close()
            return

        obs, info = environment.reset()
        update_state(info)
        episode = 0
        episode_reward = 0.0

        if curriculum_scheduler:
            for step in range(total_steps):
                if not _training_active:
                    break
                if step % 50 == 0:
                    progress = step / max(total_steps, 1)
                    config = curriculum_scheduler.get_config(progress)
                    unwrapped.set_traffic_rate(config.traffic_rate)
                    unwrapped.set_burst_config(config.burst_probability, config.burst_size)
                    unwrapped.set_failure_injection(config.failure_injection_prob)
                action = agent.predict(obs)
                obs, reward, terminated, truncated, info = environment.step(action)
                episode_reward += reward
                state = info.copy()
                state["episode"] = episode
                state["episode_reward"] = episode_reward
                state["global_step"] = step
                state["agent_type"] = agent_type
                state["current_action"] = action
                state["last_reward"] = reward
                _append_last_event(state, action, reward, info)
                update_state(state)
                _replay_buffer.append(state.copy())
                if len(_replay_buffer) > _REPLAY_MAX:
                    _replay_buffer.pop(0)
                if terminated or truncated:
                    add_metrics({
                        "episode": episode,
                        "reward": episode_reward,
                        "steps": info.get("step", 0),
                        "success_rate": info.get("success_rate", 0),
                        "avg_latency": info.get("avg_latency", 0),
                        "avg_cost": info.get("avg_cost", 0),
                    })
                    obs, info = environment.reset()
                    episode_reward = 0.0
                    episode += 1
        else:
            for step in range(total_steps):
                if not _training_active:
                    break
                action = agent.predict(obs)
                obs, reward, terminated, truncated, info = environment.step(action)
                episode_reward += reward
                state = info.copy()
                state["episode"] = episode
                state["episode_reward"] = episode_reward
                state["global_step"] = step
                state["agent_type"] = agent_type
                state["current_action"] = action
                state["last_reward"] = reward
                _append_last_event(state, action, reward, info)
                update_state(state)
                _replay_buffer.append(state.copy())
                if len(_replay_buffer) > _REPLAY_MAX:
                    _replay_buffer.pop(0)
                if terminated or truncated:
                    add_metrics({
                        "episode": episode,
                        "reward": episode_reward,
                        "steps": info.get("step", 0),
                        "success_rate": info.get("success_rate", 0),
                        "avg_latency": info.get("avg_latency", 0),
                        "avg_cost": info.get("avg_cost", 0),
                    })
                    obs, info = environment.reset()
                    episode_reward = 0.0
                    episode += 1

    environment.close()


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the dashboard server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
