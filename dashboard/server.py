"""
FastAPI dashboard server with WebSocket streaming.

Endpoints:
    GET  /                      - Serves the live dashboard
    WS   /ws/state              - Streams live environment state
    POST /api/start             - Starts training in background
    POST /api/stop              - Stops training
    GET  /api/status            - Returns current training status
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


def update_state(state: dict) -> None:
    """Called by the training loop to push new state."""
    global _current_state
    with _state_lock:
        _current_state = state.copy()


def get_state() -> dict:
    with _state_lock:
        return _current_state.copy()


def add_metrics(metrics: dict) -> None:
    global _training_metrics
    _training_metrics.append(metrics)
    if len(_training_metrics) > 10_000:
        _training_metrics = _training_metrics[-5000:]


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard HTML."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>Dashboard — static/index.html not found</h1>")


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
):
    """Start training in a background thread."""
    global _training_active, _training_thread

    if _training_active:
        return {"status": "already_running"}

    _training_active = True

    def _run():
        global _training_active
        try:
            _run_training(agent_type, steps)
        finally:
            _training_active = False

    _training_thread = threading.Thread(target=_run, daemon=True)
    _training_thread.start()
    return {"status": "started", "agent": agent_type, "steps": steps}


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


def _run_training(agent_type: str, total_steps: int):
    """Run training in a thread, pushing state updates."""
    import gymnasium as gym
    import numpy as np
    import env as _env_reg  # noqa

    environment = gym.make("MicroserviceRouting-v0")

    if agent_type == "ppo":
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(device="cpu")
        # For dashboard mode, we manually step to stream state
        obs, info = environment.reset()
        update_state(info)
        episode = 0
        episode_reward = 0.0

        for step in range(total_steps):
            if not _training_active:
                break
            action = agent.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = environment.step(action)
            episode_reward += reward

            # Stream state
            state = info.copy()
            state["episode"] = episode
            state["episode_reward"] = episode_reward
            state["global_step"] = step
            state["agent_type"] = agent_type
            update_state(state)

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
        # Run with baseline or IMPALA in eval mode
        from agents.baselines import BASELINES
        agent_cls = BASELINES.get(agent_type)
        if agent_cls:
            agent = agent_cls()
        else:
            return

        obs, info = environment.reset()
        update_state(info)
        episode = 0
        episode_reward = 0.0

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
            update_state(state)

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
