from __future__ import annotations

import logging
import json
import threading
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from string import ascii_uppercase
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "Models"
FRONTEND_DIR = ROOT / "frontend"
LOGGER = logging.getLogger("sign_to_text")
MAX_FRAME_BYTES = 2_000_000
SESSION_IDLE_SECONDS = 3 * 60 * 60
WORD_PAUSE_SECONDS = 3.0
BLANK_RELEASE_SECONDS = 0.12
FAST_ACCEPT_CONFIDENCE = 0.70


class ActionRequest(BaseModel):
    action: Literal["space", "commit", "backspace", "clear"]


class Recognizer:
    """Loads the project's four CNNs and applies its original image pipeline."""

    def __init__(self) -> None:
        # Delay TensorFlow import until startup so health and static pages remain
        # available if a model cannot be loaded in a particular environment.
        from tensorflow.keras import layers
        from tensorflow.keras.models import Sequential

        self._predict_lock = threading.Lock()
        self.main = self._load(Sequential, layers, "model_new")
        self.dru = self._load(Sequential, layers, "model-bw_dru")
        self.tkdi = self._load(Sequential, layers, "model-bw_tkdi")
        self.smn = self._load(Sequential, layers, "model-bw_smn")

    @staticmethod
    def _load(model_class, keras_layers, name: str):
        architecture = json.loads((MODEL_DIR / f"{name}.json").read_text(encoding="utf-8"))
        layer_configs = architecture["config"]["layers"]
        input_shape = None
        has_input_layer = any(layer["class_name"] == "InputLayer" for layer in layer_configs)
        for layer in layer_configs:
            config = layer["config"]
            legacy_shape = config.pop("batch_input_shape", None)
            if layer["class_name"] == "InputLayer" and legacy_shape:
                config["batch_shape"] = legacy_shape
            elif legacy_shape and input_shape is None:
                input_shape = legacy_shape
        if not has_input_layer and input_shape:
            layer_configs.insert(0, {
                "class_name": "InputLayer",
                "config": {
                    "batch_shape": input_shape,
                    "dtype": "float32",
                    "sparse": False,
                    "ragged": False,
                    "name": f"{name.replace('-', '_')}_input",
                },
            })
        # The saved JSON files were created with Keras 2.4. Translate its
        # batch_input_shape field so the same architectures load under Keras 3.
        model_config = architecture["config"]
        model = model_class(name=model_config.get("name", name))
        for layer_config in layer_configs:
            kind = layer_config["class_name"]
            config = layer_config["config"]
            layer_name = config.get("name")
            if kind == "InputLayer":
                model.add(keras_layers.InputLayer(
                    batch_shape=config.get("batch_shape", [None, 128, 128, 1]),
                    dtype=config.get("dtype", "float32"),
                    name=layer_name,
                ))
            elif kind == "Conv2D":
                model.add(keras_layers.Conv2D(
                    filters=config["filters"],
                    kernel_size=config["kernel_size"],
                    strides=config.get("strides", (1, 1)),
                    padding=config.get("padding", "valid"),
                    data_format=config.get("data_format", "channels_last"),
                    dilation_rate=config.get("dilation_rate", (1, 1)),
                    groups=config.get("groups", 1),
                    activation=config.get("activation"),
                    use_bias=config.get("use_bias", True),
                    name=layer_name,
                ))
            elif kind == "MaxPooling2D":
                model.add(keras_layers.MaxPooling2D(
                    pool_size=config.get("pool_size", (2, 2)),
                    strides=config.get("strides"),
                    padding=config.get("padding", "valid"),
                    data_format=config.get("data_format", "channels_last"),
                    name=layer_name,
                ))
            elif kind == "Flatten":
                model.add(keras_layers.Flatten(
                    data_format=config.get("data_format", "channels_last"),
                    name=layer_name,
                ))
            elif kind == "Dense":
                model.add(keras_layers.Dense(
                    units=config["units"],
                    activation=config.get("activation"),
                    use_bias=config.get("use_bias", True),
                    name=layer_name,
                ))
            elif kind == "Dropout":
                model.add(keras_layers.Dropout(rate=config["rate"], name=layer_name))
            else:
                raise ValueError(f"Unsupported saved model layer: {kind}")
        model.load_weights(str(MODEL_DIR / f"{name}.h5"))
        return model

    def predict(self, image_bytes: bytes) -> tuple[str, float]:
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("The uploaded frame is not a supported image.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        adaptive = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        _, processed = cv2.threshold(
            adaptive, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        batch = cv2.resize(processed, (128, 128)).astype(np.float32).reshape(1, 128, 128, 1)

        with self._predict_lock:
            primary = self._call(self.main, batch)
            index = int(np.argmax(primary))
            label = "blank" if index == 0 else ascii_uppercase[index - 1]
            confidence = float(primary[index])

            # Apply the original second-layer classifiers for visually similar letters.
            if label in "DRU":
                specialist = self._call(self.dru, batch)
                index = int(np.argmax(specialist))
                label, confidence = "DRU"[index], float(specialist[index])
            elif label in "DIKT":
                specialist = self._call(self.tkdi, batch)
                index = int(np.argmax(specialist))
                label, confidence = "DIKT"[index], float(specialist[index])
            elif label in "MNS":
                specialist = self._call(self.smn, batch)
                index = int(np.argmax(specialist))
                label, confidence = "MNS"[index], float(specialist[index])
        return label, confidence

    @staticmethod
    def _call(model, batch: np.ndarray) -> np.ndarray:
        result = model(batch, training=False)
        if hasattr(result, "numpy"):
            result = result.numpy()
        return np.asarray(result)[0]


@dataclass
class TranscriptionSession:
    current_word: str = ""
    words: list[str] = field(default_factory=list)
    history: deque[tuple[str, float]] = field(default_factory=lambda: deque(maxlen=3))
    candidate: str = ""
    candidate_since: float = field(default_factory=time.monotonic)
    armed: bool = True
    last_accepted_label: str = ""
    last_letter_at: float = 0.0
    last_seen: float = field(default_factory=time.monotonic)
    last_event: str = ""
    last_event_at: float = 0.0

    @property
    def transcript(self) -> str:
        committed = " ".join(self.words)
        if committed and self.current_word:
            return f"{committed} {self.current_word}"
        return committed or self.current_word

    def snapshot(
        self, label: str, confidence: float, stable: str, letter_accepted: bool = False
    ) -> dict:
        completed_sentence = " ".join(self.words)
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "stable_label": stable,
            "letter_accepted": letter_accepted,
            "awaiting_release": bool(
                not self.armed and self.candidate == self.last_accepted_label
            ),
            "current_word": self.current_word,
            # Keep the main sentence field live so clients do not wait for the
            # auto-commit timer before showing the word currently being spelled.
            "sentence": self.transcript,
            "completed_sentence": completed_sentence,
            "transcript": self.transcript,
            "last_event": self.last_event,
            "last_event_at": self.last_event_at,
        }

    def update(self, label: str, confidence: float) -> tuple[str, bool]:
        now = time.monotonic()
        self.last_seen = now
        observed = label if label == "blank" or confidence >= 0.55 else "blank"
        self.history.append((observed, confidence))
        counts = Counter(item[0] for item in self.history)
        stable = counts.most_common(1)[0][0]
        if counts[stable] < 2:
            stable = ""
        # High-confidence changes can skip the extra majority-frame wait.
        # The accepted-label gate below still prevents duplicate letters.
        if (
            label != "blank"
            and confidence >= FAST_ACCEPT_CONFIDENCE
            and (self.armed or label != self.last_accepted_label)
        ):
            stable = label

        if stable and stable != self.candidate:
            self.candidate = stable
            self.candidate_since = now

        held_for = now - self.candidate_since
        letter_accepted = False
        if stable == "blank" and held_for >= BLANK_RELEASE_SECONDS:
            self.armed = True
            self.last_accepted_label = ""
        elif (
            stable
            and stable != "blank"
            and (self.armed or stable != self.last_accepted_label)
        ):
            self.current_word += stable
            self.last_letter_at = now
            self.last_event = stable
            self.last_event_at = now
            self.armed = False
            self.last_accepted_label = stable
            letter_accepted = True

        # A quiet pause closes the current fingerspelled word automatically.
        if self.current_word and now - self.last_letter_at >= WORD_PAUSE_SECONDS:
            self.commit_word()
        return stable, letter_accepted

    def commit_word(self) -> None:
        if self.current_word:
            self.words.append(self.current_word)
            self.last_event = f"Word completed: {self.current_word}"
            self.last_event_at = time.monotonic()
            self.current_word = ""

    def apply_action(self, action: str) -> None:
        self.last_seen = time.monotonic()
        if action in ("space", "commit"):
            self.commit_word()
        elif action == "backspace":
            if self.current_word:
                self.current_word = self.current_word[:-1]
            elif self.words:
                self.current_word = self.words.pop()[:-1]
            self.last_event = "Backspace"
            self.last_event_at = time.monotonic()
        elif action == "clear":
            self.current_word = ""
            self.words.clear()
            self.history.clear()
            self.armed = self.candidate in ("", "blank")
            self.last_accepted_label = "" if self.armed else self.last_accepted_label
            self.last_event = "Transcript cleared"
            self.last_event_at = time.monotonic()


sessions: dict[str, TranscriptionSession] = {}
sessions_lock = threading.Lock()
recognizer: Recognizer | None = None
recognizer_error: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global recognizer, recognizer_error
    try:
        recognizer = Recognizer()
        recognizer_error = None
        LOGGER.info("Loaded ASL fingerspelling models from %s", MODEL_DIR)
    except Exception as exc:
        recognizer = None
        recognizer_error = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("Could not load the sign recognition models")
    yield


app = FastAPI(
    title="Sign Language to Text",
    description="Live ASL fingerspelling recognition and transcription.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/styles.css", include_in_schema=False)
async def stylesheet() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
async def browser_script() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "app.js", media_type="text/javascript")


@app.get("/api/health")
async def health() -> dict:
    return {
        "ready": recognizer is not None,
        "recognition": "ASL fingerspelling: A-Z and blank",
        "model_error": recognizer_error,
    }


def get_session(session_id: str) -> TranscriptionSession:
    now = time.monotonic()
    with sessions_lock:
        expired = [key for key, value in sessions.items() if now - value.last_seen > SESSION_IDLE_SECONDS]
        for key in expired:
            sessions.pop(key, None)
        if session_id not in sessions:
            if len(sessions) >= 1000:
                oldest = min(sessions, key=lambda key: sessions[key].last_seen)
                sessions.pop(oldest, None)
            sessions[session_id] = TranscriptionSession()
        return sessions[session_id]


@app.post("/api/sessions/{session_id}/frame")
async def predict_frame(session_id: str, request: Request) -> dict:
    if recognizer is None:
        raise HTTPException(
            status_code=503,
            detail=f"Recognition model is not ready. {recognizer_error or ''}".strip(),
        )
    image_bytes = await request.body()
    if not image_bytes or len(image_bytes) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="Frame is empty or too large.")
    try:
        label, confidence = await run_in_threadpool(recognizer.predict, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session = get_session(session_id)
    stable, letter_accepted = session.update(label, confidence)
    return session.snapshot(label, confidence, stable, letter_accepted)


@app.post("/api/sessions/{session_id}/action")
async def session_action(session_id: str, body: ActionRequest) -> dict:
    session = get_session(session_id)
    session.apply_action(body.action)
    return session.snapshot("", 0.0, session.candidate)
