"""SQLite-backed persistence of every prediction.

Powers the 'Recent predictions' panel in the Streamlit app and lets the
CLI show how the system has been used historically. Single table; no
migrations needed for the assignment.
"""
from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session

from src.config import DB_PATH
from src.inference import Prediction


class Base(DeclarativeBase):
    pass


class PredictionRow(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime, default=_dt.datetime.utcnow, nullable=False)
    model_name = Column(String(64), nullable=False)
    message_preview = Column(String(280), nullable=False)
    diff_preview = Column(String(280), nullable=False)
    predicted_label = Column(String(32), nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON, nullable=False)
    source = Column(String(32), default="gui")


def _engine():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)


def init() -> None:
    Base.metadata.create_all(_engine())


def log_prediction(
    message: str,
    diff: str,
    prediction: Prediction,
    source: str = "gui",
) -> None:
    init()
    row = PredictionRow(
        model_name=prediction.model,
        message_preview=(message or "")[:280],
        diff_preview=(diff or "")[:280],
        predicted_label=prediction.label,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        source=source,
    )
    with Session(_engine()) as session:
        session.add(row)
        session.commit()


def list_recent(limit: int = 25) -> List[Dict]:
    init()
    with Session(_engine()) as session:
        rows = session.execute(
            select(PredictionRow).order_by(PredictionRow.id.desc()).limit(limit)
        ).scalars().all()
        return [
            {
                "id": r.id,
                "ts": r.ts.isoformat(),
                "model": r.model_name,
                "label": r.predicted_label,
                "confidence": r.confidence,
                "message": r.message_preview,
                "source": r.source,
            }
            for r in rows
        ]


def label_distribution() -> Dict[str, int]:
    init()
    with Session(_engine()) as session:
        rows = session.execute(select(PredictionRow.predicted_label)).all()
        out: Dict[str, int] = {}
        for (label,) in rows:
            out[label] = out.get(label, 0) + 1
        return out
