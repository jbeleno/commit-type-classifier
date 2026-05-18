"""SQLite-backed persistence (3-table schema).

Schema (see docs/diagrams/puml/02_er_diagram.puml):

    models       — registry of every trained model (name, version, headline metrics)
    predictions  — every single inference, FK to models, optional FK to batch_runs
    batch_runs   — header row for "scan repository" jobs (path, N, model)

Powers the History tab in the Streamlit app and the `history` command in
the CLI. Auto-creates a `models` row on first reference so existing
flows ("log this prediction with model='baseline_tfidf'") keep working.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship

from src.config import DB_PATH
from src.inference import Prediction


class Base(DeclarativeBase):
    pass


class ModelRow(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), nullable=False, unique=True, index=True)
    version = Column(String(32), default="0.1.0")
    trained_at = Column(DateTime, default=_dt.datetime.utcnow)
    accuracy_test = Column(Float, nullable=True)
    macro_f1_test = Column(Float, nullable=True)
    notes = Column(Text, default="")

    predictions = relationship("PredictionRow", back_populates="model")
    batch_runs = relationship("BatchRunRow", back_populates="model")


class BatchRunRow(Base):
    __tablename__ = "batch_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime, default=_dt.datetime.utcnow, nullable=False)
    repo_path = Column(String(512), nullable=False)
    n_commits = Column(Integer, nullable=False)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)

    model = relationship("ModelRow", back_populates="batch_runs")
    predictions = relationship("PredictionRow", back_populates="batch_run")


class PredictionRow(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime, default=_dt.datetime.utcnow, nullable=False)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False, index=True)
    batch_run_id = Column(Integer, ForeignKey("batch_runs.id"), nullable=True, index=True)
    message_preview = Column(String(280), nullable=False)
    diff_preview = Column(String(280), nullable=False)
    predicted_label = Column(String(32), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON, nullable=False)
    source = Column(String(32), default="gui")

    model = relationship("ModelRow", back_populates="predictions")
    batch_run = relationship("BatchRunRow", back_populates="predictions")


def _engine():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)


def init() -> None:
    Base.metadata.create_all(_engine())


def _get_or_create_model(session: Session, model_name: str) -> ModelRow:
    row = session.execute(
        select(ModelRow).where(ModelRow.name == model_name)
    ).scalar_one_or_none()
    if row is not None:
        return row
    row = ModelRow(name=model_name)
    session.add(row)
    session.flush()
    return row


def log_prediction(
    message: str,
    diff: str,
    prediction: Prediction,
    source: str = "gui",
    batch_run_id: Optional[int] = None,
) -> int:
    init()
    with Session(_engine()) as session:
        model_row = _get_or_create_model(session, prediction.model)
        row = PredictionRow(
            model_id=model_row.id,
            batch_run_id=batch_run_id,
            message_preview=(message or "")[:280],
            diff_preview=(diff or "")[:280],
            predicted_label=prediction.label,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            source=source,
        )
        session.add(row)
        session.commit()
        return row.id


def open_batch_run(repo_path: str, n_commits: int, model_name: str) -> int:
    init()
    with Session(_engine()) as session:
        model_row = _get_or_create_model(session, model_name)
        run = BatchRunRow(repo_path=repo_path, n_commits=n_commits, model_id=model_row.id)
        session.add(run)
        session.commit()
        return run.id


def list_recent(limit: int = 25) -> List[Dict]:
    init()
    with Session(_engine()) as session:
        rows = session.execute(
            select(PredictionRow, ModelRow)
            .join(ModelRow, PredictionRow.model_id == ModelRow.id)
            .order_by(PredictionRow.id.desc())
            .limit(limit)
        ).all()
        return [
            {
                "id": p.id,
                "ts": p.ts.isoformat(timespec="seconds"),
                "model": m.name,
                "label": p.predicted_label,
                "confidence": p.confidence,
                "message": p.message_preview,
                "source": p.source,
            }
            for (p, m) in rows
        ]


def list_batch_runs(limit: int = 25) -> List[Dict]:
    init()
    with Session(_engine()) as session:
        rows = session.execute(
            select(BatchRunRow, ModelRow)
            .join(ModelRow, BatchRunRow.model_id == ModelRow.id)
            .order_by(BatchRunRow.id.desc())
            .limit(limit)
        ).all()
        return [
            {
                "id": b.id,
                "ts": b.ts.isoformat(timespec="seconds"),
                "repo": b.repo_path,
                "n_commits": b.n_commits,
                "model": m.name,
            }
            for (b, m) in rows
        ]


def label_distribution() -> Dict[str, int]:
    init()
    with Session(_engine()) as session:
        rows = session.execute(select(PredictionRow.predicted_label)).all()
        out: Dict[str, int] = {}
        for (label,) in rows:
            out[label] = out.get(label, 0) + 1
        return out


def register_model_metrics(
    name: str,
    accuracy_test: float | None = None,
    macro_f1_test: float | None = None,
    version: str | None = None,
) -> None:
    """Upsert the models row with eval metrics. Called from evaluate scripts."""
    init()
    with Session(_engine()) as session:
        row = _get_or_create_model(session, name)
        if version is not None:
            row.version = version
        if accuracy_test is not None:
            row.accuracy_test = accuracy_test
        if macro_f1_test is not None:
            row.macro_f1_test = macro_f1_test
        row.trained_at = _dt.datetime.utcnow()
        session.commit()
