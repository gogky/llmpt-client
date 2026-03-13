"""
Snapshot-safe Python representations of libtorrent alerts.

Raw libtorrent alert objects are only valid until the next ``pop_alerts()``
call.  The helpers in this module convert those short-lived native objects
into plain Python data before they leave the global alert-dispatch path.
"""

from dataclasses import dataclass
import logging


@dataclass(frozen=True, slots=True)
class ResumeDataReadyEvent:
    """Serialized fastresume payload ready to be written to disk."""

    resume_data: bytes


@dataclass(frozen=True, slots=True)
class AlertLogEvent:
    """Snapshot of an alert that only needs to emit a log message later."""

    log_level: int
    prefix: str
    message: str


AlertEvent = ResumeDataReadyEvent | AlertLogEvent


def _matches_alert_type(alert: object, lt_module: object, attr_name: str) -> bool:
    alert_type = getattr(lt_module, attr_name, None)
    if alert_type is None:
        return False
    try:
        return isinstance(alert, alert_type)
    except TypeError:
        return False


def _serialize_resume_data(alert: object, lt_module: object) -> bytes:
    if hasattr(lt_module, "write_resume_data_buf"):
        return lt_module.write_resume_data_buf(alert.params)
    return lt_module.bencode(alert.params)


def _alert_message(alert: object) -> str:
    try:
        return str(alert.message())
    except Exception:
        return repr(alert)


def snapshot_alert(alert: object, lt_module: object) -> AlertEvent | None:
    """Convert a raw libtorrent alert into a plain Python event.

    Unknown or intentionally ignored alert types return ``None``.
    """
    if _matches_alert_type(alert, lt_module, "save_resume_data_alert"):
        return ResumeDataReadyEvent(
            resume_data=_serialize_resume_data(alert, lt_module)
        )

    if _matches_alert_type(alert, lt_module, "save_resume_data_failed_alert"):
        return AlertLogEvent(
            log_level=logging.DEBUG,
            prefix="Save resume data failed",
            message=_alert_message(alert),
        )

    if _matches_alert_type(alert, lt_module, "peer_error_alert"):
        return AlertLogEvent(
            log_level=logging.WARNING,
            prefix="PEER ERROR",
            message=_alert_message(alert),
        )

    if _matches_alert_type(alert, lt_module, "peer_disconnected_alert"):
        return AlertLogEvent(
            log_level=logging.WARNING,
            prefix="PEER DISCONNECTED",
            message=_alert_message(alert),
        )

    if _matches_alert_type(alert, lt_module, "torrent_error_alert"):
        return AlertLogEvent(
            log_level=logging.WARNING,
            prefix="TORRENT ERROR",
            message=_alert_message(alert),
        )

    if _matches_alert_type(alert, lt_module, "hash_failed_alert"):
        return AlertLogEvent(
            log_level=logging.WARNING,
            prefix="HASH FAILED",
            message=_alert_message(alert),
        )

    if _matches_alert_type(alert, lt_module, "file_error_alert"):
        return AlertLogEvent(
            log_level=logging.WARNING,
            prefix="FILE ERROR",
            message=_alert_message(alert),
        )

    return None
