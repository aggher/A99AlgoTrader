"""
scheduler.py — APScheduler pipeline: collect → predict → alert every 60s.
Also schedules weekly model retraining.
"""
from __future__ import annotations
import logging, os
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron   import CronTrigger
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

from backend.database   import init_db
from backend.data_collector   import collect_all
from backend.prediction_engine import run_all
from backend.alert_service    import process_signals_sync

scheduler = BlockingScheduler(timezone="UTC")


def pipeline_tick():
    log.info("[PIPELINE] Tick ...")
    try:
        collect_all(initial=False)
        signals = run_all()
        actionable = [s for s in signals if s.signal != "HOLD"]
        if actionable:
            process_signals_sync(actionable)
        log.info("[PIPELINE] Done — %d signal(s), %d actionable", len(signals), len(actionable))
    except Exception as e:
        log.exception("Pipeline error: %s", e)


def retrain():
    log.info("[RETRAIN] Weekly retrain ...")
    try:
        from backend.institutional_train import institutional_retrain
        institutional_retrain()
    except Exception as e:
        log.exception("Retrain error: %s", e)


def run():
    init_db()
    pipeline_tick()   # immediate first run

    scheduler.add_job(pipeline_tick, IntervalTrigger(seconds=60),
                      id="pipeline", replace_existing=True)
    scheduler.add_job(retrain,       CronTrigger(day_of_week="sun", hour=2),
                      id="retrain",  replace_existing=True)

    log.info("Scheduler running (60s pipeline, Sunday 02:00 UTC retrain).")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Stopped.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
