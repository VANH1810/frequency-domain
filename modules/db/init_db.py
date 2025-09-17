import os
from modules.db.session import SessionLocal, engine
from modules.db.models import Alert, Base
from datetime import datetime


def init_db():
    db_path = "alerts.db"  # hoặc extract từ session.DB_URL
    if not os.path.exists(db_path):
        print("[DB] Initializing database...")
        Base.metadata.create_all(bind=engine)
    else:
        print("[DB] Database already exists.")


def insert_alert(cam_id, alert_time, pid, action, img_path, link_video):
    session = SessionLocal()
    try:
        new_alert = Alert(
            cam_id=cam_id,
            pid=pid,
            action=action,
            timestamp=alert_time,
            img_path=img_path,
            link_video=link_video, 
            seen=False
        )
        session.add(new_alert)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Insert error: {e}")
    finally:
        session.close()