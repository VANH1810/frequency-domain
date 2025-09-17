from fastapi import APIRouter, Query, Path, HTTPException
from fastapi import APIRouter, Query
from modules.db.session import SessionLocal
from modules.db.models import Alert
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from sqlalchemy import func

router = APIRouter()

class AlertOut(BaseModel):
    id: int
    cam_id: str
    pid: str
    action: str
    timestamp: datetime
    img_path: str
    link_video: str
    seen: bool 

    class Config:
        orm_mode = True
        
@router.post("/alerts/{alert_id}/seen")
def mark_alert_seen(alert_id: int = Path(...)):
    session = SessionLocal()
    try:
        alert = session.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert.seen = True
        session.commit()
        return {"message": f"Alert {alert_id} marked as seen."}
    finally:
        session.close()


@router.get("/alerts", response_model=List[AlertOut])
def get_alerts(limit: int = 50, offset: int = 0):
    session = SessionLocal()
    try:
        alerts = (
            session.query(Alert)
            .order_by(Alert.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return alerts  
    finally:
        session.close()

@router.get("/alerts/count")
def count_alerts(seen: Optional[bool] = None):
    session = SessionLocal()
    try:
        query = session.query(func.count(Alert.id))
        if seen is not None:
            query = query.filter(Alert.seen == seen)
        count = query.scalar()
        return {"count": count}
    finally:
        session.close()