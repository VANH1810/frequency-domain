from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    cam_id = Column(String)
    pid = Column(String)
    action = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    img_path = Column(Text) 
    link_video = Column(Text)
    seen = Column(Boolean, default=False)
