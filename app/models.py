from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False, unique=True, index=True)

    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable= False) 