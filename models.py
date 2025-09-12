from sqlmodel import SQLModel, Field, Relationship
from enum import Enum
from datetime import datetime,date
from typing import Optional, List, Literal

# --- User Model ---

class Role(Enum):
    USER = "USER"
    ADMIN = "ADMIN"
    GUEST = "GUEST"


class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True) # This will be the email
    password: str = Field(max_length=60, index=True)
    role: Role = Field(default=Role.USER, index=True)
    
    # Profile Fields
    first_name: Optional[str] = Field(default=None)
    last_name: Optional[str] = Field(default=None)
    company: Optional[str] = Field(default=None)
    job_title: Optional[str] = Field(default=None)
    avatar_url: Optional[str] = Field(default=None) # <-- Add avatar URL field

    meetings: List["Meeting"] = Relationship(back_populates="user")

class CreateUser(SQLModel):
    username: str
    password: str
    full_name: str

class UserUpdate(SQLModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    
class UserProfile(SQLModel):
    id: int
    username: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    avatar_url: Optional[str] = None # <-- Add avatar URL field

# --- JWT Token Model ---

class Token(SQLModel):
    access_token: str
    token_type: Literal["bearer"]

# Payload
class TokenData(SQLModel):
    username: str
    exp: int
    role: Role | None = None
    sub: str | None = None


class TaskStatus(str, Enum):
    PENDING = "Pending"
    PROCESSING = "Processing"
    COMPLETED = "Completed"
    ERROR = "Error"

class ProcessingTask(SQLModel, table=True):
    id: str = Field(primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    original_filename: str
    meeting_id: Optional[int] = Field(default=None, foreign_key="meeting.id")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    details: str = Field(default="Task has been created.")
    progress_percent: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Meeting(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    task_id: str = Field(foreign_key="processingtask.id", index=True)
    title: str
    transcript: str
    notes: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    user: Optional[User] = Relationship(back_populates="meetings")