from sqlmodel import SQLModel, Field, Relationship
from enum import Enum
from datetime import datetime
from typing import Optional, List, Literal
from pydantic import EmailStr

# --- User Model ---

class Role(Enum):
    USER = "USER"
    ADMIN = "ADMIN"
    GUEST = "GUEST"


class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True) # This will be the email
    password: str
    role: Role = Field(default=Role.USER)
    
    # Profile Fields
    first_name: Optional[str] = Field(default=None)
    last_name: Optional[str] = Field(default=None)
    company: Optional[str] = Field(default=None)
    job_title: Optional[str] = Field(default=None)
    avatar_url: Optional[str] = Field(default=None)

    # --- NEW: Fields for Verification and Password Reset ---
    is_verified: bool = Field(default=False)
    verification_code: Optional[str] = Field(default=None)
    reset_token: Optional[str] = Field(default=None, index=True)
    reset_token_expires: Optional[datetime] = Field(default=None)

    meetings: List["Meeting"] = Relationship(back_populates="user")

class CreateUser(SQLModel):
    username: EmailStr
    password: str
    full_name: str

class UserUpdate(SQLModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    
class UserProfile(SQLModel):
    id: int
    username: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    avatar_url: Optional[str] = None

# --- NEW: Request Body Models for Auth ---
class VerificationRequest(SQLModel):
    email: EmailStr
    code: str

class ResetPasswordRequest(SQLModel):
    token: str
    password: str

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
    original_language: Optional[str] = Field(default=None)
    transcript_en: Optional[str] = Field(default=None)

    user: Optional[User] = Relationship(back_populates="meetings")

