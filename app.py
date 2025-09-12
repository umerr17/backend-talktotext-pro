# main.py
import os
import shutil
import uuid
import time
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlmodel import Session, select, col, func

import assemblyai as aai
import google.generativeai as genai
import cloudinary
import cloudinary.uploader


from database import create_db_and_tables, engine, SessionDependency
from models import User, CreateUser, Token, TokenData, Meeting, Role, ProcessingTask, TaskStatus, UserUpdate, UserProfile

load_dotenv()

# -------- CONFIG --------
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", "60"))
MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 1 GB

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Cloudinary Config
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

if not all([SECRET_KEY, ASSEMBLYAI_API_KEY, GOOGLE_API_KEY, CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET]):
    raise EnvironmentError("Missing required environment variables")

# -------- Initialize clients / settings --------
aai.settings.api_key = ASSEMBLYAI_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

# -------- Password hashing & OAuth2 --------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# -------- App bootstrap --------
app = FastAPI(title="TalkToText Pro - Backend")
create_db_and_tables()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Pydantic Models --------
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

# -------- Utility functions --------
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def find_user_by_username(username: str, session: Session) -> User | None:
    statement = select(User).where(User.username == username)
    return session.exec(statement).first()

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": int(expire.timestamp())})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Dependency: decode token and return current user
async def get_current_user(session: SessionDependency, token: Annotated[str, Depends(oauth2_scheme)]) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        token_data = TokenData.model_validate(payload)
        username = token_data.username
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = find_user_by_username(username, session)
    if not user:
        raise credentials_exception
    return user

CurrentUser = Annotated[User, Depends(get_current_user)]


# -------- Auth endpoints --------
@app.post("/users/", response_model=User)
async def register_user(new_user: CreateUser, session: SessionDependency):
    existing = find_user_by_username(new_user.username, session)
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed = get_password_hash(new_user.password)
    
    first_name = ""
    last_name = ""
    if new_user.full_name:
        parts = new_user.full_name.strip().split(" ", 1)
        first_name = parts[0]
        if len(parts) > 1:
            last_name = parts[1]

    user = User(
        username=new_user.username, 
        password=hashed,
        first_name=first_name,
        last_name=last_name
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user

@app.post("/login", response_model=Token)
async def login_for_access_token(session: SessionDependency, form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = find_user_by_username(form_data.username, session)
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"username": user.username, "role": user.role.value},
        expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")

@app.post("/forgot-password", status_code=status.HTTP_200_OK)
async def forgot_password(request: ForgotPasswordRequest, session: SessionDependency):
    user = find_user_by_username(request.email, session)
    if not user:
        # To prevent user enumeration, we return a success response even if the user doesn't exist.
        return JSONResponse(content={"message": "If an account with that email exists, a password reset link has been sent."})
    
    # In a real application, you would generate a unique, expiring token,
    # save it to the database, and email it to the user.
    # For this simulation, we'll just confirm the action.
    print(f"Password reset requested for {user.username}")
    
    return JSONResponse(content={"message": "If an account with that email exists, a password reset link has been sent."})

# -------- Profile Endpoints --------
@app.get("/profile", response_model=UserProfile)
async def get_user_profile(current_user: CurrentUser):
    return current_user

@app.patch("/profile", response_model=UserProfile)
async def update_user_profile(
    session: SessionDependency,
    current_user: CurrentUser,
    user_update: UserUpdate
):
    user_data = user_update.model_dump(exclude_unset=True)
    for key, value in user_data.items():
        setattr(current_user, key, value)
    
    session.add(current_user)
    session.commit()
    session.refresh(current_user)
    return current_user

@app.post("/profile/avatar", response_model=UserProfile)
async def upload_avatar(
    session: SessionDependency,
    current_user: CurrentUser,
    file: UploadFile = File(...)
):
    try:
        # Upload to Cloudinary
        result = cloudinary.uploader.upload(
            file.file,
            folder="talktotext_pro_avatars",
            public_id=f"user_{current_user.id}",
            overwrite=True,
            resource_type="image"
        )
        avatar_url = result.get("secure_url")

        # Update user in DB
        current_user.avatar_url = avatar_url
        session.add(current_user)
        session.commit()
        session.refresh(current_user)
        
        return current_user
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not upload avatar: {e}")

@app.delete("/profile", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_account(session: SessionDependency, current_user: CurrentUser):
    # This is a destructive action.
    # First, delete associated meetings and tasks.
    
    # Get all tasks for the user
    task_statement = select(ProcessingTask).where(ProcessingTask.user_id == current_user.id)
    tasks = session.exec(task_statement).all()
    for task in tasks:
        session.delete(task)

    # Get all meetings for the user
    meeting_statement = select(Meeting).where(Meeting.user_id == current_user.id)
    meetings = session.exec(meeting_statement).all()
    for meeting in meetings:
        session.delete(meeting)
    
    # Finally, delete the user
    session.delete(current_user)
    session.commit()
    return

# -------- Background processing & transcription pipeline --------
def update_task_progress(task_id: str, status: TaskStatus, details: str, progress_percent: int):
    with Session(engine) as session:
        statement = select(ProcessingTask).where(ProcessingTask.id == task_id)
        task = session.exec(statement).first()
        if task:
            task.status = status
            task.details = details
            task.progress_percent = progress_percent
            session.add(task)
            session.commit()

# NEW: The core function now handles conversion
def process_audio_task(task_id: str, file_path: str, user_id: int, original_filename: str):
    file_to_process = file_path
    
    # Check if the file is not already an MP3, and convert it if necessary.
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() != ".mp3":
        audio_file_path = f"temp_audio_{task_id}.mp3"

        try:
            # Command to convert any video/audio file to MP3.
            # This is reliable as it re-encodes, unlike the `copy` method.
            update_task_progress(task_id, TaskStatus.PROCESSING, "Converting file to MP3...", 10)
            subprocess.run(
                ["ffmpeg", "-i", file_path, "-vn", "-acodec", "libmp3lame", "-q:a", "2", audio_file_path],
                check=True,
                capture_output=True,
                text=True
            )
            file_to_process = audio_file_path
        except subprocess.CalledProcessError as e:
            update_task_progress(task_id, TaskStatus.ERROR, f"Conversion failed: {e.stderr}", 0)
            # The process stops here if conversion fails.
            return
    
    try:
        # Step 1: Transcribe via AssemblyAI (now using the converted file)
        update_task_progress(task_id, TaskStatus.PROCESSING, "Transcribing audio (this may take a while)...", 20)
        
        config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.universal)
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(file_to_process)

        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"Transcription failed: {transcript.error}")

        transcript_text = transcript.text or "Could not transcribe audio."
        # Step 2: Summarize via Google Gemini
        update_task_progress(task_id, TaskStatus.PROCESSING, "Generating meeting notes with AI...", 70)
        
        full_prompt = (
            "You are an expert meeting notes assistant. Transform the following meeting transcript into structured meeting notes. "
            "Your output must be professional, clear, and the length of the response should be according to the transcription length. "
            "**Important**: Do not include any conversational preamble, introduction, or any text before the first section. "
            "Your response must begin directly with the '### Executive Summary' heading. "
            "The required sections, each with a '###' heading, are: "
            "### Executive Summary, ### Key Discussion Points, ### Decisions Made, ### Action Items, and ### Sentiment Analysis. "
            "For Sentiment Analysis, provide a single word (Positive, Neutral, or Negative) followed by a brief justification.\n\n"
            f"--- TRANSCRIPT ---\n{transcript_text}"
        )
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(full_prompt)
        notes_text = response.text or "Notes could not be generated."
        # Step 3: Save Meeting to DB
        update_task_progress(task_id, TaskStatus.PROCESSING, "Saving results...", 95)
        with Session(engine) as session:
            meeting_title = os.path.splitext(original_filename)[0].replace("_", " ").title()
            meeting = Meeting(
                user_id=user_id,
                title=meeting_title,
                transcript=transcript_text,
                notes=notes_text,
                task_id=task_id
            )
            session.add(meeting)
            session.commit()
            session.refresh(meeting)

            # Finalize task status
            statement = select(ProcessingTask).where(ProcessingTask.id == task_id)
            task = session.exec(statement).first()
            if task:
                task.status = TaskStatus.COMPLETED
                task.details = "Processing complete."
                task.progress_percent = 100
                task.meeting_id = meeting.id
                session.add(task)
                session.commit()

    except Exception as exc:
        update_task_progress(task_id, TaskStatus.ERROR, str(exc), 0)
    finally:
        # Clean up temporary files.
        # This will remove the original uploaded file AND the temporary converted file if one was created.
        if os.path.exists(file_path):
            os.remove(file_path)
        if file_to_process != file_path and os.path.exists(file_to_process):
            os.remove(file_to_process)

# -------- Core Application Endpoints -------- #
@app.post("/upload-audio")
async def upload_audio(
    request: Request,
    current_user: CurrentUser,
    session: SessionDependency,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    # Enforce file size limit
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)} MB."
        )

    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_extension = os.path.splitext(file.filename)[1]
    task_id = str(uuid.uuid4())
    filename = f"{task_id}{file_extension}"
    file_path = os.path.join(temp_dir, filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception:
        raise HTTPException(status_code=500, detail="Could not save uploaded file.")
        
    new_task = ProcessingTask(
        id=task_id,
        user_id=current_user.id,
        original_filename=file.filename,
        status=TaskStatus.PENDING,
        details="File uploaded, task queued.",
        progress_percent=1
    )
    session.add(new_task)
    session.commit()
    
    background_tasks.add_task(process_audio_task, task_id, file_path, current_user.id, file.filename)

    return {"task_id": task_id, "status": "Started"}

@app.get("/progress/{task_id}", response_model=ProcessingTask)
async def get_progress(task_id: str, session: SessionDependency, current_user: CurrentUser):
    statement = select(ProcessingTask).where(ProcessingTask.id == task_id)
    task = session.exec(statement).first()
    
    if not task or task.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Task not found")
        
    return task

@app.get("/tasks/ongoing", response_model=list[ProcessingTask])
async def get_ongoing_tasks(session: SessionDependency, current_user: CurrentUser):
    statement = select(ProcessingTask).where(
        ProcessingTask.user_id == current_user.id,
        col(ProcessingTask.status).in_([TaskStatus.PENDING, TaskStatus.PROCESSING])
    ).order_by(col(ProcessingTask.created_at).desc())
    tasks = session.exec(statement).all()
    return tasks

@app.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(task_id: str, session: SessionDependency, current_user: CurrentUser):
    statement = select(ProcessingTask).where(ProcessingTask.id == task_id, ProcessingTask.user_id == current_user.id)
    task = session.exec(statement).first()

    if not task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    if task.meeting_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete a completed task that has a meeting. Delete the meeting instead.")

    session.delete(task)
    session.commit()
    return

@app.get("/meetings", response_model=list[Meeting])
async def get_meetings(session: SessionDependency, current_user: CurrentUser):
    statement = select(Meeting).where(Meeting.user_id == current_user.id).order_by(col(Meeting.created_at).desc())
    meetings = session.exec(statement).all()
    return meetings

@app.get("/meetings/{meeting_id}", response_model=Meeting)
async def get_meeting_details(meeting_id: int, session: SessionDependency, current_user: CurrentUser):
    statement = select(Meeting).where(Meeting.id == meeting_id)
    meeting = session.exec(statement).first()

    if not meeting or meeting.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Meeting not found")

    return meeting

@app.delete("/meetings/{meeting_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_meeting(meeting_id: int, session: SessionDependency, current_user: CurrentUser):
    statement = select(Meeting).where(Meeting.id == meeting_id, Meeting.user_id == current_user.id)
    meeting = session.exec(statement).first()

    if not meeting:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting not found")

    # Also delete the associated task
    task_statement = select(ProcessingTask).where(ProcessingTask.id == meeting.task_id)
    task = session.exec(task_statement).first()
    if task:
        session.delete(task)
        
    session.delete(meeting)
    session.commit()
    return

@app.get("/dashboard/stats")
async def get_dashboard_stats(session: SessionDependency, current_user: CurrentUser):
    # Count total meetings
    meeting_statement = select(func.count(Meeting.id)).where(Meeting.user_id == current_user.id)
    total_meetings = session.exec(meeting_statement).one()

    # Note: Hours processed and accuracy are not tracked in the current model.
    # We will return assumed values for demonstration.
    hours_processed = total_meetings * 0.3 # Assuming avg 18 mins per meeting
    accuracy_rate = 98.5

    # Team members is 1 in this context
    team_members = 1 

    return {
        "total_meetings": total_meetings,
        "hours_processed": round(hours_processed, 1),
        "team_members": team_members,
        "accuracy_rate": accuracy_rate
    }