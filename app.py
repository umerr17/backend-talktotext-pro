# main.py
import os
import re
import shutil
import uuid
import time
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Annotated
import random
import string
from email.mime.text import MIMEText
import smtplib

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, RedirectResponse
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlmodel import Session, select, col, func
from sqlalchemy import text
from authlib.integrations.starlette_client import OAuth

import assemblyai as aai
import google.generativeai as genai
import cloudinary
import cloudinary.uploader


from database import create_db_and_tables, engine, SessionDependency
from models import User, CreateUser, Token, TokenData, Meeting, Role, ProcessingTask, TaskStatus, UserUpdate, UserProfile, VerificationRequest, ResetPasswordRequest

load_dotenv()

# -------- CONFIG --------
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", "60"))
MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 1 GB

# --- Environment-specific config ---
APP_ENV = os.getenv("APP_ENV", "development")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")


ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Google OAuth Config ---
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

# --- Email Config ---
SMTP_SERVER = os.getenv("MAIL_SERVER")
SMTP_PORT = int(os.getenv("MAIL_PORT", 587))
SMTP_USERNAME = os.getenv("MAIL_USERNAME")
SMTP_PASSWORD = os.getenv("MAIL_PASSWORD")
SENDER_EMAIL = os.getenv("MAIL_FROM")


# Cloudinary Config
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

if not all([SECRET_KEY, ASSEMBLYAI_API_KEY, GOOGLE_API_KEY, CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET]):
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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://talktotext-pro.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Environment-aware Session Middleware ---
if APP_ENV == "production":
    app.add_middleware(
        SessionMiddleware,
        secret_key=SECRET_KEY,
        https_only=True,
        same_site='none'
    )
else:  # development
    app.add_middleware(
        SessionMiddleware,
        secret_key=SECRET_KEY
    )


# --- OAuth Setup ---
oauth = OAuth()
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# -------- Pydantic Models --------
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

# Add this class definition near your other Pydantic models
class ShareRequest(BaseModel):
    recipient_email: EmailStr

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
    
def send_email(to_email: str, subject: str, body: str):
    if not all([SMTP_SERVER, SMTP_USERNAME, SMTP_PASSWORD, SENDER_EMAIL]):
        print("!!! SMTP settings not configured. Printing email to console instead. !!!")
        print(f"--- To: {to_email} ---")
        print(f"--- Subject: {subject} ---")
        print(f"--- Body ---\n{body}")
        print("--------------------")
        return

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
            print(f"Email sent successfully to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")


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
    if not user or not user.is_verified:
        raise credentials_exception
    return user

CurrentUser = Annotated[User, Depends(get_current_user)]


# -------- Auth endpoints --------
@app.post("/users/", response_model=User)
async def register_user(new_user: CreateUser, session: SessionDependency):
    existing = find_user_by_username(new_user.username, session)
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")
    
    hashed = get_password_hash(new_user.password)
    verification_code = ''.join(random.choices(string.digits, k=6))
    
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
        last_name=last_name,
        verification_code=verification_code
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    send_email(
        to_email=user.username,
        subject="Verify Your TalkToText Pro Account",
        body=f"Hi {user.first_name},\n\nYour verification code is: {verification_code}\n\nThanks,\nThe TalkToText Pro Team"
    )
    
    return user

@app.post("/verify-email", status_code=status.HTTP_200_OK)
async def verify_email(request: VerificationRequest, session: SessionDependency):
    user = find_user_by_username(request.email, session)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    
    if user.is_verified:
        return {"message": "Email is already verified."}

    if user.verification_code != request.code:
        raise HTTPException(status_code=400, detail="Invalid verification code.")
    
    user.is_verified = True
    user.verification_code = None
    session.add(user)
    session.commit()
    
    return {"message": "Email verified successfully."}


@app.post("/login", response_model=Token)
async def login_for_access_token(session: SessionDependency, form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = find_user_by_username(form_data.username, session)
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    if not user.is_verified:
        raise HTTPException(status_code=401, detail="Please verify your email before logging in.")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"username": user.username, "role": user.role.value},
        expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")

@app.get("/login/google")
async def login_google(request: Request):
    redirect_uri = request.url_for('auth_google')
    return await oauth.google.authorize_redirect(request, str(redirect_uri))

@app.get("/auth/google")
async def auth_google(request: Request, session: SessionDependency):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Could not authorize access token: {e}")
    
    user_info = token.get('userinfo')
    if not user_info:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not retrieve user info")

    email = user_info['email']
    user = find_user_by_username(email, session)

    if not user:
        user = User(
            username=email,
            password=get_password_hash(str(uuid.uuid4())),
            first_name=user_info.get('given_name'),
            last_name=user_info.get('family_name'),
            avatar_url=user_info.get('picture'),
            is_verified=True
        )
        session.add(user)
        session.commit()
        session.refresh(user)

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"username": user.username, "role": user.role.value},
        expires_delta=access_token_expires
    )
    
    return RedirectResponse(url=f"{FRONTEND_URL}/auth/callback?token={access_token}")


@app.post("/forgot-password", status_code=status.HTTP_200_OK)
async def forgot_password(request: ForgotPasswordRequest, session: SessionDependency):
    user = find_user_by_username(request.email, session)
    if user:
        reset_token = str(uuid.uuid4())
        user.reset_token = reset_token
        user.reset_token_expires = datetime.now(timezone.utc) + timedelta(hours=1)
        session.add(user)
        session.commit()
        
        reset_link = f"{FRONTEND_URL}/reset-password?token={reset_token}"
        send_email(
            to_email=user.username,
            subject="Password Reset Request for TalkToText Pro",
            body=f"Hi {user.first_name},\n\nPlease use the following link to reset your password:\n{reset_link}\n\nThis link will expire in 1 hour.\n\nThanks,\nThe TalkToText Pro Team"
        )
    return JSONResponse(content={"message": "If an account with that email exists, a password reset link has been sent."})


@app.post("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(request: ResetPasswordRequest, session: SessionDependency):
    statement = select(User).where(User.reset_token == request.token)
    user = session.exec(statement).first()

    if not user or user.reset_token_expires < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Invalid or expired password reset token.")
    
    user.password = get_password_hash(request.password)
    user.reset_token = None
    user.reset_token_expires = None
    session.add(user)
    session.commit()
    
    return {"message": "Password has been reset successfully."}


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
        result = cloudinary.uploader.upload(
            file.file,
            folder="talktotext_pro_avatars",
            public_id=f"user_{current_user.id}",
            overwrite=True,
            resource_type="image"
        )
        avatar_url = result.get("secure_url")

        current_user.avatar_url = avatar_url
        session.add(current_user)
        session.commit()
        session.refresh(current_user)
        
        return current_user
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not upload avatar: {e}")

@app.delete("/profile", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_account(session: SessionDependency, current_user: CurrentUser):
    task_statement = select(ProcessingTask).where(ProcessingTask.user_id == current_user.id)
    tasks = session.exec(task_statement).all()
    for task in tasks:
        session.delete(task)

    meeting_statement = select(Meeting).where(Meeting.user_id == current_user.id)
    meetings = session.exec(meeting_statement).all()
    for meeting in meetings:
        session.delete(meeting)
    
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

def process_audio_task(task_id: str, file_path: str, user_id: int, original_filename: str):
    # Keep track of all temporary files for easy cleanup
    cleanup_paths = [file_path]
    file_to_process = file_path
    temp_dir = "temp_uploads" # Define temp directory once
    
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() != ".mp3":
        # CHANGE 1: Save the converted file in the same temp directory
        audio_file_path = os.path.join(temp_dir, f"temp_audio_{task_id}.mp3")
        cleanup_paths.append(audio_file_path) # Add new file to the cleanup list

        try:
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
            # The finally block will still run and clean up all files in cleanup_paths
            return
    
    try:
        update_task_progress(task_id, TaskStatus.PROCESSING, "Transcribing audio (this may take a while)...", 20)
        
        config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.nano, language_detection=True)
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(file_to_process)

        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"Transcription failed: {transcript.error}")

        transcript_text = transcript.text or "Could not transcribe audio."
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
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(full_prompt)
        notes_text = response.text or "Notes could not be generated."
        update_task_progress(task_id, TaskStatus.PROCESSING, "Saving results...", 95)
        
        with Session(engine) as session:
            meeting_title = os.path.splitext(original_filename)[0].replace("_", " ").title()
            
            meeting = Meeting(
                user_id=user_id,
                title=meeting_title,
                transcript=original_transcript_text,
                transcript_en=english_transcript_text,
                original_language=language_code,
                notes=notes_text,
                task_id=task_id
            )
            session.add(meeting)
            session.commit()
            session.refresh(meeting)

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
        # CHANGE 2: Robust cleanup logic using the list of paths
        for path in cleanup_paths:
            if os.path.exists(path):
                os.remove(path)

# -------- Core Application Endpoints -------- #
@app.post("/upload-audio")
async def upload_audio(
    request: Request,
    current_user: CurrentUser,
    session: SessionDependency,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
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

    task_statement = select(ProcessingTask).where(ProcessingTask.id == meeting.task_id)
    task = session.exec(task_statement).first()
    if task:
        session.delete(task)
        
    session.delete(meeting)
    session.commit()
    return

@app.post("/meetings/{meeting_id}/share", status_code=status.HTTP_200_OK)
async def share_meeting_by_email(
    meeting_id: int,
    share_request: ShareRequest,
    session: SessionDependency,
    current_user: CurrentUser
):
    statement = select(Meeting).where(Meeting.id == meeting_id)
    meeting = session.exec(statement).first()

    # Verify the meeting exists and belongs to the user
    if not meeting or meeting.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Meeting not found")

    # Format the email content from the stored notes
    email_subject = f"Meeting Notes: {meeting.title}"

    # Simple text formatting for the email. You can enhance this with HTML later.
    # This removes the '###' markdown headers for a cleaner look.
    formatted_notes = re.sub(r'###\s*(.*?)\s*\n', r'\1\n\n', meeting.notes)
    email_body = f"""
Hello,

Please find the notes for the meeting "{meeting.title}", shared by {current_user.first_name or current_user.username}.

Date: {meeting.created_at.strftime('%Y-%m-%d')}
--------------------------------------------------

{formatted_notes}

--------------------------------------------------
Generated by TalkToText Pro.
    """

    # Use the existing email utility to send the message
    try:
        send_email(
            to_email=share_request.recipient_email,
            subject=email_subject,
            body=email_body.strip()
        )
    except Exception as e:
        # Handle potential SMTP errors
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

    return {"message": "Meeting notes sent successfully."}

@app.get("/dashboard/stats")
async def get_dashboard_stats(session: SessionDependency, current_user: CurrentUser):
    # Total Meetings
    meeting_statement = select(func.count(Meeting.id)).where(Meeting.user_id == current_user.id)
    total_meetings = session.exec(meeting_statement).one_or_none() or 0

    # Meetings in the last 7 days for the bar chart
    last_7_days = datetime.now(timezone.utc) - timedelta(days=7)
    meetings_over_time_statement = (
        select(
            func.date(Meeting.created_at).label("date"),
            func.count(Meeting.id).label("count")
        )
        .where(Meeting.user_id == current_user.id)
        .where(Meeting.created_at >= last_7_days)
        .group_by(func.date(Meeting.created_at))
        .order_by(func.date(Meeting.created_at))
    )
    meetings_over_time = session.exec(meetings_over_time_statement).all()
    
    # Create a dictionary of the last 7 days initialized to 0
    date_range = [(last_7_days + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(8)]
    meetings_data = {day: 0 for day in date_range}
    
    # Populate with actual data
    for row in meetings_over_time:
        meetings_data[row.date] = row.count

    # Format for the frontend
    meetings_chart_data = [{"date": date, "meetings": count} for date, count in meetings_data.items()]


    # Task status distribution for the pie chart
    status_distribution_statement = (
        select(
            ProcessingTask.status,
            func.count(ProcessingTask.id).label("count")
        )
        .where(ProcessingTask.user_id == current_user.id)
        .group_by(ProcessingTask.status)
    )
    status_distribution_raw = session.exec(status_distribution_statement).all()
    status_distribution = {status.value: count for status, count in status_distribution_raw}


    # Mockup some additional stats for a richer dashboard
    hours_processed = total_meetings * 0.3  # Assuming avg 18 mins per meeting
    accuracy_rate = 98.5
    team_members = 1

    return {
        "total_meetings": total_meetings,
        "hours_processed": round(hours_processed, 1),
        "team_members": team_members,
        "accuracy_rate": accuracy_rate,
        "meetings_over_time": meetings_chart_data,
        "status_distribution": status_distribution
    }

@app.get("/dashboard/weekly-activity")
async def get_weekly_activity(session: SessionDependency, current_user: CurrentUser):
    """
    Returns the number of meetings processed per day for the last 7 days.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=6)
    
    # This SQL function is database-agnostic for grabbing day-of-week data
    # For SQLite: strftime('%w', created_at) returns 0 for Sunday, 6 for Saturday
    
    query = (
        select(
            # --- THIS IS THE CORRECTED LINE ---
            func.strftime('%w', Meeting.created_at).label("day_of_week"),
            func.count(Meeting.id).label("meeting_count")
        )
        .where(
            Meeting.user_id == current_user.id,
            Meeting.created_at >= start_date,
            Meeting.created_at <= end_date
        )
        .group_by("day_of_week") # Grouping by the label is cleaner
    )
    
    results = session.exec(query).all()
    
    daily_counts = {day: 0 for day in range(7)}
    for row in results:
        daily_counts[int(row.day_of_week)] = row.meeting_count

    # Create a list for the last 7 days in order
    days_in_order = [(end_date - timedelta(days=i)) for i in range(6, -1, -1)]
    day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

    final_data = []
    for day in days_in_order:
        day_index = int(day.strftime("%w"))
        final_data.append({
            "day": day_names[day_index],
            "meetings": daily_counts.get(day_index, 0)
        })

    return final_data

@app.get("/dashboard/meeting-types")
async def get_meeting_types(session: SessionDependency, current_user: CurrentUser):
    """
    Categorizes meetings by parsing the 'Meeting Category' from the AI-generated notes.
    """
    statement = select(Meeting.notes).where(Meeting.user_id == current_user.id)
    notes_list = session.exec(statement).all()
    
    category_counts = {}
    
    # This regex will find the text after '### Meeting Category' until the next heading or end of string
    category_pattern = re.compile(r"### Meeting Category\s*\n(.*?)(?=\n###|\Z)", re.DOTALL)
    
    for notes in notes_list:
        match = category_pattern.search(notes)
        if match:
            category = match.group(1).strip()
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1
    
    return [{"name": name, "value": count} for name, count in category_counts.items()]


@app.get("/dashboard/processing-speed")
async def get_processing_speed(session: SessionDependency, current_user: CurrentUser):
    """
    Calculates processing duration by comparing task and meeting creation times.
    """
    query = (
        select(ProcessingTask.created_at, Meeting.created_at)
        .join(Meeting, ProcessingTask.id == Meeting.task_id)
        .where(ProcessingTask.user_id == current_user.id)
        .where(ProcessingTask.status == TaskStatus.COMPLETED)
    )
    
    results = session.exec(query).all()
    
    durations_in_minutes = []
    for task_start, meeting_end in results:
        duration = (meeting_end - task_start).total_seconds() / 60
        durations_in_minutes.append(duration)
        
    speed_buckets = {
        "0-5min": 0,
        "5-10min": 0,
        "10-15min": 0,
        "15-20min": 0,
        "20+min": 0
    }
    
    for duration in durations_in_minutes:
        if duration <= 5:
            speed_buckets["0-5min"] += 1
        elif duration <= 10:
            speed_buckets["5-10min"] += 1
        elif duration <= 15:
            speed_buckets["10-15min"] += 1
        elif duration <= 20:
            speed_buckets["15-20min"] += 1
        else:
            speed_buckets["20+min"] += 1
            
    return [{"time": name, "count": count} for name, count in speed_buckets.items()]
