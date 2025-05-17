from fastapi import FastAPI, APIRouter, Body, HTTPException, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from memory_manager import MemoryManager  
from passlib.context import CryptContext
from sqlalchemy.dialects.sqlite import BLOB as SQLiteUUID
from sqlalchemy.types import CHAR
from jose import JWTError, jwt
from passlib.hash import bcrypt
from datetime import datetime, timedelta
from pydantic import BaseModel
import json
import shutil
import uuid
import sqlite3
import requests
import os

# === CONFIG ===
DB_URL = "sqlite:///data/chatbot.db"
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()
memory_store = {}


# === MODELS ===
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    chats = relationship("Chat", back_populates="user")

class Chat(Base):
    __tablename__ = 'chats'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="chats")
    # Add this field to Chat:
    conversation_id = Column(String(36), index=True)  # For compatibility across DBs


    
# === INIT DATABASE ===
def init_db():
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)

# === AUTH ===
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# === APP SETUP ===
app = FastAPI()
chat_router = APIRouter()
init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/llm/token")

# === UTILS ===
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# === AUTH ROUTES ===
@chat_router.post("/register")
def register_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    if db.query(User).filter_by(username=form_data.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed = get_password_hash(form_data.password)
    user = User(username=form_data.username, hashed_password=hashed)
    db.add(user)
    db.commit()
    print("User registered : "+form_data.username)
    return {"msg": "User registered"}

   
@chat_router.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter_by(username=form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user.username})
    print("User logged in : "+ token)
    return {"access_token": token, "token_type": "bearer","username": user.username}
    
    
# TODO: [debug this function later in the day]    
# Update password 
@chat_router.post("/update-password")
def update_password(token: str = Depends(oauth2_scheme), old_password: str = Body(None),new_password: str = Body(None),db: Session = Depends(get_db)):
    try:
        # Verify token validity and expiration date
        payload = jwt.decode(
            token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": True}
        )
        username = payload.get("sub")

        if not username:
            raise HTTPException(status_code=401, detail="Invalid token or expired")

        user = db.query(User).filter_by(username=username).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Input validation
        if not old_password or not new_password:
            raise HTTPException(status_code=400, detail="Missing input parameters")

        # Validate password length and format
        if len(new_password) < 8:
            raise HTTPException(
                status_code=400, detail="Password must be at least 8 characters long"
            )

        # Check if the provided old password matches the hashed password in the database
        if not pwd_context.verify(old_password, user.hashed_password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password")

        # Hash the new password using bcrypt
        hashed_password = pwd_context.hash(new_password)

        # Update the hashed password in the database
        try:
            db.commit()
        except Exception as e:
            # Handle potential exceptions during database commit
            print(f"Error updating password: {e}")
            db.rollback()

        return {"msg": "Password updated successfully"}

    except JWTError as e:
        # Handle JWT errors and invalid tokens
        raise HTTPException(status_code=401, detail="Invalid token") from e
    
   


# Instead of a simple post, use streaming response from the model
async def generate_streaming_response(prompt):
    # Open a streaming connection to the model API
    with requests.post("http://localhost:11434/api/generate", json={
        "model": "qwen1.5_q8",
        "prompt": prompt,
        "stream": True
    }, stream=True) as response:
        if response.status_code == 200:
            # Yield chunks as they're received
            for chunk in response.iter_lines():
                if chunk:
                    yield chunk.decode('utf-8')  # Decode the chunk to text
        else:
            raise HTTPException(status_code=500, detail="Error streaming response from model.")


@chat_router.post("/chat")
async def chat_streaming(
    conversation_id: str = Body(None),
    message: str = Body(...),
    new_chat: bool = Body(False),
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
    ):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": True})
        username = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter_by(username=username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    print("Received Message : " + message)

    # Generate new UUID if starting a new chat or none provided
    if new_chat:
        if conversation_id:
            raise HTTPException(
                status_code=400,
                detail="Cannot provide both 'new_chat=true' and a conversation_id. Choose one."
            )
        conversation_id = str(uuid.uuid4())
    elif conversation_id:
        if not db.query(Chat).filter_by(conversation_id=conversation_id).first():
            raise HTTPException(
                status_code=400,
                detail="Invalid Conversation ID."
            )
    elif not conversation_id:
        raise HTTPException(
            status_code=400,
            detail="conversation_id not specified."
        )

    # Prepare conversation memory...
    key = f"{conversation_id}"
    memory_dir = f"data/memory/{username}/{conversation_id}"
    os.makedirs(memory_dir, exist_ok=True)
    index_path = os.path.join(memory_dir, "faiss.index")
    memory_path = os.path.join(memory_dir, "memory.pkl")

    if key not in memory_store:
        memory = MemoryManager(index_path=index_path, memory_path=memory_path)
        if not os.path.exists(index_path) or not os.path.exists(memory_path):
            history = db.query(Chat).filter_by(user_id=user.id, conversation_id=conversation_id).order_by(Chat.timestamp.asc()).all()
            for h in history:
                memory.add_message(h.role, h.content)
        memory_store[key] = memory

    memory = memory_store[key]
    memory.add_message("user", message)
    context = memory.get_context(message)
    prompt = ""
    for msg in context:
        prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    prompt += "Assistant:"

    # Stream the model response

    def generate():
        with requests.post("http://localhost:11434/api/generate", json={
            "model":"physics-llama",
            "prompt":prompt,
            "stream":True,
            }, stream=True) as r:
            assistant_msg = ""
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response")
                    assistant_msg += token
                    yield line.decode("utf-8") +"\n"
            # Add message to db
            memory.add_message("assistant",assistant_msg)
            db.add(Chat(user_id=user.id, conversation_id=conversation_id, role="user", content=message))
            
            db.add(Chat(user_id=user.id, conversation_id=conversation_id, role="assistant", content=assistant_msg))
            db.commit()
    return StreamingResponse(generate(), media_type="text/event-stream", headers={"X-Conversation-ID":conversation_id})
    


#==== Explain unit conversions
@chat_router.post("/conversion")
def get_conversion(message:str = Body(...)):
    print("Received : ", message)
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
        "model": "qwen1.5_q8", # Switch to physics-llama before deploy
        "prompt": message,
        "stream": False
        }).json()["response"]
        return {"response" : response}
    except Exception as e:
        raise HTTPException(
                status_code=500,
                detail="An Error Occured !"
            )
        
        
#===== Conversations ====
# Get chat history
@chat_router.get("/conversations")
def list_conversations(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter_by(username=username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    conversation_ids = (
        db.query(Chat.conversation_id)
        .filter_by(user_id=user.id)
        .distinct()
        .all()
    )

    print("Convos:",conversation_ids)
    

    conversations = []
    for conv_id_tuple in conversation_ids:
        conv_id = conv_id_tuple[0]
        first_msg = (
            db.query(Chat)
            .filter_by(user_id=user.id, conversation_id=conv_id, role="user")
            .order_by(Chat.timestamp.asc())
            .first()
        )
        chat_description = (
            db.query(Chat)
            .filter_by(user_id=user.id, conversation_id=conv_id, role="assistant")
            .order_by(Chat.timestamp.asc())
            .first()
        )
        if first_msg:
            header = " ".join(first_msg.content.split()[:20])
            description = " ".join(chat_description.content.split()[:20])
            conversations.append({
                "conversation_id": conv_id,
                "conversation_header": header,
                "conversation_desc": description
            })

    return {"messages": conversations, "total": len(conversations)}


# Fetch messages for converstion
@chat_router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter_by(username=username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    chats = db.query(Chat).filter_by(user_id=user.id, conversation_id=conversation_id).order_by(Chat.timestamp).all()
    if not chats:
        raise HTTPException(status_code=404, detail="Conversation not found")

    user_msgs = [chat.content for chat in chats if chat.role == "user"]
    bot_msgs = [chat.content for chat in chats if chat.role == "assistant"]

    return {
        "conversation_id": conversation_id,
        "conversations": {
            "userMessages": user_msgs,
            "botMessages": bot_msgs
        }
    }

# Delete conversation
@chat_router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter_by(username=username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    chats = db.query(Chat).filter_by(user_id=user.id, conversation_id=conversation_id).all()
    if not chats:
        raise HTTPException(status_code=404, detail="Conversation not found")

    for chat in chats:
        db.delete(chat)
    db.commit()

    # securely delete memory
    shutil.rmtree(f"data/memory/{username}/{conversation_id}", ignore_errors=True)
    memory_store.pop(f"{username}:{conversation_id}", None)

    return {"status": "deleted"}


# Get info about conversation (/llm/debug/memory?conversation_id=1)
@chat_router.get("/debug/memory")
def debug_memory(conversation_id: str = Query(...), token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter_by(username=username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    key = f"{username}:{conversation_id}"
    memory_dir = f"data/memory/{username}/{conversation_id}"
    index_path = os.path.join(memory_dir, "faiss.index")
    memory_path = os.path.join(memory_dir, "memory.pkl")

    if key not in memory_store:
        if not os.path.exists(index_path) or not os.path.exists(memory_path):
            raise HTTPException(status_code=404, detail="No memory found for this conversation.")
        memory = MemoryManager(index_path=index_path, memory_path=memory_path)
        memory_store[key] = memory

    memory = memory_store[key]

    return {
        "recent_history": memory.recent_history,
        "summary": memory.summarize_old(),
        "long_term_count": len(memory.long_term_memory),
        "vector_index_size": len(memory.vector_map)
    }


# === ROUTE REGISTRATION ===
app.include_router(chat_router, prefix="/llm")

@app.get("/")
def root():
    return {"status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
