import os
from dotenv import load_dotenv
from app.main import app

load_dotenv()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT env variable
    uvicorn.run("run:app", host="0.0.0.0", port=port, reload=True)
