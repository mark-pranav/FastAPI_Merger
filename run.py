import os
import uvicorn
from app.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render's PORT or fallback to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
