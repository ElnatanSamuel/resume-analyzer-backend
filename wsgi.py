import os
from dotenv import load_dotenv
from app import app

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Railway uses port 8080 by default
    app.run(host="0.0.0.0", port=port) 