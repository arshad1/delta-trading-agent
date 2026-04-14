import multiprocessing
import os

# Gunicorn configuration
# You can override this with the GUNICORN_BIND environment variable
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")

# Uvicorn's ASGI worker class is required for FastAPI
worker_class = "uvicorn.workers.UvicornWorker"

# Automatically spawn workers based on CPU count
# Rule of thumb: (2 x $num_cores) + 1
workers = multiprocessing.cpu_count() * 2 + 1

# Automatic restarts to prevent long-term memory leaks
max_requests = 1000
max_requests_jitter = 50

# Ensure the agent has enough time to respond for heavy LLM calls
timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
