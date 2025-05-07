# Gunicorn configuration for memory-intensive image processing
import multiprocessing

# Timeout settings
timeout = 300  # 5 minutes timeout instead of default 30 seconds

# Worker settings - use fewer workers for memory-intensive apps
workers = 2  # Reduce number of workers to conserve memory
worker_class = 'gthread'  # Use threads
threads = 4  # Number of threads per worker

# Memory optimization
max_requests = 500  # Restart workers after handling this many requests
max_requests_jitter = 100  # Add jitter to prevent all workers from restarting at once

# Log settings
loglevel = 'info'
accesslog = '-'  # Log to stdout
errorlog = '-'   # Log errors to stdout

# Keep-alive settings
keepalive = 5  # Keep connections alive for 5 seconds

# Prevent worker timeout
graceful_timeout = 120  # Give workers 2 minutes to finish processing