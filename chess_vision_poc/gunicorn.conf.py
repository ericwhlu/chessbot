"""Single process preserves the in-memory visitor registry; threads serve friends."""
import os

bind = '0.0.0.0:' + os.environ.get('PORT','8765')
workers = 1
worker_class = 'gthread'
threads = 4
timeout = 120
accesslog = '-'
errorlog = '-'
