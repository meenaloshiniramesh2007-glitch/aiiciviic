"""
config.py — AI Civic Guardian
Reads from environment variables when deployed on Render.
Falls back to local defaults for development.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    # On Render, SECRET_KEY is auto-generated via render.yaml generateValue
    SECRET_KEY         = os.environ.get("SECRET_KEY", "aicivic_dev_secret_key_2024")

    # DATABASE_URL is injected by Render automatically (PostgreSQL)
    # Not set locally → SQLite is used instead
    DATABASE_URL       = os.environ.get("DATABASE_URL", "")
    DATABASE           = os.path.join(BASE_DIR, "civic_guardian.db")  # SQLite fallback

    UPLOAD_FOLDER      = os.path.join(BASE_DIR, "static", "uploads")
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024    # 100 MB
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
    ALLOWED_VIDEO_EXT  = {"mp4", "webm", "ogg", "mov", "avi"}

    # Built-in credentials
    OFFICER_EMAIL    = os.environ.get("OFFICER_EMAIL",   "officer@aicivic.com")
    OFFICER_PASSWORD = os.environ.get("OFFICER_PASSWORD","officer123")
    ADMIN_EMAIL      = os.environ.get("ADMIN_EMAIL",     "admin@aicivic.com")
    ADMIN_PASSWORD   = os.environ.get("ADMIN_PASSWORD",  "admin123")

    # Department → officer email mapping
    DEPT_EMAILS = {
        "Roads & Infrastructure Dept"  : "roads@aicivic.com",
        "Sanitation & Waste Dept"      : "sanitation@aicivic.com",
        "Water Supply & Sewage Board"  : "water@aicivic.com",
        "Electricity & Lighting Dept"  : "electricity@aicivic.com",
        "Public Works Dept"            : "publicworks@aicivic.com",
        "Building & Planning Dept"     : "building@aicivic.com",
        "Municipal Corporation"        : "municipal@aicivic.com",
        "Horticulture & Parks Dept"    : "horticulture@aicivic.com",
        "Environment & Pollution Dept" : "environment@aicivic.com",
        "Animal Control & Health Dept" : "animal@aicivic.com",
        "Traffic Engineering Dept"     : "traffic@aicivic.com",
        "Civic Maintenance Dept"       : "civic@aicivic.com",
    }
