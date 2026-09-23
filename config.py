"""
config.py — AI Civic Guardian
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    SECRET_KEY         = "aicivic_secret_key_2024"
    DATABASE           = os.path.join(BASE_DIR, "civic_guardian.db")
    UPLOAD_FOLDER      = os.path.join(BASE_DIR, "static", "uploads")
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024   # 100 MB — needed for videos
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}
    ALLOWED_VIDEO_EXT  = {"mp4", "webm", "ogg", "mov", "avi"}

    OFFICER_EMAIL    = "officer@aicivic.com"
    OFFICER_PASSWORD = "officer123"
    ADMIN_EMAIL      = "admin@aicivic.com"
    ADMIN_PASSWORD   = "admin123"

    # Department → Officer email mapping (for auto-letter routing)
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
