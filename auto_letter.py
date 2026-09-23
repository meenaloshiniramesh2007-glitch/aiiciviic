"""
auto_letter.py — AI Civic Guardian
====================================
Generates a formal complaint letter automatically when a complaint is submitted.
The letter is addressed to the respective department head.

FUNCTIONS:
    generate_letter(complaint_dict)  → returns letter text (str)
    save_letter(complaint_id, dept, letter_text)
    get_letter(complaint_id)         → returns letter row or None
    acknowledge_letter(complaint_id) → marks letter as seen by officer
"""

import os
from datetime import datetime
from database.database import get_connection

# ─── PRIORITY RULES ───────────────────────────────────────────────────────────
# Auto-assign priority based on keywords in description
PRIORITY_KEYWORDS = {
    "High": [
        "accident", "dangerous", "urgent", "emergency", "fire", "flood",
        "burst", "collapse", "injury", "bleeding", "attack", "dead",
        "electric shock", "live wire", "toxic", "poisonous", "critical",
    ],
    "Medium": [
        "broken", "damaged", "blocked", "overflow", "leaking", "pothole",
        "garbage", "stray", "missing", "not working", "complaint", "issue",
        "problem", "stuck", "delay", "pending",
    ],
}

def determine_priority(description):
    """Return High / Medium / Normal based on description keywords."""
    desc_lower = description.lower()
    for kw in PRIORITY_KEYWORDS["High"]:
        if kw in desc_lower:
            return "High"
    for kw in PRIORITY_KEYWORDS["Medium"]:
        if kw in desc_lower:
            return "Medium"
    return "Normal"


# ─── DEPARTMENT HEAD TITLES ────────────────────────────────────────────────────
DEPT_HEAD = {
    "Roads & Infrastructure Dept"  : "The Executive Engineer, Roads & Infrastructure",
    "Sanitation & Waste Dept"      : "The Health Officer, Sanitation & Waste Management",
    "Water Supply & Sewage Board"  : "The Chief Engineer, Water Supply & Sewage Board",
    "Electricity & Lighting Dept"  : "The Superintendent Engineer, Electricity & Lighting",
    "Public Works Dept"            : "The Executive Engineer, Public Works Department",
    "Building & Planning Dept"     : "The Town Planning Officer, Building & Planning",
    "Municipal Corporation"        : "The Commissioner, Municipal Corporation",
    "Horticulture & Parks Dept"    : "The Horticulture Officer, Parks & Gardens",
    "Environment & Pollution Dept" : "The Environmental Officer, Pollution Control Board",
    "Animal Control & Health Dept" : "The Veterinary Officer, Animal Control & Health",
    "Traffic Engineering Dept"     : "The Traffic Engineer, Traffic Management",
    "Civic Maintenance Dept"       : "The Maintenance Officer, Civic Infrastructure",
}

PRIORITY_LABELS = {
    "High"  : "⚠️  HIGH PRIORITY — Immediate Action Required",
    "Medium": "🔶 MEDIUM PRIORITY — Action Required Within 3 Days",
    "Normal": "🔵 NORMAL PRIORITY — Action Required Within 7 Days",
}


# ─── LETTER GENERATOR ─────────────────────────────────────────────────────────
def generate_letter(complaint):
    """
    Generate a formal complaint letter for the given complaint.

    Args:
        complaint: dict or sqlite3.Row with complaint fields

    Returns:
        str: formatted letter text
    """
    # Convert sqlite3.Row to plain dict so .get() works safely
    if hasattr(complaint, "keys"):
        c = dict(complaint)
    else:
        c = complaint

    dept       = c.get("department", "Municipal Corporation")
    location   = c.get("location", "Not specified")
    desc       = c.get("description", "")
    citizen    = c.get("user_name") or "A Citizen"
    phone      = c.get("user_phone") or "Not provided"
    comp_id    = c.get("id", 0)
    submitted  = str(c.get("submitted_at", ""))[:16] or datetime.now().strftime("%Y-%m-%d %H:%M")
    ai_cat     = c.get("ai_category") or "General Civic Issue"
    priority   = c.get("priority") or determine_priority(desc)
    img_info   = "Attached (see complaint portal)" if c.get("image") else "Not attached"
    vid_info   = "Attached (see complaint portal)" if c.get("video") else "Not attached"
    voice_info = "Yes — voice description provided" if c.get("voice_transcript") else "No"

    to_address     = DEPT_HEAD.get(dept, f"The Officer In-Charge, {dept}")
    priority_label = PRIORITY_LABELS.get(priority, PRIORITY_LABELS["Normal"])

    letter = f"""
================================================================================
                        AI CIVIC GUARDIAN SYSTEM
               AUTOMATED CIVIC COMPLAINT DISPATCH LETTER
================================================================================

                                                    Date: {datetime.now().strftime("%d %B %Y")}
                                                    Time: {datetime.now().strftime("%I:%M %p")}
                                                    Ref : AICG-{comp_id:05d}

TO:
{to_address}
{dept}

SUBJECT: Civic Complaint Requiring Immediate Departmental Action — Ref #{comp_id:05d}

{priority_label}

Dear Sir/Madam,

This is an automatically generated complaint dispatch letter from the
AI Civic Guardian Smart Complaint Management System. A civic complaint
has been registered by a citizen and has been classified and routed to
your department for necessary action.

────────────────────────────────────────────────────────────────────────────────
COMPLAINT DETAILS
────────────────────────────────────────────────────────────────────────────────

  Complaint ID       : AICG-{comp_id:05d}
  Submitted On       : {submitted}
  Priority Level     : {priority}
  AI Classification  : {ai_cat}
  Department Routed  : {dept}

────────────────────────────────────────────────────────────────────────────────
CITIZEN INFORMATION
────────────────────────────────────────────────────────────────────────────────

  Complainant Name   : {citizen}
  Contact Number     : {phone}

────────────────────────────────────────────────────────────────────────────────
LOCATION OF ISSUE
────────────────────────────────────────────────────────────────────────────────

  {location}

────────────────────────────────────────────────────────────────────────────────
COMPLAINT DESCRIPTION
────────────────────────────────────────────────────────────────────────────────

  {desc}

────────────────────────────────────────────────────────────────────────────────
EVIDENCE ATTACHED
────────────────────────────────────────────────────────────────────────────────

  Photographic Evidence  : {img_info}
  Video Evidence         : {vid_info}
  Voice Description      : {voice_info}

  (All media files are accessible via the AI Civic Guardian portal under
   Complaint ID AICG-{comp_id:05d})

────────────────────────────────────────────────────────────────────────────────
ACTION REQUIRED
────────────────────────────────────────────────────────────────────────────────

You are hereby requested to:

  1. Acknowledge receipt of this complaint within 24 hours.
  2. Assign a field officer for on-site inspection.
  3. Update the complaint status on the AI Civic Guardian portal.
  4. Resolve the issue within the prescribed SLA period:
       High Priority   → Within 24 hours
       Medium Priority → Within 3 working days
       Normal Priority → Within 7 working days
  5. Provide resolution remarks upon completion.

Failure to respond within the prescribed timeline will escalate the complaint
to the Municipal Commissioner and Admin Dashboard automatically.

────────────────────────────────────────────────────────────────────────────────

This letter has been auto-generated by the AI Civic Guardian System.
No manual signature is required. Please refer to the online portal for
real-time tracking and status updates.

Portal: http://localhost:5000
Admin:  admin@aicivic.com

                                        Yours faithfully,
                                        AI Civic Guardian System
                                        Smart Civic Complaint Management
                                        Government Urban Services Division

================================================================================
                     CONFIDENTIAL — OFFICIAL USE ONLY
================================================================================
"""
    return letter.strip()


# ─── DATABASE OPERATIONS ──────────────────────────────────────────────────────

def save_letter(complaint_id, department, letter_text):
    """Save generated letter to auto_letters table."""
    conn = get_connection()
    # Remove old letter if exists (regenerate)
    conn.execute("DELETE FROM auto_letters WHERE complaint_id = ?", (complaint_id,))
    conn.execute(
        "INSERT INTO auto_letters (complaint_id, department, letter_text) VALUES (?, ?, ?)",
        (complaint_id, department, letter_text)
    )
    # Also store summary in complaints table
    conn.execute(
        "UPDATE complaints SET auto_letter = ?, letter_sent_at = CURRENT_TIMESTAMP WHERE id = ?",
        (letter_text[:500] + "..." if len(letter_text) > 500 else letter_text, complaint_id)
    )
    conn.commit()
    conn.close()


def get_letter(complaint_id):
    """Get auto-letter for a complaint."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM auto_letters WHERE complaint_id = ?", (complaint_id,)
    ).fetchone()
    conn.close()
    return row


def get_letters_by_dept(department):
    """Get all auto-letters for a specific department."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT al.*, c.location, c.description, c.status, c.priority,
                  c.submitted_at, c.user_name, c.image, c.video
           FROM auto_letters al
           JOIN complaints c ON al.complaint_id = c.id
           WHERE al.department = ?
           ORDER BY al.generated_at DESC""",
        (department,)
    ).fetchall()
    conn.close()
    return rows


def acknowledge_letter(complaint_id):
    """Mark letter as acknowledged by department officer."""
    conn = get_connection()
    conn.execute(
        "UPDATE auto_letters SET acknowledged = 1 WHERE complaint_id = ?",
        (complaint_id,)
    )
    conn.commit()
    conn.close()
