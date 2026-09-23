"""
app.py — AI Civic Guardian
Main Flask application.

New features:
  - Voice input (Web Speech API, English + Tamil)
  - Bilingual UI (English / Tamil)
  - Video upload support
  - Auto complaint letter generation → routed to department
  - /dept_inbox/<department> — department-specific complaint inbox
  - /auto_letter/<complaint_id> — view/print auto-generated letter
  - /acknowledge/<complaint_id> — officer acknowledges letter
  - /transcribe — AJAX endpoint for voice transcript saving
"""

import os
import time
from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, jsonify, Response)
from werkzeug.utils import secure_filename
from config import Config
from database.database import create_tables, get_connection
from classifier import predict_category, predict_image, get_classifier_status
from auto_letter import (generate_letter, save_letter,
                         get_letter, get_letters_by_dept,
                         acknowledge_letter, determine_priority)

# ─── APP SETUP ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
create_tables()

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DEPARTMENTS = [
    "Roads & Infrastructure Dept",
    "Sanitation & Waste Dept",
    "Water Supply & Sewage Board",
    "Electricity & Lighting Dept",
    "Public Works Dept",
    "Building & Planning Dept",
    "Municipal Corporation",
    "Horticulture & Parks Dept",
    "Environment & Pollution Dept",
    "Animal Control & Health Dept",
    "Traffic Engineering Dept",
    "Civic Maintenance Dept",
]

STATUS_LIST = ["Pending", "In Progress", "Resolved", "Rejected"]

PRIORITY_LIST = ["High", "Medium", "Normal"]

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in app.config["ALLOWED_EXTENSIONS"]

def allowed_video(filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in app.config["ALLOWED_VIDEO_EXT"]

def login_required(role=None):
    if "user" not in session:
        return False
    if role and session.get("role") != role:
        return False
    return True

def save_upload(file, prefix, allowed_fn):
    """Save uploaded file and return filename or None."""
    if not file or not file.filename:
        return None
    if not allowed_fn(file.filename):
        return None
    filename    = secure_filename(file.filename)
    unique_name = f"{prefix}_{int(time.time())}_{filename}"
    save_path   = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)
    return unique_name

# ─── HOME ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ─── REGISTER ─────────────────────────────────────────────────────────────────
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        phone    = request.form.get("phone", "").strip()
        password = request.form.get("password", "").strip()
        lang     = request.form.get("lang", "en")

        if not name or not email or not password:
            flash("Name, email and password are required.", "error")
            return render_template("register.html")

        conn = get_connection()
        if conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone():
            conn.close()
            flash("Email already registered. Please login.", "error")
            return render_template("register.html")

        conn.execute(
            "INSERT INTO users (name, email, phone, password, preferred_lang) VALUES (?,?,?,?,?)",
            (name, email, phone, password, lang)
        )
        conn.commit()
        conn.close()
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

# ─── LOGIN ────────────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        if email == app.config["OFFICER_EMAIL"] and password == app.config["OFFICER_PASSWORD"]:
            session.update(user="Officer", role="officer", email=email)
            return redirect(url_for("officer_dashboard"))

        if email == app.config["ADMIN_EMAIL"] and password == app.config["ADMIN_PASSWORD"]:
            session.update(user="Admin", role="admin", email=email)
            return redirect(url_for("admin_dashboard"))

        conn = get_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE email=? AND password=?", (email, password)
        ).fetchone()
        conn.close()

        if user:
            session.update(
                user=user["name"], role="user",
                user_id=user["id"], email=email,
                lang=user["preferred_lang"] or "en"
            )
            return redirect(url_for("user_dashboard"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")

# ─── LOGOUT ───────────────────────────────────────────────────────────────────
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))

# ─── USER DASHBOARD ───────────────────────────────────────────────────────────
@app.route("/user")
def user_dashboard():
    if not login_required("user"):
        flash("Please login first.", "error")
        return redirect(url_for("login"))

    conn = get_connection()
    complaints = conn.execute(
        "SELECT * FROM complaints WHERE user_id=? ORDER BY submitted_at DESC",
        (session["user_id"],)
    ).fetchall()
    conn.close()
    return render_template("user_dashboard.html",
                           complaints=complaints, user=session["user"])

# ─── SUBMIT COMPLAINT ─────────────────────────────────────────────────────────
@app.route("/complaint", methods=["GET", "POST"])
def complaint():
    if not login_required("user"):
        flash("Please login to submit a complaint.", "error")
        return redirect(url_for("login"))

    if request.method == "POST":
        department       = request.form.get("department", "").strip()
        location         = request.form.get("location", "").strip()
        description      = request.form.get("description", "").strip()
        voice_transcript = request.form.get("voice_transcript", "").strip()
        desc_lang        = request.form.get("desc_lang", "en")

        # Use voice transcript as description if description is empty
        if not description and voice_transcript:
            description = voice_transcript

        if not department or not location or not description:
            flash("Department, location and description are required.", "error")
            return render_template("complaint.html", departments=DEPARTMENTS)

        # ── AI: Text classification ───────────────────────────────────────────
        ai_result     = predict_category(description)
        ai_category   = ai_result.get("category", "")
        ai_confidence = ai_result.get("confidence", 0)

        if department == "auto" and ai_result.get("available"):
            department = ai_result.get("department", "Municipal Corporation")

        # ── Auto priority ─────────────────────────────────────────────────────
        priority = determine_priority(description)

        # ── Image upload ──────────────────────────────────────────────────────
        image_filename = save_upload(
            request.files.get("image"),
            f"{session['user_id']}_img",
            allowed_file
        )
        ai_image_label = None
        if image_filename:
            img_path   = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
            img_result = predict_image(img_path)
            if img_result.get("available"):
                ai_image_label = img_result.get("display", "")

        # ── Video upload ──────────────────────────────────────────────────────
        video_filename = save_upload(
            request.files.get("video"),
            f"{session['user_id']}_vid",
            allowed_video
        )

        # ── Get user phone ────────────────────────────────────────────────────
        conn = get_connection()
        user_row = conn.execute(
            "SELECT phone FROM users WHERE id=?", (session["user_id"],)
        ).fetchone()
        user_phone = user_row["phone"] if user_row else ""

        # ── Save complaint ────────────────────────────────────────────────────
        from database.database import USE_POSTGRES
        if USE_POSTGRES:
            # PostgreSQL: use RETURNING id to get the new row's id
            cur = conn.execute(
                """INSERT INTO complaints
                   (user_id, user_name, user_phone, department, category, location,
                    description, description_lang, image, video, voice_transcript,
                    status, priority, ai_category, ai_confidence, ai_image_label)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'Pending',%s,%s,%s,%s)
                   RETURNING id""",
                (session["user_id"], session["user"], user_phone,
                 department, ai_category, location,
                 description, desc_lang, image_filename, video_filename, voice_transcript,
                 priority, ai_category, ai_confidence, ai_image_label)
            )
            complaint_id = cur.fetchone()["id"]
        else:
            cur = conn.execute(
                """INSERT INTO complaints
                   (user_id, user_name, user_phone, department, category, location,
                    description, description_lang, image, video, voice_transcript,
                    status, priority, ai_category, ai_confidence, ai_image_label)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,  'Pending',?,?,?,?)""",
                (session["user_id"], session["user"], user_phone,
                 department, ai_category, location,
                 description, desc_lang, image_filename, video_filename, voice_transcript,
                 priority, ai_category, ai_confidence, ai_image_label)
            )
            complaint_id = cur.lastrowid
        conn.commit()

        # ── Auto-generate and save complaint letter ───────────────────────────
        complaint_row = conn.execute(
            "SELECT * FROM complaints WHERE id=?", (complaint_id,)
        ).fetchone()
        conn.close()

        letter_text = generate_letter(complaint_row)
        save_letter(complaint_id, department, letter_text)

        flash(f"Complaint #{complaint_id} submitted! Auto-letter generated and sent to {department}.",
              "success")
        return redirect(url_for("user_dashboard"))

    return render_template("complaint.html", departments=DEPARTMENTS)

# ─── AI SUGGEST (AJAX) ────────────────────────────────────────────────────────
@app.route("/ai_suggest", methods=["POST"])
def ai_suggest():
    description = request.form.get("description", "").strip()
    if not description or len(description) < 8:
        return jsonify({"available": False})
    result = predict_category(description)
    result["priority"] = determine_priority(description)
    return jsonify(result)

# ─── VOICE TRANSCRIPT SAVE (AJAX) ─────────────────────────────────────────────
@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Save voice transcript sent from browser speech recognition."""
    transcript = request.form.get("transcript", "").strip()
    lang       = request.form.get("lang", "en")
    if not transcript:
        return jsonify({"ok": False})
    # Run AI classify on transcript too
    result = predict_category(transcript)
    result["transcript"] = transcript
    result["lang"]       = lang
    result["priority"]   = determine_priority(transcript)
    result["ok"]         = True
    return jsonify(result)

# ─── TRACK ────────────────────────────────────────────────────────────────────
@app.route("/track")
def track():
    if not login_required("user"):
        flash("Please login.", "error")
        return redirect(url_for("login"))

    status_filter = request.args.get("status", "")
    conn = get_connection()
    if status_filter:
        complaints = conn.execute(
            "SELECT * FROM complaints WHERE user_id=? AND status=? ORDER BY submitted_at DESC",
            (session["user_id"], status_filter)
        ).fetchall()
    else:
        complaints = conn.execute(
            "SELECT * FROM complaints WHERE user_id=? ORDER BY submitted_at DESC",
            (session["user_id"],)
        ).fetchall()
    conn.close()

    return render_template("track.html",
                           complaints=complaints,
                           status_list=STATUS_LIST,
                           selected_status=status_filter,
                           user=session["user"])

# ─── OFFICER DASHBOARD ────────────────────────────────────────────────────────
@app.route("/officer")
def officer_dashboard():
    if not login_required("officer"):
        flash("Officer login required.", "error")
        return redirect(url_for("login"))

    status_filter   = request.args.get("status", "")
    dept_filter     = request.args.get("dept", "")
    priority_filter = request.args.get("priority", "")

    conn   = get_connection()
    query  = "SELECT * FROM complaints WHERE 1=1"
    params = []
    if status_filter:
        query += " AND status=?";   params.append(status_filter)
    if dept_filter:
        query += " AND department=?"; params.append(dept_filter)
    if priority_filter:
        query += " AND priority=?"; params.append(priority_filter)
    query += " ORDER BY CASE priority WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END, submitted_at DESC"

    complaints = conn.execute(query, params).fetchall()

    counts = {s: conn.execute(
        "SELECT COUNT(*) FROM complaints WHERE status=?", (s,)
    ).fetchone()[0] for s in STATUS_LIST}
    counts["Total"] = conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]

    # Unacknowledged letter count
    pending_letters = conn.execute(
        "SELECT COUNT(*) FROM auto_letters WHERE acknowledged=0"
    ).fetchone()[0]

    conn.close()
    return render_template("officer_dashboard.html",
                           complaints=complaints,
                           status_list=STATUS_LIST,
                           priority_list=PRIORITY_LIST,
                           departments=DEPARTMENTS,
                           selected_status=status_filter,
                           selected_dept=dept_filter,
                           selected_priority=priority_filter,
                           counts=counts,
                           pending_letters=pending_letters)

# ─── DEPARTMENT INBOX ─────────────────────────────────────────────────────────
@app.route("/dept_inbox/<path:department>")
def dept_inbox(department):
    """Department-specific complaint inbox with auto-generated letters."""
    if not login_required("officer") and not login_required("admin"):
        flash("Login required.", "error")
        return redirect(url_for("login"))

    conn = get_connection()
    complaints = conn.execute(
        """SELECT c.*, al.letter_text, al.acknowledged, al.generated_at
           FROM complaints c
           LEFT JOIN auto_letters al ON c.id = al.complaint_id
           WHERE c.department=?
           ORDER BY CASE c.priority WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END,
                    c.submitted_at DESC""",
        (department,)
    ).fetchall()

    counts = {s: conn.execute(
        "SELECT COUNT(*) FROM complaints WHERE department=? AND status=?",
        (department, s)
    ).fetchone()[0] for s in STATUS_LIST}
    counts["Total"] = conn.execute(
        "SELECT COUNT(*) FROM complaints WHERE department=?", (department,)
    ).fetchone()[0]
    conn.close()

    return render_template("dept_inbox.html",
                           department=department,
                           complaints=complaints,
                           status_list=STATUS_LIST,
                           priority_list=PRIORITY_LIST,
                           counts=counts)

# ─── UPDATE STATUS ────────────────────────────────────────────────────────────
@app.route("/update_status/<int:complaint_id>/<status>")
def update_status(complaint_id, status):
    if not login_required("officer"):
        flash("Officer login required.", "error")
        return redirect(url_for("login"))

    if status not in STATUS_LIST:
        flash("Invalid status.", "error")
        return redirect(url_for("officer_dashboard"))

    conn = get_connection()
    conn.execute(
        "UPDATE complaints SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (status, complaint_id)
    )
    conn.commit()
    conn.close()
    flash(f"Complaint #{complaint_id} updated to '{status}'.", "success")

    # Return to dept_inbox if that's where officer came from
    ref = request.referrer or ""
    if "dept_inbox" in ref:
        return redirect(ref)
    return redirect(url_for("officer_dashboard"))

# ─── ACKNOWLEDGE LETTER ───────────────────────────────────────────────────────
@app.route("/acknowledge/<int:complaint_id>")
def ack_letter(complaint_id):
    if not login_required("officer") and not login_required("admin"):
        return jsonify({"ok": False})
    acknowledge_letter(complaint_id)
    return jsonify({"ok": True})

# ─── VIEW AUTO-LETTER ─────────────────────────────────────────────────────────
@app.route("/auto_letter/<int:complaint_id>")
def view_auto_letter(complaint_id):
    if "user" not in session:
        flash("Please login.", "error")
        return redirect(url_for("login"))

    conn = get_connection()
    complaint = conn.execute(
        "SELECT * FROM complaints WHERE id=?", (complaint_id,)
    ).fetchone()
    conn.close()

    if not complaint:
        flash("Complaint not found.", "error")
        return redirect(url_for("index"))

    if session.get("role") == "user" and complaint["user_id"] != session.get("user_id"):
        flash("Access denied.", "error")
        return redirect(url_for("user_dashboard"))

    letter_row = get_letter(complaint_id)
    # If letter was never generated (old complaint), generate now
    if not letter_row:
        letter_text = generate_letter(dict(complaint))
        save_letter(complaint_id, complaint["department"], letter_text)
        letter_row  = get_letter(complaint_id)

    return render_template("auto_letter.html",
                           complaint=complaint,
                           letter=letter_row)

# ─── DOWNLOAD LETTER AS TEXT ──────────────────────────────────────────────────
@app.route("/download_letter/<int:complaint_id>")
def download_letter(complaint_id):
    if "user" not in session:
        return redirect(url_for("login"))

    letter_row = get_letter(complaint_id)
    if not letter_row:
        flash("Letter not found.", "error")
        return redirect(url_for("index"))

    text = letter_row["letter_text"]
    return Response(
        text,
        mimetype="text/plain",
        headers={"Content-Disposition":
                 f"attachment; filename=complaint_AICG{complaint_id:05d}.txt"}
    )

# ─── COMPLAINT LETTER (CITIZEN VIEW) ─────────────────────────────────────────
@app.route("/letter/<int:complaint_id>")
def complaint_letter(complaint_id):
    if "user" not in session:
        flash("Please login.", "error")
        return redirect(url_for("login"))

    conn = get_connection()
    complaint = conn.execute("SELECT * FROM complaints WHERE id=?", (complaint_id,)).fetchone()
    conn.close()

    if not complaint:
        flash("Complaint not found.", "error")
        return redirect(url_for("index"))

    if session.get("role") == "user" and complaint["user_id"] != session.get("user_id"):
        flash("Access denied.", "error")
        return redirect(url_for("user_dashboard"))

    return render_template("complaint_letter.html", complaint=complaint)

# ─── ADMIN DASHBOARD ──────────────────────────────────────────────────────────
@app.route("/admin")
def admin_dashboard():
    if not login_required("admin"):
        flash("Admin login required.", "error")
        return redirect(url_for("login"))

    conn = get_connection()
    complaints  = conn.execute("SELECT * FROM complaints ORDER BY submitted_at DESC").fetchall()
    users       = conn.execute("SELECT * FROM users ORDER BY id DESC").fetchall()

    stats = {
        "total_complaints": conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0],
        "total_users"     : conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
        "high_priority"   : conn.execute("SELECT COUNT(*) FROM complaints WHERE priority='High'").fetchone()[0],
        "pending_letters" : conn.execute("SELECT COUNT(*) FROM auto_letters WHERE acknowledged=0").fetchone()[0],
    }
    for s in STATUS_LIST:
        stats[s.lower().replace(" ", "_")] = conn.execute(
            "SELECT COUNT(*) FROM complaints WHERE status=?", (s,)
        ).fetchone()[0]

    dept_counts = conn.execute(
        "SELECT department, COUNT(*) as cnt FROM complaints GROUP BY department ORDER BY cnt DESC"
    ).fetchall()

    ai_status = get_classifier_status()
    conn.close()

    return render_template("admin_dashboard.html",
                           complaints=complaints, users=users,
                           stats=stats, dept_counts=dept_counts,
                           status_list=STATUS_LIST,
                           departments=DEPARTMENTS,
                           ai_status=ai_status)

# ─── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
