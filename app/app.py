import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os
import sqlite3
import datetime
import io
import smtplib
import random
import re
import textwrap
import json
from email.mime.text import MIMEText
from flask import Flask, render_template, request, redirect, session, url_for, flash, send_file
import pickle
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)
app.secret_key = "simple-secret-key"
app.permanent_session_lifetime = datetime.timedelta(minutes=10)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "users.db")
model_path = os.path.join(BASE_DIR, "..", "model", "model.pkl")
model = pickle.load(open(model_path, "rb"))

# OTP email credentials from environment variables.
# Set EMAIL_ADDRESS and EMAIL_PASSWORD in your terminal before running the app.
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column_exists(conn, table_name, column_name, column_definition):
    columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    column_names = {column["name"] for column in columns}
    if column_name not in column_names:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")


def init_db():
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            last_prediction TEXT,
            graph_image TEXT,
            last_suggestion TEXT,
            last_weekly_plan TEXT,
            last_resume_score REAL DEFAULT NULL
        )
        """
    )

    ensure_column_exists(conn, "users", "last_prediction", "TEXT")
    ensure_column_exists(conn, "users", "graph_image", "TEXT")
    ensure_column_exists(conn, "users", "last_suggestion", "TEXT")
    ensure_column_exists(conn, "users", "last_weekly_plan", "TEXT")
    ensure_column_exists(conn, "users", "last_resume_score", "REAL DEFAULT NULL")
    ensure_column_exists(conn, "users", "email", "TEXT")
    ensure_column_exists(conn, "users", "reset_otp", "TEXT")
    ensure_column_exists(conn, "users", "otp_expiry", "TEXT")
    conn.execute(
        """
        UPDATE users
        SET email = 'user_' || id || '@placeholder.local'
        WHERE email IS NULL OR TRIM(email) = ''
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique
        ON users(email)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            cgpa REAL,
            aptitude REAL,
            coding REAL,
            communication REAL,
            projects REAL,
            internships REAL,
            result TEXT,
            chance REAL,
            suggestion TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    ensure_column_exists(conn, "prediction_history", "user_id", "INTEGER")
    ensure_column_exists(conn, "prediction_history", "date", "TEXT")
    ensure_column_exists(conn, "prediction_history", "cgpa", "REAL")
    ensure_column_exists(conn, "prediction_history", "aptitude", "REAL")
    ensure_column_exists(conn, "prediction_history", "coding", "REAL")
    ensure_column_exists(conn, "prediction_history", "communication", "REAL")
    ensure_column_exists(conn, "prediction_history", "projects", "REAL")
    ensure_column_exists(conn, "prediction_history", "internships", "REAL")
    ensure_column_exists(conn, "prediction_history", "result", "TEXT")
    ensure_column_exists(conn, "prediction_history", "chance", "REAL")
    ensure_column_exists(conn, "prediction_history", "suggestion", "TEXT")
    ensure_column_exists(conn, "prediction_history", "username", "TEXT")
    ensure_column_exists(conn, "prediction_history", "created_at", "TEXT")
    conn.execute(
        """
        UPDATE prediction_history
        SET created_at = date
        WHERE (created_at IS NULL OR created_at = '')
          AND date IS NOT NULL
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resume_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            suggestions TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    ensure_column_exists(conn, "resume_analysis", "user_id", "INTEGER")
    ensure_column_exists(conn, "resume_analysis", "suggestions", "TEXT")
    ensure_column_exists(conn, "resume_analysis", "created_at", "TEXT")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS preparation_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            selected_duration INTEGER NOT NULL,
            selected_focus_area TEXT NOT NULL,
            generated_plan TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    conn.commit()
    conn.close()


def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return user


def is_valid_password_strength(password):
    if len(password) < 6:
        return False
    if not re.search(r"[A-Za-z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    return True


def send_otp_email(email, otp):
    print(f"[OTP EMAIL] Starting send for: {email}")
    if not email or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        print(f"[OTP EMAIL] Invalid recipient email: {email}")
        return False
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("[OTP EMAIL] Missing EMAIL_ADDRESS or EMAIL_PASSWORD environment variable.")
        return False

    body = f"Your OTP is: {otp}\nThis OTP is valid for 10 minutes."
    message = MIMEText(body)
    message["Subject"] = "Password Reset OTP"
    message["From"] = EMAIL_ADDRESS
    message["To"] = email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, email, message.as_string())
        print(f"[OTP EMAIL] OTP sent successfully to: {email}")
        return True
    except smtplib.SMTPAuthenticationError as auth_error:
        print(f"[OTP EMAIL] Authentication error: {auth_error}")
    except (smtplib.SMTPConnectError, smtplib.SMTPServerDisconnected, TimeoutError, OSError) as conn_error:
        print(f"[OTP EMAIL] Connection error: {conn_error}")
    except smtplib.SMTPException as smtp_error:
        print(f"[OTP EMAIL] SMTP error: {smtp_error}")
    except Exception as unknown_error:
        print(f"[OTP EMAIL] Unexpected error: {unknown_error}")
    return False


@app.before_request
def session_timeout_handler():
    # Auto logout user after 10 minutes of inactivity.
    if "user_id" in session:
        session.permanent = True
        current_time = datetime.datetime.now()
        last_activity = session.get("last_activity")
        if last_activity:
            last_time = datetime.datetime.fromisoformat(last_activity)
            if current_time - last_time > app.permanent_session_lifetime:
                session.clear()
                flash("Session expired due to inactivity. Please login again.")
                return redirect(url_for("login"))
        session["last_activity"] = current_time.isoformat()


def get_latest_prediction_for_user(user_id):
    conn = get_db_connection()
    latest = conn.execute(
        """
        SELECT result, chance, suggestion
        FROM prediction_history
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (user_id,),
    ).fetchone()
    conn.close()
    return latest


def get_latest_prediction_scores_for_user(user_id):
    conn = get_db_connection()
    row = conn.execute(
        """
        SELECT cgpa, aptitude, coding, communication, projects, internships, COALESCE(created_at, date) AS created_at
        FROM prediction_history
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (user_id,),
    ).fetchone()
    conn.close()
    return row


def get_weak_areas(cgpa, coding, communication):
    weak_areas = []
    if cgpa < 7:
        weak_areas.append("Academics")
    if coding < 70:
        weak_areas.append("Coding")
    if communication < 70:
        weak_areas.append("Communication")
    return weak_areas


def get_strong_areas(cgpa, aptitude, coding, communication, projects):
    strong = []
    if cgpa is not None and cgpa >= 7.5:
        strong.append("Academics")
    if aptitude is not None and aptitude >= 75:
        strong.append("Aptitude")
    if coding is not None and coding >= 75:
        strong.append("Coding")
    if communication is not None and communication >= 75:
        strong.append("Communication")
    if projects is not None and projects >= 2:
        strong.append("Projects")
    return strong


def build_weak_area_text(weak_areas):
    if not weak_areas:
        return "No major weak areas detected."
    return ", ".join(weak_areas)


def build_strong_area_text(strong_areas):
    if not strong_areas:
        return "No major strong areas detected yet. Keep improving consistently."
    return ", ".join(strong_areas)


def build_suggestion_and_weekly_plan(cgpa, coding, communication, projects):
    weak_areas = get_weak_areas(cgpa, coding, communication)
    is_strong_profile = len(weak_areas) == 0 and projects >= 2

    if is_strong_profile:
        suggestion = (
            "Your profile is strong. Focus on mock interviews, coding contests, and system design basics. "
            "Keep building projects and sharpen interview communication."
        )
        plan = [
            "Mon: Solve 3 medium/hard problems on LeetCode and review one system design topic.",
            "Tue: Take a full mock interview on Pramp and improve your answers.",
            "Wed: Join one coding contest on CodeChef or HackerRank.",
            "Thu: Build one project feature and push clean updates to GitHub.",
            "Fri: Revise aptitude and complete one final mock interview round.",
        ]
    else:
        suggestion_parts = []
        if "Academics" in weak_areas:
            suggestion_parts.append(
                "Academics needs work. Revise one core subject daily and target a better CGPA."
            )
        if "Coding" in weak_areas:
            suggestion_parts.append(
                "Coding is weak. Practice consistently on LeetCode, HackerRank, and CodeChef."
            )
        if "Communication" in weak_areas:
            suggestion_parts.append(
                "Communication needs improvement. Use Pramp mock interviews and daily speaking practice."
            )
        if projects < 2:
            suggestion_parts.append(
                "Build stronger projects and keep your GitHub profile active."
            )

        suggestion = " ".join(suggestion_parts)
        plan = []
        if "Coding" in weak_areas:
            plan.append("Mon: Solve 3 LeetCode problems and 1 timed HackerRank challenge.")
        if "Communication" in weak_areas:
            plan.append("Tue: Practice mock interview on Pramp and improve confidence.")
        if "Academics" in weak_areas:
            plan.append("Wed: Revise one academic topic and practice aptitude questions.")
        plan.append("Thu: Improve one project module and update GitHub README.")
        plan.append("Fri: Mixed revision of coding, communication, and interview questions.")

    return weak_areas, suggestion, plan


def analyze_resume_text(resume_text):
    text = resume_text.lower()
    required_skills = ["python", "java", "sql", "ml", "html", "css"]
    detected_skills = [skill.upper() for skill in required_skills if skill in text]

    has_projects = "project" in text
    has_experience = "internship" in text or "experience" in text
    has_achievements = "achievement" in text or "award" in text

    skill_score = round((len(detected_skills) / len(required_skills)) * 30, 2)
    project_score = 25 if has_projects else 0
    experience_score = 25 if has_experience else 0
    achievement_score = 20 if has_achievements else 0
    resume_score = round(skill_score + project_score + experience_score + achievement_score, 2)

    missing_sections = []
    if len(detected_skills) < len(required_skills):
        missing_sections.append("Skills")
    if not has_projects:
        missing_sections.append("Projects")
    if not has_experience:
        missing_sections.append("Experience")
    if not has_achievements:
        missing_sections.append("Achievements")

    improvement_tips = []
    if "Skills" in missing_sections:
        improvement_tips.append("Add key skills like Python, Java, SQL, ML, HTML, and CSS.")
    if "Projects" in missing_sections:
        improvement_tips.append("Add a Projects section with project title, tech stack, and impact.")
    if "Experience" in missing_sections:
        improvement_tips.append("Add internship or work experience with clear responsibilities.")
    if "Achievements" in missing_sections:
        improvement_tips.append("Add achievements or awards to make your resume stand out.")
    if not improvement_tips:
        improvement_tips.append("Great resume structure. Add quantified achievements for even better impact.")

    return {
        "resume_score": resume_score,
        "missing_sections": missing_sections,
        "improvement_tips": improvement_tips,
        "detected_skills": detected_skills,
    }


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)
    return "\n".join(pages_text).strip()


def get_latest_resume_analysis_for_user(user_id):
    conn = get_db_connection()
    latest_resume = conn.execute(
        """
        SELECT suggestions, created_at
        FROM resume_analysis
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (user_id,),
    ).fetchone()
    conn.close()
    return latest_resume


def get_preparation_plans_for_user(user_id, limit=10):
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT id, selected_duration, selected_focus_area, created_at
        FROM preparation_plans
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, int(limit)),
    ).fetchall()
    conn.close()
    return rows


def get_preparation_plan_by_id(user_id, plan_id):
    conn = get_db_connection()
    row = conn.execute(
        """
        SELECT id, selected_duration, selected_focus_area, generated_plan, created_at
        FROM preparation_plans
        WHERE user_id = ? AND id = ?
        """,
        (user_id, plan_id),
    ).fetchone()
    conn.close()
    return row


def get_latest_preparation_plan_for_user(user_id):
    conn = get_db_connection()
    row = conn.execute(
        """
        SELECT id, selected_duration, selected_focus_area, generated_plan, created_at
        FROM preparation_plans
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (user_id,),
    ).fetchone()
    conn.close()
    return row


def _hours_split(total_hours, weights):
    """
    Split total_hours into len(weights) parts, rounded to 0.5h.
    Ensures sum(parts) == total_hours (within rounding).
    """
    if total_hours <= 0:
        return [0 for _ in weights]
    w_sum = sum(weights) if sum(weights) > 0 else 1
    raw = [(total_hours * (w / w_sum)) for w in weights]
    parts = [max(0.5, round(x * 2) / 2) for x in raw]
    diff = round((total_hours - sum(parts)) * 2) / 2
    # Adjust by 0.5h steps to match total.
    idx = 0
    while abs(diff) >= 0.5 and idx < 50:
        j = idx % len(parts)
        if diff > 0:
            parts[j] += 0.5
            diff -= 0.5
        else:
            if parts[j] - 0.5 >= 0.5:
                parts[j] -= 0.5
                diff += 0.5
        idx += 1
    return parts


def generate_weekly_preparation_plan(duration_weeks, focus_area):
    duration_weeks = int(duration_weeks)
    if duration_weeks not in (4, 6):
        duration_weeks = 4

    focus_area = (focus_area or "mixed").strip().lower()
    allowed_focus = {"aptitude", "coding", "communication", "core", "mixed"}
    if focus_area not in allowed_focus:
        focus_area = "mixed"

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

    # Realistic hours: steady + slightly increasing, with Saturday a bit heavier.
    if duration_weeks == 4:
        base_hours = [3.0, 3.5, 4.0, 4.5]
    else:
        base_hours = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

    week_stage = {
        1: ("Fundamentals", "Build foundations and consistency"),
        2: ("Core Practice", "Strengthen basics with timed practice"),
        3: ("Problem Solving", "Improve speed + accuracy with patterns"),
        4: ("Interview Readiness", "Mocks + revision + confidence building"),
        5: ("Advanced Prep", "Harder sets + deeper concepts + mocks"),
        6: ("Final Sprint", "Full mocks + weak-area fixes + polish"),
    }

    coding_curriculum = {
        "Fundamentals": ["Time/Space Complexity", "Arrays", "Strings", "Two Pointers", "Hashing Basics"],
        "Core Practice": ["Linked List", "Stack/Queue", "Sliding Window", "Binary Search", "Sorting Patterns"],
        "Problem Solving": ["Trees (BT/BST)", "Heaps", "Greedy Basics", "Recursion/Backtracking", "Graph Basics"],
        "Interview Readiness": ["Dynamic Programming Basics", "Graph Traversals", "Company Patterns", "Mock Interviews", "Revision Sets"],
        "Advanced Prep": ["DP Patterns", "Advanced Graphs", "Hard Arrays/Strings", "System Design Lite", "CS Fundamentals Review"],
        "Final Sprint": ["Full-length Mock Coding", "Speed Drills", "Revision + Notes", "Behavioral + HR", "Project Polishing"],
    }

    aptitude_curriculum = {
        "Fundamentals": ["Percentages", "Ratio/Proportion", "Average", "Number System", "Grammar Basics"],
        "Core Practice": ["Time & Work", "Speed/Distance", "Profit & Loss", "Blood Relations", "Reading Comprehension"],
        "Problem Solving": ["Permutation/Combination", "Probability", "Data Interpretation", "Seating Arrangement", "Critical Reasoning"],
        "Interview Readiness": ["Mixed Timed Sets", "Error Log Revision", "Mock Aptitude Test", "Verbal Sprint", "Weak-topic Fixes"],
        "Advanced Prep": ["Advanced DI", "Puzzle Sets", "High Difficulty Quant", "Verbal + Logic Mix", "Full Sectionals"],
        "Final Sprint": ["Full-length Aptitude Mocks", "Speed/Accuracy Drills", "Revision + Formula Sheet", "Weak-area Sets", "Stress-test Practice"],
    }

    communication_curriculum = {
        "Fundamentals": ["Self Introduction", "Basic HR Questions", "Communication Warm-up", "Listening + Clarity", "Confidence Building"],
        "Core Practice": ["STAR Stories", "Project Explanation", "GD Basics", "Tone + Pace", "Email/LinkedIn Basics"],
        "Problem Solving": ["Mock HR Round", "Case Questions", "Negotiation Basics", "Handling Pressure", "Communication Feedback"],
        "Interview Readiness": ["Mock Interviews", "Resume Walkthrough", "Strength/Weakness", "Why Company/Role", "Closing Questions"],
        "Advanced Prep": ["Mock + Feedback Loop", "Cross-questioning", "Leadership Examples", "Conflict Questions", "Polish & Presence"],
        "Final Sprint": ["Daily Mock/Practice", "HR Rapid-fire", "Confidence Drills", "Final Resume/Portfolio", "Interview Day Routine"],
    }

    core_curriculum = {
        "Fundamentals": ["OOP Basics", "DBMS Basics", "OS Basics", "CN Basics", "SQL Practice"],
        "Core Practice": ["OOP Design", "Normalization/Indexing", "Processes/Threads", "TCP/UDP", "SQL Joins/Queries"],
        "Problem Solving": ["Deadlocks/Scheduling", "Transactions/ACID", "HTTP + DNS", "Design Principles", "SQL Case Sets"],
        "Interview Readiness": ["Core Subject Viva", "Revision Notes", "SQL Timed Tests", "Mock Core Interview", "Mixed Q&A"],
        "Advanced Prep": ["Concurrency Deep Dive", "Query Optimization", "Network Troubleshooting", "OOP Patterns", "System Basics"],
        "Final Sprint": ["Core Rapid Revision", "Mixed Mock Q&A", "SQL + DBMS Sprint", "OS/CN Flashcards", "Final Weak-area Fixes"],
    }

    resource_map = {
        "coding": ["LeetCode", "HackerRank", "CodeChef", "GeeksforGeeks", "Mock interview"],
        "aptitude": ["Aptitude practice", "Timed sectionals", "Mock aptitude test"],
        "communication": ["HR questions", "Mock interview", "GD practice", "Speaking practice"],
        "core": ["Core notes", "Previous questions", "SQL practice", "Mock interview"],
        "mixed": ["LeetCode", "Aptitude practice", "HR questions", "Project work", "Mock interview"],
    }

    def pick_topics(stage_name):
        if focus_area == "coding":
            return coding_curriculum[stage_name]
        if focus_area == "aptitude":
            return aptitude_curriculum[stage_name]
        if focus_area == "communication":
            return communication_curriculum[stage_name]
        if focus_area == "core":
            return core_curriculum[stage_name]
        # mixed: blend
        return [
            coding_curriculum[stage_name][0],
            aptitude_curriculum[stage_name][0],
            communication_curriculum[stage_name][0],
            core_curriculum[stage_name][0],
        ]

    def focus_weights():
        if focus_area == "coding":
            return ("Coding", [0.75, 0.25])
        if focus_area == "aptitude":
            return ("Aptitude", [0.7, 0.3])
        if focus_area == "communication":
            return ("Communication", [0.7, 0.3])
        if focus_area == "core":
            return ("Core Subjects", [0.7, 0.3])
        return ("Mixed", [0.4, 0.25, 0.2, 0.15])

    plan = {
        "duration_weeks": duration_weeks,
        "focus_area": focus_area,
        "weeks": [],
        "notes": "Sunday is optional for rest + light revision. Keep weekdays consistent and track errors in a notebook.",
    }

    for w in range(1, duration_weeks + 1):
        stage_name = week_stage.get(w, week_stage[4])[0]
        stage_tag, stage_goal = week_stage.get(w, week_stage[4])
        topics_pool = pick_topics(stage_name)

        week_obj = {"week": w, "stage": stage_tag, "goal": stage_goal, "days": []}
        for d_i, day in enumerate(day_names):
            total = base_hours[w - 1] + (0.5 if day == "Saturday" else 0.0)
            total = float(total)

            if focus_area == "mixed":
                # 4 blocks: coding/aptitude/communication/core
                blocks = ["Coding", "Aptitude", "Communication", "Core"]
                split = _hours_split(total, focus_weights()[1])
                topics = [
                    {"topic": blocks[0] + ": " + topics_pool[0], "hours": split[0]},
                    {"topic": blocks[1] + ": " + topics_pool[1], "hours": split[1]},
                    {"topic": blocks[2] + ": " + topics_pool[2], "hours": split[2]},
                    {"topic": blocks[3] + ": " + topics_pool[3], "hours": split[3]},
                ]
                practice = [
                    {"type": "Timed practice", "style": "LeetCode" if split[0] >= 1 else "HackerRank"},
                    {"type": "Sectional practice", "style": "Aptitude practice"},
                    {"type": "Speaking/HR", "style": "HR questions"},
                    {"type": "Concept Q&A", "style": "Core notes"},
                ]
            else:
                area_label, weights = focus_weights()
                split = _hours_split(total, weights)
                primary_topic = topics_pool[(w + d_i) % len(topics_pool)]
                secondary_topic = topics_pool[(w + d_i + 1) % len(topics_pool)]
                topics = [
                    {"topic": f"{area_label}: {primary_topic}", "hours": split[0]},
                    {"topic": f"{area_label}: {secondary_topic} (practice set)", "hours": split[1]},
                ]
                styles = resource_map.get(focus_area, resource_map["mixed"])
                practice = [
                    {"type": "Topic learning + notes", "style": styles[0]},
                    {"type": "Timed practice", "style": styles[1] if len(styles) > 1 else styles[0]},
                ]

            # Add week-specific interview readiness items progressively.
            extras = []
            if w >= max(3, duration_weeks - 2):
                extras.append({"type": "Mock", "style": "Mock interview"})
            if w >= 2 and focus_area in ("coding", "mixed"):
                extras.append({"type": "Revision", "style": "Error-log review"})
            if w >= 2 and focus_area in ("communication", "mixed"):
                extras.append({"type": "HR prep", "style": "Answer framework (STAR)"})
            if w >= 3 and focus_area in ("mixed", "coding"):
                extras.append({"type": "Project", "style": "Project work"})

            # Keep Saturday as review + mock heavy.
            if day == "Saturday":
                extras = [{"type": "Weekly review", "style": "Revision + error log"}, {"type": "Mock", "style": "Mock interview"}] + extras

            day_obj = {
                "day": day,
                "total_hours": total,
                "topics": topics,
                "practice": practice + extras,
                "difficulty": "Beginner" if w == 1 else ("Intermediate" if w <= max(3, duration_weeks - 1) else "Advanced"),
            }
            week_obj["days"].append(day_obj)

        plan["weeks"].append(week_obj)

    return plan


def _clamp_int(value, min_v, max_v, default):
    try:
        iv = int(value)
    except Exception:
        return default
    return max(min_v, min(max_v, iv))


def _clamp_float(value, min_v, max_v, default=None):
    if value is None or value == "":
        return default
    try:
        fv = float(value)
    except Exception:
        return default
    return max(min_v, min(max_v, fv))


def _normalized_weakness(score, weak_threshold=70.0):
    """
    score: 0..100 where higher is better
    returns 0..1 where higher means weaker
    """
    if score is None:
        return 0.35
    score = max(0.0, min(100.0, float(score)))
    if score >= weak_threshold:
        return 0.15
    # If score is low, increase weakness
    return min(1.0, (weak_threshold - score) / weak_threshold + 0.25)


def generate_personalized_preparation_plan(duration_weeks, profile):
    """
    Personalized mixed plan (Mon-Sat) based on user's scores.
    duration_weeks: 1..12
    profile: dict with cgpa, aptitude, coding, communication, core(optional), projects(optional)
    """
    duration_weeks = _clamp_int(duration_weeks, 1, 12, 4)
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

    # Hours ramp: realistic, slightly increasing by weeks.
    # Week-1 starts ~3h/day, week-12 ends ~6h/day, Saturday +0.5h.
    def week_base_hours(w):
        if duration_weeks == 1:
            return 3.5
        return round(3.0 + (w - 1) * (3.0 / (duration_weeks - 1)), 1)  # 3.0 -> 6.0

    stages = [
        ("Fundamentals", "Build foundations and consistency"),
        ("Core Practice", "Strengthen basics with timed practice"),
        ("Problem Solving", "Improve speed + accuracy with patterns"),
        ("Advanced", "Push difficulty, fill gaps, and build stamina"),
        ("Mocks & Interview", "Mocks + revision + interview readiness"),
        ("Final Sprint", "Full mocks + weak-area fixes + polish"),
    ]

    coding_curriculum = {
        "Fundamentals": ["Time/Space Complexity", "Arrays", "Strings", "Two Pointers", "Hashing Basics"],
        "Core Practice": ["Linked List", "Stack/Queue", "Sliding Window", "Binary Search", "Sorting Patterns"],
        "Problem Solving": ["Trees (BT/BST)", "Heaps", "Greedy Basics", "Recursion/Backtracking", "Graph Basics"],
        "Advanced": ["DP Patterns", "Advanced Graphs", "Hard Arrays/Strings", "System Design Lite", "Company Patterns"],
        "Mocks & Interview": ["Mock Coding Round", "Revision Sets", "Company-wise Sheets", "CS Fundamentals Review", "Speed Drills"],
        "Final Sprint": ["Full-length Mock Coding", "Error-log Marathon", "Revision + Notes", "Project Polishing", "Interview Day Routine"],
    }

    aptitude_curriculum = {
        "Fundamentals": ["Percentages", "Ratio/Proportion", "Average", "Number System", "Grammar Basics"],
        "Core Practice": ["Time & Work", "Speed/Distance", "Profit & Loss", "Blood Relations", "Reading Comprehension"],
        "Problem Solving": ["Permutation/Combination", "Probability", "Data Interpretation", "Seating Arrangement", "Critical Reasoning"],
        "Advanced": ["Advanced DI", "Puzzle Sets", "High Difficulty Quant", "Verbal + Logic Mix", "Full Sectionals"],
        "Mocks & Interview": ["Mock Aptitude Test", "Mixed Timed Sets", "Error Log Revision", "Verbal Sprint", "Weak-topic Fixes"],
        "Final Sprint": ["Full-length Aptitude Mocks", "Speed/Accuracy Drills", "Revision + Formula Sheet", "Weak-area Sets", "Stress-test Practice"],
    }

    communication_curriculum = {
        "Fundamentals": ["Self Introduction", "Basic HR Questions", "Communication Warm-up", "Listening + Clarity", "Confidence Building"],
        "Core Practice": ["STAR Stories", "Project Explanation", "GD Basics", "Tone + Pace", "Resume Walkthrough"],
        "Problem Solving": ["Mock HR Round", "Handling Pressure", "Cross-questioning", "Behavioral Answers", "Communication Feedback"],
        "Advanced": ["Mock + Feedback Loop", "Leadership Examples", "Conflict Questions", "Negotiation Basics", "Presence & Delivery"],
        "Mocks & Interview": ["Mock Interviews", "Why Company/Role", "Strength/Weakness", "Closing Questions", "Follow-up Practice"],
        "Final Sprint": ["Daily Mock/Practice", "HR Rapid-fire", "Confidence Drills", "Final Resume/Portfolio", "Interview Day Routine"],
    }

    core_curriculum = {
        "Fundamentals": ["OOP Basics", "DBMS Basics", "OS Basics", "CN Basics", "SQL Practice"],
        "Core Practice": ["OOP Design", "Normalization/Indexing", "Processes/Threads", "TCP/UDP", "SQL Joins/Queries"],
        "Problem Solving": ["Deadlocks/Scheduling", "Transactions/ACID", "HTTP + DNS", "OOP Principles", "SQL Case Sets"],
        "Advanced": ["Concurrency Deep Dive", "Query Optimization", "Network Troubleshooting", "OOP Patterns", "System Basics"],
        "Mocks & Interview": ["Core Subject Viva", "Mixed Q&A", "SQL Timed Tests", "Mock Core Interview", "Revision Notes"],
        "Final Sprint": ["Core Rapid Revision", "Mixed Mock Q&A", "SQL + DBMS Sprint", "OS/CN Flashcards", "Final Weak-area Fixes"],
    }

    # Build weights from weakness (more weak -> more time).
    weaknesses = {
        "coding": _normalized_weakness(profile.get("coding")),
        "aptitude": _normalized_weakness(profile.get("aptitude")),
        "communication": _normalized_weakness(profile.get("communication")),
        "core": _normalized_weakness(profile.get("core")),
    }
    # Convert weakness to weights (ensure non-zero).
    raw = {k: max(0.12, v) for k, v in weaknesses.items()}
    total_w = sum(raw.values()) or 1.0
    weights = [raw["coding"] / total_w, raw["aptitude"] / total_w, raw["communication"] / total_w, raw["core"] / total_w]

    # Stage selection based on progress ratio.
    def stage_for_week(w):
        if duration_weeks == 1:
            return stages[0]
        p = (w - 1) / (duration_weeks - 1)
        idx = 0
        if p < 0.22:
            idx = 0
        elif p < 0.42:
            idx = 1
        elif p < 0.62:
            idx = 2
        elif p < 0.78:
            idx = 3
        elif p < 0.92:
            idx = 4
        else:
            idx = 5
        return stages[idx]

    plan = {
        "duration_weeks": duration_weeks,
        "focus_area": "personalized",
        "profile": {
            "cgpa": profile.get("cgpa"),
            "aptitude": profile.get("aptitude"),
            "coding": profile.get("coding"),
            "communication": profile.get("communication"),
            "core": profile.get("core"),
            "projects": profile.get("projects"),
        },
        "weights": {
            "coding": round(weights[0], 3),
            "aptitude": round(weights[1], 3),
            "communication": round(weights[2], 3),
            "core": round(weights[3], 3),
        },
        "weeks": [],
        "notes": "Sunday is optional for rest + light revision. Keep an error-log and revise it every Saturday.",
    }

    for w in range(1, duration_weeks + 1):
        stage_tag, stage_goal = stage_for_week(w)
        week_obj = {"week": w, "stage": stage_tag, "goal": stage_goal, "days": []}

        for d_i, day in enumerate(day_names):
            total_hours = float(week_base_hours(w) + (0.5 if day == "Saturday" else 0.0))
            split = _hours_split(total_hours, weights)

            coding_topic = coding_curriculum[stage_tag][(w + d_i) % len(coding_curriculum[stage_tag])]
            apt_topic = aptitude_curriculum[stage_tag][(w + d_i) % len(aptitude_curriculum[stage_tag])]
            comm_topic = communication_curriculum[stage_tag][(w + d_i) % len(communication_curriculum[stage_tag])]
            core_topic = core_curriculum[stage_tag][(w + d_i) % len(core_curriculum[stage_tag])]

            topics = [
                {"topic": f"Coding: {coding_topic}", "hours": split[0]},
                {"topic": f"Aptitude: {apt_topic}", "hours": split[1]},
                {"topic": f"Communication: {comm_topic}", "hours": split[2]},
                {"topic": f"Core: {core_topic}", "hours": split[3]},
            ]

            practice = [
                {"type": "Timed practice", "style": "LeetCode" if split[0] >= 1 else "HackerRank"},
                {"type": "Sectional practice", "style": "Aptitude practice"},
                {"type": "Speaking/HR", "style": "HR questions"},
                {"type": "Concept Q&A", "style": "Core notes"},
            ]

            # Progressive extras
            if stage_tag in ("Mocks & Interview", "Final Sprint") or w >= max(3, duration_weeks - 2):
                practice.append({"type": "Mock", "style": "Mock interview"})
            if w >= 2:
                practice.append({"type": "Revision", "style": "Error-log review"})
            if profile.get("projects") is not None and int(profile.get("projects") or 0) < 2 and w >= 2:
                practice.append({"type": "Project", "style": "Project work"})
            if day == "Saturday":
                practice = [
                    {"type": "Weekly review", "style": "Revision + error log"},
                    {"type": "Mock", "style": "Mock interview"},
                ] + practice

            difficulty = "Beginner" if w <= max(1, int(duration_weeks * 0.25)) else ("Intermediate" if w <= max(2, int(duration_weeks * 0.7)) else "Advanced")

            week_obj["days"].append(
                {
                    "day": day,
                    "total_hours": total_hours,
                    "topics": topics,
                    "practice": practice,
                    "difficulty": difficulty,
                }
            )

        plan["weeks"].append(week_obj)

    return plan


def summarize_preparation_plan_for_message(plan):
    if not plan or not isinstance(plan, dict):
        return "No preparation plan found yet."
    weeks = plan.get("weeks") or []
    if not weeks:
        return "No preparation plan found yet."
    w1 = weeks[0]
    day1 = (w1.get("days") or [{}])[0]
    topics = day1.get("topics") or []
    topic_text = ", ".join([t.get("topic", "") for t in topics if t.get("topic")][:3]).strip()
    hours = day1.get("total_hours")
    return f"{plan.get('duration_weeks')} weeks · {plan.get('focus_area', 'mixed').capitalize()} · Week 1 starts with ~{hours} hrs/day. Sample topics: {topic_text}."


init_db()


@app.route("/")
def home():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))
    return render_template(
        "index.html",
        username=user["username"],
        last_prediction=user["last_prediction"] or "No prediction yet",
        graph_image=None,
    )


@app.route("/career-toolkit/planner", methods=["POST"])
def career_toolkit_planner():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    duration = _clamp_int(request.form.get("duration_weeks"), 1, 12, 4)

    latest = get_latest_prediction_scores_for_user(user["id"])
    cgpa = _clamp_float(request.form.get("cgpa"), 0.0, 10.0, default=(latest["cgpa"] if latest else None))
    aptitude = _clamp_float(request.form.get("aptitude"), 0.0, 100.0, default=(latest["aptitude"] if latest else None))
    coding = _clamp_float(request.form.get("coding"), 0.0, 100.0, default=(latest["coding"] if latest else None))
    communication = _clamp_float(request.form.get("communication"), 0.0, 100.0, default=(latest["communication"] if latest else None))
    # If user doesn't provide core score, derive from CGPA lightly.
    core = _clamp_float(request.form.get("core"), 0.0, 100.0, default=(min(100.0, (cgpa or 7.0) * 10.0) if cgpa is not None else None))
    projects = _clamp_int(request.form.get("projects"), 0, 20, default=(latest["projects"] if latest else 0))

    profile = {
        "cgpa": cgpa,
        "aptitude": aptitude,
        "coding": coding,
        "communication": communication,
        "core": core,
        "projects": projects,
    }

    plan = generate_personalized_preparation_plan(duration, profile)
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO preparation_plans (user_id, selected_duration, selected_focus_area, generated_plan, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user["id"], int(plan["duration_weeks"]), "personalized", json.dumps(plan), created_at),
    )
    conn.commit()
    plan_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    conn.close()
    flash("Personalized weekly planner generated and saved.")
    return redirect(url_for("home", plan_id=plan_id))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        if not username or not email or not password:
            flash("Username, email, and password are required.")
            return redirect(url_for("signup"))

        if "@" not in email or "." not in email:
            flash("Please enter a valid email address.")
            return redirect(url_for("signup"))

        if not is_valid_password_strength(password):
            flash("Password must be at least 6 characters and include letters and numbers.")
            return redirect(url_for("signup"))

        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, hashed_password),
            )
            conn.commit()
            flash("Signup successful. Please login.")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.")
            return redirect(url_for("signup"))
        finally:
            conn.close()
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["last_activity"] = datetime.datetime.now().isoformat()
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.")
        return redirect(url_for("login"))
    return render_template("login.html")


@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        if not email:
            flash("Invalid email")
            return redirect(url_for("forgot_password"))
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            flash("Invalid email")
            return redirect(url_for("forgot_password"))

        conn = get_db_connection()
        user = conn.execute("SELECT id, email FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            conn.close()
            flash("Invalid email")
            return redirect(url_for("forgot_password"))

        otp = f"{random.randint(100000, 999999)}"
        expiry_time = datetime.datetime.now() + datetime.timedelta(minutes=10)
        expiry_str = expiry_time.strftime("%Y-%m-%d %H:%M:%S")

        conn.execute(
            "UPDATE users SET reset_otp = ?, otp_expiry = ? WHERE id = ?",
            (otp, expiry_str, user["id"]),
        )
        conn.commit()
        conn.close()

        session["reset_email"] = email
        email_sent = send_otp_email(email, otp)
        if email_sent:
            flash("OTP sent successfully")
            return redirect(url_for("verify_otp"))

        print(f"OTP for {email} is: {otp}")
        flash("Email sending failed. Check SMTP settings")
        return redirect(url_for("forgot_password"))

    return render_template("forgot_password.html")


@app.route("/verify_otp", methods=["GET", "POST"])
def verify_otp():
    reset_email = session.get("reset_email")
    if not reset_email:
        flash("Please request OTP first.")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        otp = request.form.get("otp", "").strip()
        if not otp:
            flash("OTP is required.")
            return redirect(url_for("verify_otp"))

        conn = get_db_connection()
        user = conn.execute(
            "SELECT id, reset_otp, otp_expiry FROM users WHERE email = ?",
            (reset_email,),
        ).fetchone()

        if not user or not user["reset_otp"] or not user["otp_expiry"]:
            conn.close()
            flash("Invalid OTP request. Please try again.")
            return redirect(url_for("forgot_password"))

        try:
            expiry_time = datetime.datetime.strptime(user["otp_expiry"], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            conn.close()
            flash("OTP data is invalid. Request OTP again.")
            return redirect(url_for("forgot_password"))

        if datetime.datetime.now() > expiry_time:
            conn.execute("UPDATE users SET reset_otp = NULL, otp_expiry = NULL WHERE id = ?", (user["id"],))
            conn.commit()
            conn.close()
            flash("OTP expired. Please request a new OTP.")
            return redirect(url_for("forgot_password"))

        if otp != user["reset_otp"]:
            conn.close()
            flash("Invalid OTP.")
            return redirect(url_for("verify_otp"))

        conn.close()
        session["otp_verified"] = True
        flash("OTP verified. Please set a new password.")
        return redirect(url_for("reset_password"))

    return render_template("verify_otp.html")


@app.route("/reset_password", methods=["GET", "POST"])
def reset_password():
    reset_email = session.get("reset_email")
    otp_verified = session.get("otp_verified")
    if not reset_email or not otp_verified:
        flash("Please verify OTP first.")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        new_password = request.form.get("new_password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not new_password or not confirm_password:
            flash("Both password fields are required.")
            return redirect(url_for("reset_password"))

        if new_password != confirm_password:
            flash("Passwords do not match.")
            return redirect(url_for("reset_password"))

        if not is_valid_password_strength(new_password):
            flash("Password must be at least 6 characters and include letters and numbers.")
            return redirect(url_for("reset_password"))

        hashed_password = generate_password_hash(new_password)
        conn = get_db_connection()
        conn.execute(
            """
            UPDATE users
            SET password = ?, reset_otp = NULL, otp_expiry = NULL
            WHERE email = ?
            """,
            (hashed_password, reset_email),
        )
        conn.commit()
        conn.close()

        session.pop("reset_email", None)
        session.pop("otp_verified", None)
        flash("Password reset successful. Please login.")
        return redirect(url_for("login"))

    return render_template("reset_password.html")


@app.route("/logout")
def logout():
    # Ensure the performance graph never persists across sessions.
    if "user_id" in session:
        conn = get_db_connection()
        conn.execute("UPDATE users SET graph_image = NULL WHERE id = ?", (session["user_id"],))
        conn.commit()
        conn.close()
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    conn = get_db_connection()
    history = conn.execute(
        """
        SELECT id, COALESCE(created_at, date) AS created_at, cgpa, aptitude, coding, communication, projects, internships, result, chance
        FROM prediction_history
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (user["id"],),
    ).fetchall()
    latest_resume = conn.execute(
        """
        SELECT suggestions
        FROM resume_analysis
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (user["id"],),
    ).fetchone()
    conn.close()

    weak_areas_text = "No prediction yet"
    if history:
        weak_areas = get_weak_areas(history[0]["cgpa"], history[0]["coding"], history[0]["communication"])
        weak_areas_text = build_weak_area_text(weak_areas)

    latest_prep_plan = get_latest_preparation_plan_for_user(user["id"])

    resume_suggestions = []
    if latest_resume and latest_resume["suggestions"]:
        resume_suggestions = [item for item in latest_resume["suggestions"].split("||") if item]

    return render_template(
        "dashboard.html",
        username=user["username"],
        last_prediction=user["last_prediction"] or "No prediction yet",
        last_suggestion=user["last_suggestion"] or "No suggestions yet",
        weak_areas=weak_areas_text,
        resume_score=user["last_resume_score"],
        latest_prep_plan=latest_prep_plan,
        graph_image=None,
        history=history,
        resume_suggestions=resume_suggestions,
    )


@app.route("/preparation-planner", methods=["GET", "POST"])
def preparation_planner():
    # Kept for backward-compatibility. Planner now lives inside Career Toolkit.
    return redirect(url_for("home"))


@app.route("/preparation-planner/<int:plan_id>")
def view_preparation_plan(plan_id):
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    row = get_preparation_plan_by_id(user["id"], plan_id)
    if not row:
        flash("Plan not found.")
        return redirect(url_for("home"))

    try:
        plan = json.loads(row["generated_plan"]) if row["generated_plan"] else None
    except Exception:
        plan = None

    # Viewer kept for backward-compatibility; show inside Career Toolkit.
    return redirect(url_for("home", plan_id=row["id"]) + "#weekly-prep-planner")


@app.route("/download_report")
@app.route("/export-report")
def download_report():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    latest_prediction = get_latest_prediction_for_user(user["id"])
    placement_chance = f"{latest_prediction['chance']}%" if latest_prediction else "Not available"
    last_result = latest_prediction["result"] if latest_prediction else "No prediction yet"

    weak_areas_text = "No prediction yet"
    if latest_prediction:
        conn = get_db_connection()
        latest_row = conn.execute(
            """
            SELECT cgpa, coding, communication
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (user["id"],),
        ).fetchone()
        conn.close()
        if latest_row:
            weak_areas = get_weak_areas(latest_row["cgpa"], latest_row["coding"], latest_row["communication"])
            weak_areas_text = build_weak_area_text(weak_areas)

    suggestion_text = user["last_suggestion"] or "No suggestions yet"
    latest_plan_row = get_latest_preparation_plan_for_user(user["id"])
    prep_plan_summary = "No preparation plan yet"
    if latest_plan_row and latest_plan_row["generated_plan"]:
        try:
            prep_plan_summary = summarize_preparation_plan_for_message(json.loads(latest_plan_row["generated_plan"]))
        except Exception:
            prep_plan_summary = "Preparation plan found but could not be summarized."

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    y = 800

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Placement Prediction Report")
    y -= 30

    pdf.setFont("Helvetica", 11)
    lines = [
        f"Username: {user['username']}",
        f"Last Prediction: {last_result}",
        f"Placement Chance: {placement_chance}",
        f"Weak Areas: {weak_areas_text}",
        f"Suggestions: {suggestion_text}",
        f"Preparation Planner: {prep_plan_summary}",
    ]

    for line in lines:
        pdf.drawString(50, y, line[:120])
        y -= 22

    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{user['username']}_placement_report.pdf",
        mimetype="application/pdf",
    )


@app.route("/send_email", methods=["POST"])
@app.route("/send-email", methods=["POST"])
def send_email():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    recipient_email = request.form.get("email", "").strip()
    if not recipient_email:
        flash("Please enter an email address.")
        return redirect(url_for("dashboard"))

    # Keep SMTP setup simple. Set these environment variables before using this feature.
    smtp_email = os.environ.get("SMTP_EMAIL")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    if not smtp_email or not smtp_password:
        flash("Email settings missing. Set SMTP_EMAIL and SMTP_PASSWORD.")
        return redirect(url_for("dashboard"))

    suggestion = user["last_suggestion"] or "No suggestions yet."
    latest_plan_row = get_latest_preparation_plan_for_user(user["id"])
    prep_plan_summary = "No preparation plan yet."
    if latest_plan_row and latest_plan_row["generated_plan"]:
        try:
            prep_plan_summary = summarize_preparation_plan_for_message(json.loads(latest_plan_row["generated_plan"]))
        except Exception:
            prep_plan_summary = "Preparation plan found but could not be summarized."
    body = (
        f"Hello {user['username']},\n\n"
        f"Here are your latest placement suggestions:\n{suggestion}\n\n"
        f"Weekly Preparation Planner:\n{prep_plan_summary}\n"
    )

    message = MIMEText(body)
    message["Subject"] = "Your Placement Suggestions & Preparation Plan"
    message["From"] = smtp_email
    message["To"] = recipient_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(smtp_email, smtp_password)
        server.sendmail(smtp_email, recipient_email, message.as_string())
        server.quit()
        flash("Suggestion email sent successfully.")
    except Exception:
        flash("Could not send email. Please verify SMTP credentials.")

    return redirect(url_for("dashboard"))


@app.route("/admin")
def admin():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    conn = get_db_connection()
    total_users = conn.execute("SELECT COUNT(*) AS total FROM users").fetchone()["total"]
    total_predictions = conn.execute("SELECT COUNT(*) AS total FROM prediction_history").fetchone()["total"]
    all_users = conn.execute(
        """
        SELECT id, username
        FROM users
        ORDER BY id DESC
        """
    ).fetchall()
    all_predictions = conn.execute(
        """
        SELECT p.id, COALESCE(p.username, u.username) AS username, p.user_id, p.result, p.chance, COALESCE(p.created_at, p.date) AS created_at
        FROM prediction_history p
        LEFT JOIN users u ON p.user_id = u.id
        ORDER BY p.id DESC
        LIMIT 100
        """
    ).fetchall()
    top_users = conn.execute(
        """
        SELECT u.username, MAX(p.chance) AS highest_chance
        FROM users u
        JOIN prediction_history p ON p.user_id = u.id
        GROUP BY u.id, u.username
        ORDER BY highest_chance DESC
        LIMIT 10
        """
    ).fetchall()
    conn.close()

    return render_template(
        "admin.html",
        username=user["username"],
        total_users=total_users,
        total_predictions=total_predictions,
        all_users=all_users,
        all_predictions=all_predictions,
        top_users=top_users,
    )


@app.route("/leaderboard")
def leaderboard():
    # Leaderboard section removed from the product.
    return redirect(url_for("dashboard"))


@app.route("/profile")
def profile():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    conn = get_db_connection()
    profile_stats = conn.execute(
        """
        SELECT COUNT(*) AS total, AVG(chance) AS avg_chance, MAX(chance) AS max_chance
        FROM prediction_history
        WHERE user_id = ?
        """,
        (user["id"],),
    ).fetchone()
    latest_prediction = conn.execute(
        """
        SELECT result, chance, cgpa, coding, communication
        FROM prediction_history
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (user["id"],),
    ).fetchone()
    conn.close()

    last_prediction_text = "No prediction yet"
    weak_areas_text = "No prediction yet"
    if latest_prediction:
        last_prediction_text = f"{latest_prediction['result']} | Chance: {latest_prediction['chance']}%"
        weak_areas = get_weak_areas(
            latest_prediction["cgpa"],
            latest_prediction["coding"],
            latest_prediction["communication"],
        )
        weak_areas_text = build_weak_area_text(weak_areas)

    return render_template(
        "profile.html",
        username=user["username"],
        total_predictions=profile_stats["total"] or 0,
        average_chance=round(profile_stats["avg_chance"], 2) if profile_stats["avg_chance"] is not None else None,
        highest_chance=profile_stats["max_chance"],
        last_prediction=last_prediction_text,
        resume_score=user["last_resume_score"],
        weak_areas=weak_areas_text,
    )


@app.route("/download-resume-report")
def download_resume_report():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    latest_resume = get_latest_resume_analysis_for_user(user["id"])
    suggestions = []
    if latest_resume and latest_resume["suggestions"]:
        suggestions = [item for item in latest_resume["suggestions"].split("||") if item]

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    y = 800

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Resume Score Report")
    y -= 30

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"Username: {user['username']}")
    y -= 22
    resume_score_text = (
        f"{user['last_resume_score']}/100"
        if user["last_resume_score"] is not None
        else "Not analyzed yet"
    )
    pdf.drawString(50, y, f"Resume Score: {resume_score_text}")
    y -= 28

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Resume Suggestions:")
    y -= 20
    pdf.setFont("Helvetica", 11)

    if not suggestions:
        suggestions = ["No resume suggestions available yet. Upload a resume first."]

    for suggestion in suggestions:
        wrapped_lines = textwrap.wrap(f"- {suggestion}", width=95)
        for line in wrapped_lines:
            if y < 60:
                pdf.showPage()
                y = 800
                pdf.setFont("Helvetica", 11)
            pdf.drawString(50, y, line)
            y -= 18

    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{user['username']}_resume_report.pdf",
        mimetype="application/pdf",
    )


@app.route("/companies")
def companies():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    return render_template("companies.html", username=user["username"])


@app.route("/delete-history/<int:history_id>", methods=["POST"])
@app.route("/delete_history/<int:history_id>", methods=["POST"])
def delete_history(history_id):
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    conn = get_db_connection()
    conn.execute("DELETE FROM prediction_history WHERE id = ? AND user_id = ?", (history_id, user["id"]))
    conn.commit()
    conn.close()
    flash("History record deleted.")
    return redirect(url_for("dashboard"))


@app.route("/clear-history", methods=["POST"])
@app.route("/clear_history", methods=["POST"])
def clear_history():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    conn = get_db_connection()
    conn.execute("DELETE FROM prediction_history WHERE user_id = ?", (user["id"],))
    conn.commit()
    conn.close()
    flash("All history cleared.")
    return redirect(url_for("dashboard"))


@app.route("/predict", methods=["POST"])
def predict():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    try:
        features = [
            float(request.form["semester"]),
            float(request.form["cgpa"]),
            float(request.form["attendance"]),
            float(request.form["aptitude"]),
            float(request.form["coding"]),
            float(request.form["communication"]),
            float(request.form["projects"]),
            float(request.form["internships"]),
        ]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        probability = model.predict_proba(final_features)[0][1]
        chance = round(probability * 100, 2)
        output = "Placed" if prediction[0] == 1 else "Not Placed"

        labels = ["CGPA", "Aptitude", "Coding", "Communication"]
        user_scores = [features[1], features[3], features[4], features[5]]
        ideal_scores = [8, 80, 80, 75]

        plt.figure(figsize=(7, 4.5))
        plt.plot(labels, user_scores, marker="o", linewidth=2.5, color="#6366f1", label="Your Score")
        plt.plot(labels, ideal_scores, marker="o", linewidth=2, linestyle="--", color="#22c55e", label="Ideal Score")
        plt.legend()
        plt.title("Placement Readiness Comparison")
        plt.xlabel("Parameters")
        plt.ylabel("Score")
        plt.ylim(0, 100)
        plt.grid(True, linestyle="--", alpha=0.5)

        graph_dir = os.path.join(BASE_DIR, "static", "images")
        os.makedirs(graph_dir, exist_ok=True)
        graph_filename = f"graph_user_{user['id']}.png"
        graph_path = os.path.join(graph_dir, graph_filename)
        plt.savefig(graph_path)
        plt.close()
        relative_graph_path = f"images/{graph_filename}"

        weak_areas, suggestion, weekly_plan = build_suggestion_and_weekly_plan(
            cgpa=features[1],
            coding=features[4],
            communication=features[5],
            projects=features[6],
        )
        weekly_plan_db = "||".join(weekly_plan)

        strong_areas = get_strong_areas(
            cgpa=features[1],
            aptitude=features[3],
            coding=features[4],
            communication=features[5],
            projects=features[6],
        )

        conn = get_db_connection()
        conn.execute(
            """
            UPDATE users
            SET last_prediction = ?, graph_image = NULL, last_suggestion = ?, last_weekly_plan = ?
            WHERE id = ?
            """,
            (f"{output} | Chance: {chance}%", suggestion, weekly_plan_db, user["id"]),
        )

        conn.execute(
            """
            INSERT INTO prediction_history
            (user_id, username, date, created_at, cgpa, aptitude, coding, communication, projects, internships, result, chance, suggestion)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                user["username"],
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                features[1],
                features[3],
                features[4],
                features[5],
                features[6],
                features[7],
                output,
                chance,
                suggestion,
            ),
        )
        conn.commit()
        conn.close()

        return render_template(
            "index.html",
            username=user["username"],
            prediction_text=output,
            chance=chance,
            suggestion=suggestion,
            weak_areas=build_weak_area_text(weak_areas),
            strong_areas=build_strong_area_text(strong_areas),
            weekly_plan=weekly_plan,
            graph_image=relative_graph_path,
            last_prediction=f"{output} | Chance: {chance}%",
        )
    except Exception as exc:
        return str(exc)


@app.route("/weekly-planner", methods=["GET", "POST"])
def weekly_planner():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    latest = get_latest_prediction_scores_for_user(user["id"])
    plans = get_preparation_plans_for_user(user["id"], limit=10)

    if request.method == "POST":
        if not latest:
            flash("Please generate a placement prediction first. Planner uses your latest scores.")
            return redirect(url_for("home"))

        duration = _clamp_int(request.form.get("duration_weeks"), 1, 12, 1)
        profile = {
            "cgpa": latest["cgpa"],
            "aptitude": latest["aptitude"],
            "coding": latest["coding"],
            "communication": latest["communication"],
            "core": min(100.0, float(latest["cgpa"]) * 10.0) if latest["cgpa"] is not None else None,
            "projects": int(latest["projects"] or 0),
        }
        plan = generate_personalized_preparation_plan(duration, profile)
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = get_db_connection()
        conn.execute(
            """
            INSERT INTO preparation_plans (user_id, selected_duration, selected_focus_area, generated_plan, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user["id"], int(plan["duration_weeks"]), "personalized", json.dumps(plan), created_at),
        )
        conn.commit()
        plan_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
        conn.close()
        flash("Weekly plan generated and saved.")
        return redirect(url_for("weekly_planner_view", plan_id=plan_id))

    return render_template("weekly_planner.html", username=user["username"], latest_scores=latest, plans=plans, plan=None, selected_plan_meta=None)


@app.route("/weekly-planner/<int:plan_id>")
def weekly_planner_view(plan_id):
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    latest = get_latest_prediction_scores_for_user(user["id"])
    plans = get_preparation_plans_for_user(user["id"], limit=10)
    row = get_preparation_plan_by_id(user["id"], plan_id)
    if not row:
        flash("Plan not found.")
        return redirect(url_for("weekly_planner"))

    try:
        plan = json.loads(row["generated_plan"]) if row["generated_plan"] else None
    except Exception:
        plan = None

    return render_template(
        "weekly_planner.html",
        username=user["username"],
        latest_scores=latest,
        plans=plans,
        plan=plan,
        selected_plan_meta={
            "id": row["id"],
            "created_at": row["created_at"],
            "selected_duration": row["selected_duration"],
            "selected_focus_area": row["selected_focus_area"],
        },
    )


@app.route("/resume-analyzer", methods=["GET", "POST"])
def resume_analyzer():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    suggestions = []
    missing_sections = []
    detected_skills = []
    resume_text = ""
    resume_score = user["last_resume_score"]

    if request.method == "POST":
        uploaded_file = request.files.get("resume_pdf")
        if uploaded_file and uploaded_file.filename:
            filename = secure_filename(uploaded_file.filename)
            if not filename.lower().endswith(".pdf"):
                flash("Please upload a PDF file only.")
            else:
                try:
                    resume_text = extract_text_from_pdf(uploaded_file)
                    if resume_text:
                        analysis_data = analyze_resume_text(resume_text)
                        suggestions = analysis_data["improvement_tips"]
                        missing_sections = analysis_data["missing_sections"]
                        detected_skills = analysis_data["detected_skills"]
                        resume_score = analysis_data["resume_score"]
                        suggestions_db = "||".join(suggestions)
                        conn = get_db_connection()
                        conn.execute(
                            """
                            INSERT INTO resume_analysis (user_id, suggestions, created_at)
                            VALUES (?, ?, ?)
                            """,
                            (
                                user["id"],
                                suggestions_db,
                                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            ),
                        )
                        conn.execute("UPDATE users SET last_resume_score = ? WHERE id = ?", (resume_score, user["id"]))
                        conn.commit()
                        conn.close()
                    else:
                        flash("Could not extract text from this PDF.")
                except Exception:
                    flash("Error while reading PDF. Please try another file.")
        else:
            flash("Please upload your resume in PDF format.")

    return render_template(
        "resume_analyzer.html",
        username=user["username"],
        resume_text=resume_text,
        suggestions=suggestions,
        missing_sections=missing_sections,
        detected_skills=detected_skills,
        resume_score=resume_score,
    )


if __name__ == "__main__":
    app.run(debug=True)