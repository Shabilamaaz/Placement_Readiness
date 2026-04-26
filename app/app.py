import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os
import sqlite3
import datetime
import io
import smtplib
import textwrap
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


def get_weak_areas(cgpa, coding, communication):
    weak_areas = []
    if cgpa < 7:
        weak_areas.append("Academics")
    if coding < 70:
        weak_areas.append("Coding")
    if communication < 70:
        weak_areas.append("Communication")
    return weak_areas


def build_weak_area_text(weak_areas):
    if not weak_areas:
        return "No major weak areas detected."
    return ", ".join(weak_areas)


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


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Username and password are required.")
            return redirect(url_for("signup"))

        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            flash("Signup successful. Please login.")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists.")
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

    weekly_plan = []
    if user["last_weekly_plan"]:
        weekly_plan = [item for item in user["last_weekly_plan"].split("||") if item]

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
        weekly_plan=weekly_plan,
        graph_image=None,
        history=history,
        resume_suggestions=resume_suggestions,
    )


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

    weekly_plan_text = user["last_weekly_plan"].replace("||", " | ") if user["last_weekly_plan"] else "No weekly plan yet"
    suggestion_text = user["last_suggestion"] or "No suggestions yet"

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
        f"Weekly Plan: {weekly_plan_text}",
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
    weekly_plan = user["last_weekly_plan"].replace("||", "\n- ") if user["last_weekly_plan"] else "No weekly plan yet."
    body = (
        f"Hello {user['username']},\n\n"
        f"Here are your latest placement suggestions:\n{suggestion}\n\n"
        f"Weekly Plan:\n- {weekly_plan}\n"
    )

    message = MIMEText(body)
    message["Subject"] = "Your Placement Suggestions"
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
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    conn = get_db_connection()
    top_performers = conn.execute(
        """
        SELECT u.username, MAX(p.chance) AS highest_chance
        FROM users u
        JOIN prediction_history p ON p.user_id = u.id
        GROUP BY u.id, u.username
        ORDER BY highest_chance DESC
        LIMIT 5
        """
    ).fetchall()
    conn.close()

    return render_template("leaderboard.html", username=user["username"], top_performers=top_performers)


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
            weekly_plan=weekly_plan,
            graph_image=relative_graph_path,
            last_prediction=f"{output} | Chance: {chance}%",
        )
    except Exception as exc:
        return str(exc)


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