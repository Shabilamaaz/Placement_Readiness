import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os
import sqlite3
import datetime
from flask import Flask, render_template, request, redirect, session, url_for, flash
import pickle
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader

app = Flask(__name__)
app.secret_key = "simple-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "users.db")
model_path = os.path.join(BASE_DIR, "..", "model", "model.pkl")
model = pickle.load(open(model_path, "rb"))

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


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
            last_weekly_plan TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            cgpa REAL,
            aptitude REAL,
            coding REAL,
            communication REAL,
            projects REAL,
            internships REAL,
            result TEXT,
            chance REAL,
            date TEXT NOT NULL
        )
        """
    )
    columns = conn.execute("PRAGMA table_info(prediction_history)").fetchall()
    column_names = {column["name"] for column in columns}
    required_columns = {
        "id",
        "username",
        "cgpa",
        "aptitude",
        "coding",
        "communication",
        "projects",
        "internships",
        "result",
        "chance",
        "date",
    }
    if column_names and column_names != required_columns:
        conn.execute("DROP TABLE IF EXISTS prediction_history")
        conn.execute(
            """
            CREATE TABLE prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                cgpa REAL,
                aptitude REAL,
                coding REAL,
                communication REAL,
                projects REAL,
                internships REAL,
                result TEXT,
                chance REAL,
                date TEXT NOT NULL
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


def build_weekly_plan(cgpa, coding, communication, projects):
    is_strong_profile = cgpa >= 8 and coding >= 80 and communication >= 70 and projects >= 2

    if is_strong_profile:
        suggestion = (
            "You have a strong profile. Move to advanced preparation: attend at least 2 mock interviews this week, "
            "join one coding contest, and build one real-world project feature that you can explain in interviews."
        )
        plan = [
            "Mon: Solve 3 medium/hard coding problems and write clean explanations for each approach.",
            "Tue: Take one mock interview (technical + HR round) and note your weak areas.",
            "Wed: Practice aptitude speed tests on HackerRank and revise frequently asked formulas.",
            "Thu: Build or improve one real-world project module and push updates to GitHub.",
            "Fri: Join a coding contest or timed challenge and review mistakes after completion.",
        ]
    else:
        suggestion_points = []
        if cgpa < 7:
            suggestion_points.append(
                "Improve academics by revising one core subject daily and solving 10 topic-wise questions."
            )
        if coding < 70:
            suggestion_points.append(
                "Practice 2 coding problems daily on LeetCode and 1 timed coding task on HackerRank."
            )
        if communication < 60:
            suggestion_points.append(
                "Practice mock interviews and daily speaking drills for 20 minutes to improve confidence."
            )
        if projects < 2:
            suggestion_points.append(
                "Build at least one practical mini-project and document problem statement, tech stack, and outcomes."
            )

        suggestion = " ".join(suggestion_points) + " Follow the weekly plan below consistently for steady improvement."
        plan = [
            "Mon: Coding - Solve 2 easy/medium LeetCode problems and revise one common pattern.",
            "Tue: Aptitude - Practice quant and logical reasoning sets on HackerRank for 45 minutes.",
            "Wed: Communication - Record 3 mock interview answers and improve clarity and body language.",
            "Thu: Projects - Build one feature in your project and update README with screenshots and impact.",
            "Fri: Revision - Mixed practice (coding + aptitude + HR questions) and track weekly progress.",
        ]

    return suggestion, plan


def analyze_resume_text(resume_text):
    text = resume_text.lower()
    tips = []

    if "project" not in text:
        tips.append("Add a dedicated Projects section with 2-3 projects, technologies used, and measurable outcomes.")
    if "skill" not in text and "skills" not in text:
        tips.append("Create a clear Skills section grouped by Programming Languages, Frameworks, Databases, and Tools.")
    if "experience" not in text and "internship" not in text and "work" not in text:
        tips.append("Add an Experience or Internship section, even for internships, freelancing, or college team roles.")
    if "achievement" not in text and "award" not in text and "certification" not in text:
        tips.append("Add achievements or certifications to show extra effort and domain knowledge.")
    if "responsibility" not in text and "impact" not in text and "%" not in text:
        tips.append("Write bullet points with impact numbers (for example: improved speed by 20% or reduced errors by 15%).")

    if not tips:
        tips.append("Your resume has all major sections. Next step: improve each bullet with action verbs and measurable impact.")
    return tips


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)
    return "\n".join(pages_text).strip()


init_db()


@app.route("/")
def home():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))
    return render_template(
        "index.html",
        last_prediction=user["last_prediction"] or "No prediction yet",
        graph_image=user["graph_image"],
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
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.")
        return redirect(url_for("login"))
    return render_template("login.html")


@app.route("/logout")
def logout():
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
        SELECT id, date, cgpa, aptitude, coding, communication, projects, internships, result, chance
        FROM prediction_history
        WHERE username = ?
        ORDER BY id DESC
        """
        ,
        (user["username"],),
    ).fetchall()
    conn.close()

    weekly_plan = []
    if user["last_weekly_plan"]:
        weekly_plan = [item for item in user["last_weekly_plan"].split("||") if item]

    return render_template(
        "dashboard.html",
        username=user["username"],
        last_prediction=user["last_prediction"] or "No prediction yet",
        last_suggestion=user["last_suggestion"] or "No suggestions yet",
        weekly_plan=weekly_plan,
        graph_image=user["graph_image"],
        history=history,
    )


@app.route("/delete-history/<int:history_id>", methods=["POST"])
def delete_history(history_id):
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    conn = get_db_connection()
    conn.execute(
        "DELETE FROM prediction_history WHERE id = ? AND username = ?",
        (history_id, user["username"]),
    )
    conn.commit()
    conn.close()
    flash("History record deleted.")
    return redirect(url_for("dashboard"))


@app.route("/clear-history", methods=["POST"])
def clear_history():
    user = get_current_user()
    if not user:
        return redirect(url_for("login"))

    conn = get_db_connection()
    conn.execute("DELETE FROM prediction_history WHERE username = ?", (user["username"],))
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

        plt.figure(figsize=(6, 4))
        plt.plot(labels, user_scores, marker="o", label="Your Score")
        plt.plot(labels, ideal_scores, marker="o", label="Ideal Score")
        plt.legend()
        plt.title("Performance Graph")
        plt.grid(True)

        graph_dir = os.path.join(BASE_DIR, "static", "images")
        os.makedirs(graph_dir, exist_ok=True)
        graph_filename = f"graph_user_{user['id']}.png"
        graph_path = os.path.join(graph_dir, graph_filename)
        plt.savefig(graph_path)
        plt.close()
        relative_graph_path = f"images/{graph_filename}"

        suggestion, weekly_plan = build_weekly_plan(
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
            SET last_prediction = ?, graph_image = ?, last_suggestion = ?, last_weekly_plan = ?
            WHERE id = ?
            """,
            (f"{output} | Chance: {chance}%", relative_graph_path, suggestion, weekly_plan_db, user["id"]),
        )
        conn.execute(
            """
            INSERT INTO prediction_history
            (username, cgpa, aptitude, coding, communication, projects, internships, result, chance, date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["username"],
                features[1],
                features[3],
                features[4],
                features[5],
                features[6],
                features[7],
                output,
                chance,
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        conn.close()

        return render_template(
            "index.html",
            prediction_text=output,
            chance=chance,
            suggestion=suggestion,
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
    resume_text = ""
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
                        suggestions = analyze_resume_text(resume_text)
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
    )


if __name__ == "__main__":
    app.run(debug=True)