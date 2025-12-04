import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict
from email.header import Header
import re

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

def clean_string(text: str) -> str:
    """Cleans a string by replacing non-breaking spaces and ensuring ASCII safety"""
    if not isinstance(text, str):
        return text

    cleaned_text = text.replace('\xa0', ' ')
    cleaned_text = cleaned_text.encode('ascii', 'ignore').decode('ascii')

    return cleaned_text

def send_papers_email(
    smtp_user: str,
    smtp_password: str,
    to_address: str,
    query: str,
    papers: List[Dict]
) -> None:
    """Send an email with a list of research papers"""
    clean_query = clean_string(query)
    subject_text = f"RAG Bot - Top research papers for: {clean_query}"
    subject = Header(subject_text, 'utf-8').encode()

    lines = [f"Here are the top {len(papers)} papers for your query: {clean_query}", ""]

    for i, paper in enumerate(papers, start=1):
        title = clean_string(paper.get("title", "Untitled"))
        arxiv_id = paper.get("arxiv_id", "N/A")
        url = f"https://arxiv.org/abs/{arxiv_id}"
        lines.append(f"{i}. {title}")
        lines.append(f"   {url}")
        lines.append("")

    lines.append("Sent via RAG Research Bot.")
    body = "\n".join(lines)

    msg = MIMEMultipart(_charset="utf-8")
    msg["From"] = smtp_user
    msg["To"] = to_address
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, to_address, msg.as_string())