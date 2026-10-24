"""Stage 9, mail_out: email the finished result (translation, simplified English, original article, notes).

    python3 run.py mail_out --job ID            # builds and sends the email (if stages.mail_out.enabled is true)
    python3 -m stages.mail_out --test           # sends a short test message, to check your mail settings

Two settings in config.json, under "stages.mail_out":
    enabled   false   the switch. While it is false the pipeline skips this stage and nothing is built or sent.
    dry_run   false   true: build the email and save it as 10_email.eml (open it in any mail program to see exactly
                      what would be sent), but do not send it. No account or password is needed for that.
10_sent.json records what happened.

Connection details (config.json, "mail"; the password is never stored there, only the NAME of the variable that holds it):
    smtp_host      "smtp.office365.com"   your provider's outgoing mail server
    smtp_port      587
    security       "starttls"   or "ssl" (usually with port 465), or "none"
    username       ""           your login, usually your full address
    password_env   "MAIL_PASSWORD"   in Terminal: export MAIL_PASSWORD="your app password"
    from           ""           the From address (default: username)
    to             ""           who receives it, unless the job names a recipient (see below)
    subject_prefix "Chinese study: "

A job can carry its own recipient and subject in job.json meta: reply_to, email_subject, message_id (the mail
stage that reads incoming email, step 10, sets these so the answer goes back to whoever sent the article, as
a reply in the same thread).

The message has a plain-text part and an HTML part. Standard library only (smtplib, email).
"""
import argparse
import html
import os
import smtplib
import ssl
import sys
import time
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

from core import schema
from core.config import ConfigError, get

RESULT_JSON = "09_result.json"
EML_FILE = "10_email.eml"
SENT_FILE = "10_sent.json"
DEFAULTS = {"smtp_host": "smtp.office365.com", "smtp_port": 587, "security": "starttls",
            "password_env": "MAIL_PASSWORD", "subject_prefix": "Chinese study: "}


def setting(cfg, name):
    value = get(cfg, f"mail.{name}")
    return DEFAULTS.get(name, "") if value in (None, "") else value


def subject_for(cfg, job, original):
    meta = job.status().get("meta", {})
    given = (meta.get("email_subject") or "").strip()
    if given:
        return given if given.lower().startswith("re:") else "Re: " + given
    first = " ".join(original.split())[:60].rstrip()
    return setting(cfg, "subject_prefix") + (first or job.id)


def recipient_for(cfg, job):
    to = (job.status().get("meta", {}).get("reply_to") or setting(cfg, "to") or "").strip()
    if "@" not in to:
        raise ConfigError('No recipient. Set "mail.to" in config.json to your email address.')
    return to


def sender_for(cfg):
    sender = (setting(cfg, "from") or setting(cfg, "username") or "").strip()
    if "@" not in sender:
        raise ConfigError('No sender. Set "mail.from" (or "mail.username") in config.json to your email address.')
    return sender


# ----------------------------------------------------------------------------- the message

_STYLE = {
    "page": "font-family:-apple-system,Helvetica,Arial,sans-serif;font-size:15px;line-height:1.5;color:#222;max-width:680px",
    "zh": "font-family:'PingFang TC','Noto Sans TC','Microsoft JhengHei',sans-serif;font-size:20px;line-height:1.8",
    "h": "font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#777;margin:28px 0 6px",
    "warn": "background:#fff4e5;border-left:4px solid #e08a00;padding:8px 12px;margin:0 0 16px",
    "small": "font-size:13px;color:#555",
}


def _paragraphs(text, style=""):
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return "".join(f'<p style="margin:0 0 12px;{style}">' + html.escape(p).replace("\n", "<br>") + "</p>" for p in parts)


def body_html(result):
    warnings, rest = [], []
    for line in result["notes"]:
        if line == "WARNINGS:" or line.startswith("  - "):
            if line.startswith("  - "):
                warnings.append(line[4:])
        elif line:
            rest.append(line)
    warn = ""
    if warnings:
        warn = (f'<div style="{_STYLE["warn"]}"><b>Check this:</b><ul style="margin:4px 0 0 18px;padding:0">'
                + "".join(f"<li>{html.escape(w)}</li>" for w in warnings) + "</ul></div>")
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in rest)
    return (f'<html><body><div style="{_STYLE["page"]}">{warn}'
            f'<div style="{_STYLE["h"]}">Translation</div>{_paragraphs(result["translation"], _STYLE["zh"])}'
            f'<div style="{_STYLE["h"]}">English (simplified)</div>{_paragraphs(result["english"])}'
            f'<div style="{_STYLE["h"]}">Original article</div>{_paragraphs(result["original"], _STYLE["small"])}'
            f'<div style="{_STYLE["h"]}">Notes</div><ul style="margin:0 0 0 18px;padding:0;{_STYLE["small"]}">{notes}</ul>'
            f"</div></body></html>")


def build_message(cfg, job):
    """The EmailMessage for a finished job. Makes no network call."""
    result = job.read_json(RESULT_JSON)
    meta = job.status().get("meta", {})
    msg = EmailMessage()
    msg["From"] = sender_for(cfg)
    msg["To"] = recipient_for(cfg, job)
    msg["Subject"] = subject_for(cfg, job, result["original"])
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=(sender_for(cfg).split("@")[-1] or None))
    if meta.get("message_id"):
        msg["In-Reply-To"] = meta["message_id"]
        msg["References"] = meta["message_id"]
    msg.set_content(job.read("09_result.txt"))
    msg.add_alternative(body_html(result), subtype="html")
    return msg


# ----------------------------------------------------------------------------- sending

def send(cfg, msg):
    """Send one message with the settings in config.json. Raises ConfigError or RuntimeError in plain words."""
    host, port = setting(cfg, "smtp_host"), int(setting(cfg, "smtp_port") or 587)
    security = str(setting(cfg, "security")).lower()
    username = (setting(cfg, "username") or "").strip()
    env = setting(cfg, "password_env")
    password = os.environ.get(env, "")
    if not username:
        raise ConfigError('Set "mail.username" in config.json to your login (usually your full email address).')
    if not password:
        raise ConfigError(f'The mail password is not set. In Terminal, run: export {env}="your app password"')
    if security not in ("starttls", "ssl", "none"):
        raise ConfigError('"mail.security" in config.json must be "starttls", "ssl" or "none".')
    context = ssl.create_default_context()
    try:
        if security == "ssl":
            smtp = smtplib.SMTP_SSL(host, port, timeout=30, context=context)
        else:
            smtp = smtplib.SMTP(host, port, timeout=30)
        try:
            smtp.ehlo()
            if security == "starttls":
                smtp.starttls(context=context)
                smtp.ehlo()
            smtp.login(username, password)
            smtp.send_message(msg)
        finally:
            try:
                smtp.quit()
            except (smtplib.SMTPException, OSError):
                pass
    except smtplib.SMTPAuthenticationError as e:
        raise RuntimeError(f"The mail server refused the login ({e.smtp_code}). Many providers need an app password, "
                           "not your normal one, and Outlook.com may need SMTP sending turned on in its settings.")
    except smtplib.SMTPRecipientsRefused:
        raise RuntimeError(f"The mail server refused the recipient {msg['To']}.")
    except (smtplib.SMTPException, OSError) as e:
        raise RuntimeError(f"Could not send through {host}:{port}: {e}")


def run(cfg, job):
    msg = build_message(cfg, job)                 # a missing address is a ConfigError: this stage is switched on, so it must work
    job.write(EML_FILE, msg.as_string())
    info = {"sent": False, "to": msg["To"], "from": msg["From"], "subject": msg["Subject"],
            "message_id": msg["Message-ID"]}
    if schema.stage_opts(cfg, "mail_out")["dry_run"]:
        info["dry_run"] = True
        job.write_json(SENT_FILE, info)
        return f'dry run: the email to {msg["To"]} is saved as {EML_FILE}, not sent (stages.mail_out.dry_run is true)'
    send(cfg, msg)
    info.update(sent=True, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
    job.write_json(SENT_FILE, info)
    return f'sent to {msg["To"]}: "{msg["Subject"]}"'


def main(argv=None):
    from core.config import current_config
    ap = argparse.ArgumentParser(prog="python3 -m stages.mail_out",
                                 description="Send a short test message with the mail settings in config.json.")
    ap.add_argument("--test", action="store_true", required=True, help="send the test message")
    ap.add_argument("--to", help="send it here instead of mail.to")
    args = ap.parse_args(argv)
    try:
        cfg = current_config()
        msg = EmailMessage()
        msg["From"] = sender_for(cfg)
        msg["To"] = args.to or setting(cfg, "to") or sender_for(cfg)
        msg["Subject"] = "Chinese Study Buddy: test message"
        msg["Date"] = formatdate(localtime=True)
        msg.set_content("If you can read this, your mail settings work.\n\n這是一封測試郵件。\n")
        send(cfg, msg)
    except (ConfigError, RuntimeError) as e:
        sys.exit(f"Error: {e}")
    print(f"Sent a test message to {msg['To']}.")


if __name__ == "__main__":
    main()
