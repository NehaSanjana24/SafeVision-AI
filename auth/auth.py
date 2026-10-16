"""Password hashing and Streamlit authentication helpers."""

from __future__ import annotations

import os
import re
from typing import Any

import bcrypt

from db.database import create_user, ensure_default_admin, get_connection, get_user_by_email, get_user_by_username


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def initialize_default_admin() -> None:
    connection = get_connection()
    try:
        has_admin = connection.execute("SELECT 1 FROM users WHERE role = 'admin' LIMIT 1").fetchone()
    finally:
        connection.close()
    if not has_admin:
        password = os.getenv("DEFAULT_ADMIN_PASSWORD", "ChangeMe123!")
        ensure_default_admin(hash_password(password))


def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    email = email.strip().lower()
    if len(username) < 3 or not re.fullmatch(r"[A-Za-z0-9_.-]+", username):
        return False, "Username must be at least 3 characters and use letters, numbers, dots, dashes, or underscores."
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        return False, "Enter a valid email address."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if get_user_by_username(username) or get_user_by_email(email):
        return False, "That username or email is already registered."
    try:
        create_user(username, email, hash_password(password), "supervisor")
    except Exception:
        return False, "Account could not be created. Check that the username and email are unique."
    return True, "Account created. You can now sign in."


def authenticate(username: str, password: str) -> dict[str, Any] | None:
    user = get_user_by_username(username)
    if user and verify_password(password, user["password_hash"]):
        return user
    return None


def logout() -> None:
    import streamlit as st

    st.session_state["authenticated"] = False
    for key in ("authenticated", "user_id", "username", "role"):
        if key != "authenticated":
            st.session_state.pop(key, None)


def login_screen() -> None:
    """Render the only view available to unauthenticated visitors."""
    import streamlit as st

    st.markdown("<div class='auth-kicker'>SITE SAFETY INTELLIGENCE</div>", unsafe_allow_html=True)
    st.title("AI Safety Monitoring")
    st.caption("Sign in to review PPE detection, site compliance, and safety events.")
    left, center, right = st.columns([1, 1.25, 1])
    with center:
        login_tab, signup_tab = st.tabs(["Sign in", "Create account"])
        with login_tab:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in", width="stretch", type="primary")
            if submitted:
                user = authenticate(username, password)
                if user:
                    st.session_state.update(
                        authenticated=True,
                        user_id=user["id"],
                        username=user["username"],
                        role=user["role"],
                    )
                    st.rerun()
                st.error("Incorrect username or password.")
        with signup_tab:
            with st.form("signup_form"):
                new_username = st.text_input("Username", key="signup_username")
                email = st.text_input("Email address")
                new_password = st.text_input("Password", type="password", key="signup_password")
                confirmation = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Create account", width="stretch", type="primary")
            if submitted:
                if new_password != confirmation:
                    st.error("Passwords do not match.")
                else:
                    success, message = register_user(new_username, email, new_password)
                    (st.success if success else st.error)(message)
