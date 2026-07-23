import requests
import re

BASE_URL = "http://127.0.0.1:5000"
LOGIN_URL = f"{BASE_URL}/login"

# Credentials for an approved user in your database
USER_CREDENTIALS = {
    "username": "test",  # Updated to match LoginForm 'username' field
    "password": "123456"
}


def extract_csrf_token(html_text):
    """Extracts CSRF token even if extra attributes exist between name and value."""
    match = re.search(r'name=["\']csrf_token["\'][^>]*value=["\']([^"\'\s>]+)["\']', html_text)
    if not match:
        match = re.search(r'value=["\']([^"\'\s>]+)["\'][^>]*name=["\']csrf_token["\']', html_text)
    return match.group(1) if match else None


def login_user(session):
    """Fetches CSRF token and logs in the user."""
    # Step 1: GET login page
    get_res = session.get(LOGIN_URL)
    csrf_token = extract_csrf_token(get_res.text)

    if not csrf_token:
        print("[ERROR] Could not extract CSRF token from login page HTML!")
        return False

    # Step 2: POST login payload matching forms.py (username, password, csrf_token)
    payload = {
        "username": USER_CREDENTIALS["username"],
        "password": USER_CREDENTIALS["password"],
        "csrf_token": csrf_token
    }

    post_res = session.post(LOGIN_URL, data=payload)

    # Verify authentication success
    if "Invalid username or password" in post_res.text:
        print("[ERROR] Login failed: Invalid credentials provided.")
        return False
    if "pending administrator verification approval" in post_res.text:
        print("[ERROR] Login failed: Test user account is not approved (is_approved=False).")
        return False

    print(f"Successfully authenticated as '{USER_CREDENTIALS['username']}'!")
    return True


def run_multiturn_cache_test():
    print("=== STARTING MULTI-TURN CACHE TEST ===")

    session1 = requests.Session()

    # --- Step 0: Authenticate Session 1 ---
    print("\n--- Step 0: Authenticating Session 1 ---")
    if not login_user(session1):
        return

    # --- Turn 1: Primary Query ---
    print("\n--- Turn 1: Asking initial question ---")
    payload1 = {"question": "Where is the Education Support Office located?"}
    res1 = session1.post(f"{BASE_URL}/ask", json=payload1)

    if res1.status_code != 200:
        print(f"[ERROR] /ask returned status {res1.status_code}: {res1.text[:200]}")
        return

    r1 = res1.json()
    conv_id = r1.get("conversation_id")
    print(f"Conversation ID: {conv_id}")
    print(f"Response: {r1.get('answer', '')[:100]}...\n")

    # --- Turn 2: Contextual Follow-up ---
    print("--- Turn 2: Asking contextual follow-up ('Which floor is it on?') ---")
    payload2 = {
        "question": "Which floor is it on?",
        "conversation_id": conv_id
    }
    r2 = session1.post(f"{BASE_URL}/ask", json=payload2).json()
    print(f"Response: {r2.get('answer', '')[:100]}...\n")

    # --- Turn 3: Duplicate Standalone Query in NEW Session ---
    print("--- Turn 3: Asking 'What are the Education Support Office opening hours?' in NEW Session ---")
    session2 = requests.Session()
    login_user(session2)

    payload3 = {"question": "What floor is the Education Support Office located on?"}
    r3 = session2.post(f"{BASE_URL}/ask", json=payload3).json()
    print(f"Response: {r3.get('answer', '')[:100]}...\n")


if __name__ == "__main__":
    run_multiturn_cache_test()