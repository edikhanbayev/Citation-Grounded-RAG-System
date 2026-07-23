import requests
import re
import time

BASE_URL = "http://127.0.0.1:5000"
LOGIN_URL = f"{BASE_URL}/login"

USER_CREDENTIALS = {
    "username": "test",
    "password": "123456"
}

TARGET_QUERIES = [
    "Who are the Senior Tutors in the School of Computer Science?",
    "How can I contact the Computer Science Welfare Team?",
    "What support is offered by the Student Disability Service?",
    "What skills does the Academic Skills Centre help develop?",
    "What is the Peer Assisted Study Sessions scheme?",
    "How do I request an appointment with a Wellbeing Officer?"
]


def extract_csrf_token(html_text):
    match = re.search(r'name=["\']csrf_token["\'][^>]*value=["\']([^"\'\s>]+)["\']', html_text)
    if not match:
        match = re.search(r'value=["\']([^"\'\s>]+)["\'][^>]*name=["\']csrf_token["\']', html_text)
    return match.group(1) if match else None


def login_user(session):
    get_res = session.get(LOGIN_URL)
    csrf_token = extract_csrf_token(get_res.text)
    if not csrf_token:
        print("[ERROR] Could not extract CSRF token.")
        return False

    payload = {
        "username": USER_CREDENTIALS["username"],
        "password": USER_CREDENTIALS["password"],
        "csrf_token": csrf_token
    }

    post_res = session.post(LOGIN_URL, data=payload)
    if "Invalid username or password" in post_res.text or "pending administrator" in post_res.text:
        print("[ERROR] Authentication failed.")
        return False

    print(f"Successfully authenticated as '{USER_CREDENTIALS['username']}'!")
    return True


def run_hot_tier_load_test():
    print("=== STARTING HOT-TIER PROMOTION LOAD TEST ===")
    session = requests.Session()

    if not login_user(session):
        return

    for idx, query in enumerate(TARGET_QUERIES, start=1):
        print(f"\n[Request {idx}/6] Sending: '{query}'")
        payload = {"question": query}
        start = time.time()
        res = session.post(f"{BASE_URL}/ask", json=payload)
        elapsed = round((time.time() - start) * 1000, 2)

        if res.status_code == 200:
            print(f" Status: 200 OK | Latency: {elapsed} ms")
        else:
            print(f" Error {res.status_code}: {res.text[:150]}")

        time.sleep(0.5)


if __name__ == "__main__":
    run_hot_tier_load_test()