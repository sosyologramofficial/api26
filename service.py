"""
Service Module - Yolly AI
Integrates with Yolly.ai service.
Uses fakemail.net temp mail for on-the-fly account registration and verification.
Saves created accounts directly to the database.
"""
import os
import json
import time
import random
import string
import re
import threading
import atexit
import requests
from concurrent.futures import ThreadPoolExecutor
import database as db

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except Exception:
    BeautifulSoup = None
    _HAS_BS4 = False

# Graceful shutdown event
_shutdown_event = threading.Event()
atexit.register(lambda: _shutdown_event.set())

# --- Model Configurations & AVAILABLE_MODELS ---
MODELS = {
    "veo3.1-basic": {
        "label": "Veo 3.1 Basic",
        "type": "video",
        "aspect_ratios": ["16:9", "9:16"],
        "resolutions": ["1080p"],
        "durations": ["5"],
        "input_modes": ["text", "image"],
        "supports_start_end_frame": True,
        "default_resolution": "1080p",
        "default_duration": "5",
        "default_aspect_ratio": "16:9",
        "extra_params": {
            "negativePrompt": "",
            "audioUrl": "",
            "enablePromptExpansion": False,
            "cameraFixed": False,
            "generateAudio": False,
            "cfgScale": 0.5
        }
    },
    "grok-imagine": {
        "label": "Grok Imagine",
        "type": "video",
        "aspect_ratios": ["16:9", "9:16", "1:1", "2:3", "3:2"],
        "resolutions": ["480p", "720p"],
        "durations": ["6", "10"],
        "input_modes": ["text", "image"],
        "supports_start_end_frame": False,
        "default_resolution": "480p",
        "default_duration": "6",
        "default_aspect_ratio": "16:9",
        "extra_params": {
            "negativePrompt": "",
            "audioUrl": "",
            "enablePromptExpansion": False,
            "cameraFixed": False,
            "cfgScale": 0.5
        }
    },
    "nano-banana": {
        "label": "Nano Banana",
        "type": "image",
        "aspect_ratios": ["Auto", "1:1", "4:3", "3:4", "16:9", "9:16"],
        "resolutions": [],
        "input_modes": ["text", "image"],
        "default_aspect_ratio": "1:1",
        "default_resolution": None,
    },
    "nano-banana-pro": {
        "label": "Nano Banana Pro",
        "type": "image",
        "aspect_ratios": ["1:1", "3:2", "2:3", "3:4", "4:3", "9:16", "16:9", "21:9"],
        "resolutions": ["1k", "2k", "4k"],
        "input_modes": ["text", "image"],
        "default_aspect_ratio": "1:1",
        "default_resolution": "1k",
    },
    "nano-banana-2": {
        "label": "Nano Banana 2",
        "type": "image",
        "aspect_ratios": ["Auto", "1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
        "resolutions": ["1k", "2k", "4k"],
        "input_modes": ["text", "image"],
        "default_aspect_ratio": "1:1",
        "default_resolution": "1k",
    },
    "gpt-image-2": {
        "label": "GPT-Image 2",
        "type": "image",
        "aspect_ratios": ["Auto", "1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
        "resolutions": ["1k", "2k", "4k"],
        "input_modes": ["text", "image"],
        "default_aspect_ratio": "1:1",
        "default_resolution": "1k",
    },
}

AVAILABLE_MODELS = {
    "image": [
        {
            "id": "nano-banana",
            "name": "Nano Banana",
            "description": "Standard image model with aspect ratio selection",
            "supports_reference_images": True,
            "max_reference_images": 5,
            "supported_sizes": ["Auto", "1:1", "4:3", "3:4", "16:9", "9:16"],
            "default_size": "1:1",
            "max_prompt_length": 4000
        },
        {
            "id": "nano-banana-pro",
            "name": "Nano Banana Pro",
            "description": "Pro image model supporting resolution (1k/2k/4k) and aspect ratios",
            "supports_reference_images": True,
            "max_reference_images": 5,
            "supported_sizes": ["1:1", "3:2", "2:3", "3:4", "4:3", "9:16", "16:9", "21:9"],
            "supported_resolutions": ["1k", "2k", "4k"],
            "default_size": "1:1",
            "default_resolution": "1k",
            "max_prompt_length": 4000
        },
        {
            "id": "nano-banana-2",
            "name": "Nano Banana 2",
            "description": "Updated image model supporting resolution (1k/2k/4k) and aspect ratios",
            "supports_reference_images": True,
            "max_reference_images": 5,
            "supported_sizes": ["Auto", "1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
            "supported_resolutions": ["1k", "2k", "4k"],
            "default_size": "1:1",
            "default_resolution": "1k",
            "max_prompt_length": 4000
        },
        {
            "id": "gpt-image-2",
            "name": "GPT-Image 2",
            "description": "GPT based image generator supporting resolution (1k/2k/4k)",
            "supports_reference_images": True,
            "max_reference_images": 5,
            "supported_sizes": ["Auto", "1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
            "supported_resolutions": ["1k", "2k", "4k"],
            "default_size": "1:1",
            "default_resolution": "1k",
            "max_prompt_length": 4000
        }
    ],
    "video": [
        {
            "id": "veo3.1-basic",
            "name": "Veo 3.1 Basic",
            "description": "High-fidelity video generation. Supports Start/End frame.",
            "supports_start_frame": True,
            "supports_end_frame": True,
            "supports_reference_images": False,
            "supported_sizes": ["16:9", "9:16"],
            "supported_durations": [5],
            "supported_resolutions": ["1080p"],
            "default_size": "16:9",
            "default_resolution": "1080p",
            "default_duration": 5,
            "max_prompt_length": 2000
        },
        {
            "id": "grok-imagine",
            "name": "Grok Imagine",
            "description": "High quality video model with resolution support (480p/720p) and durations.",
            "supports_start_frame": True,
            "supports_end_frame": False,
            "supports_reference_images": False,
            "supported_sizes": ["16:9", "9:16", "1:1", "2:3", "3:2"],
            "supported_durations": [6, 10],
            "supported_resolutions": ["480p", "720p"],
            "default_size": "16:9",
            "default_resolution": "480p",
            "default_duration": 6,
            "max_prompt_length": 2000
        }
    ],
    "tts": [],
    "music": []
}

# Proxy settings
PROXYSCRAPE_URL = "https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&proxy_format=protocolipport&format=text"

def get_available_models(mode=None):
    if mode:
        return AVAILABLE_MODELS.get(mode, [])
    return AVAILABLE_MODELS

# --- Proxy Crawl & Test Helpers ---

def fetch_proxies() -> list:
    try:
        r = requests.get(PROXYSCRAPE_URL, timeout=10)
        if r.status_code == 200:
            proxies = [line.strip() for line in r.text.splitlines() if line.strip()]
            random.shuffle(proxies)
            return proxies
    except Exception as e:
        print(f"[-] Proxy scraping failed: {e}")
    return []

def test_proxy(proxy_url: str, timeout: int = 5) -> bool:
    try:
        proxies = {"http": proxy_url, "https": proxy_url}
        r = requests.get("https://www.yolly.ai", proxies=proxies, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False

def find_working_proxy(max_workers: int = 30) -> str:
    proxy_list = fetch_proxies()
    if not proxy_list:
        return None

    import queue
    result_q = queue.Queue()
    found_event = threading.Event()

    def probe(proxy: str):
        if found_event.is_set():
            return
        if test_proxy(proxy):
            if not found_event.is_set():
                found_event.set()
                result_q.put(proxy)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(probe, proxy_list)

    try:
        working = result_q.get_nowait()
        return working
    except queue.Empty:
        return None

# --- Session & Temp Mail helpers ---

def make_session():
    s = requests.Session()
    s.headers.update({
        "accept": "application/json, text/plain, */*",
        "accept-language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
        "origin": "https://www.yolly.ai",
        "referer": "https://www.yolly.ai/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
    })
    return s

FAKEMAIL_BASE = "https://www.fakemail.net"

def make_mail_session():
    """Creates a fakemail.net session and returns (session, email).
    The mailbox is bound to the session cookies, so the session must be reused
    while polling for the verification message.
    """
    ms = requests.Session()
    ms.headers.update({
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "tr-TR,tr;q=0.9",
        "sec-ch-ua": '"Chromium";v="146", "Not-A.Brand";v="24", "Google Chrome";v="146"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    })

    try:
        html = ms.get(f"{FAKEMAIL_BASE}/", timeout=20).text
        m = re.search(r'const CSRF="([a-f0-9]+)"', html)
        if not m:
            print("[-] fakemail CSRF token not found.")
            return None, None
        csrf_token = m.group(1)

        ms.headers.update({
            "accept": "application/json, text/javascript, */*; q=0.01",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "x-requested-with": "XMLHttpRequest",
            "referer": f"{FAKEMAIL_BASE}/",
        })

        res = ms.get(f"{FAKEMAIL_BASE}/index/index", params={"csrf_token": csrf_token}, timeout=20)
        data = json.loads(res.content.decode("utf-8-sig"))
        email = data.get("email")
        if not email:
            print("[-] fakemail address could not be obtained.")
            return None, None
        return ms, email
    except Exception as e:
        print(f"[-] Temp mail creation failed: {e}")
        return None, None

def _extract_code_from_html(mail_html: str):
    """Extracts a 4-8 digit verification code from a fakemail message body."""
    if _HAS_BS4:
        try:
            soup = BeautifulSoup(mail_html, "html.parser")
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if re.fullmatch(r"\d{4,8}", text):
                    return text
            plain = soup.get_text(" ", strip=True)
        except Exception:
            plain = re.sub(r"<[^>]+>", " ", mail_html)
    else:
        plain = re.sub(r"<[^>]+>", " ", mail_html)

    otp = re.search(r"\b(\d{6})\b", plain)
    if otp:
        return otp.group(1)
    otp = re.search(r"\b(\d{4,8})\b", plain)
    if otp:
        return otp.group(1)
    return None

def poll_temp_mail_code(mail_session, attempts: int = 25, delay: int = 2):
    """Polls the fakemail inbox until a Yolly verification code arrives."""
    seen_ids = set()
    for _ in range(attempts):
        if _shutdown_event.wait(delay):
            return None
        try:
            refresh = mail_session.get(f"{FAKEMAIL_BASE}/index/refresh", timeout=20)
            if refresh.status_code != 200:
                continue
            messages = json.loads(refresh.content.decode("utf-8-sig"))
            if not isinstance(messages, list):
                continue

            for msg in messages:
                msg_id = msg.get("id")
                if not msg_id or msg_id in seen_ids:
                    continue
                seen_ids.add(msg_id)

                subject = (msg.get("predmet") or "").lower()
                sender = (msg.get("od") or msg.get("from") or "").lower()

                mail_html = mail_session.get(f"{FAKEMAIL_BASE}/email/id/{msg_id}", timeout=20).text
                code = _extract_code_from_html(mail_html)
                if not code:
                    continue

                if ("yolly" in subject or "verification" in subject
                        or "yolly" in sender or "verification" in sender):
                    return code
                # Fallback: single-message inbox with a valid looking code
                if len(code) == 6:
                    return code
        except Exception as e:
            print(f"[!] Email poll exception: {e}")
    return None

def create_yolly_account(api_key_id):
    """Creates a new Yolly.ai account dynamically on-the-fly.
    Uses fakemail.net for temp mail. Saves account to database.
    """
    mail_session, email = make_mail_session()
    if not mail_session or not email:
        return None, None

    print(f"[*] Temp mail created: {email}")

    s = make_session()

    # 1. send-code
    send_code_url = "https://www.yolly.ai/api/auth/send-code"

    use_proxy = False
    try:
        res = s.post(send_code_url, json={"email": email}, timeout=15)
        is_rate_limited = False
        if res.status_code != 200:
            is_rate_limited = True
        else:
            try:
                res_json = res.json()
                if res_json.get("message") == "RATE_LIMIT_IP" or res_json.get("code") == -1:
                    is_rate_limited = True
            except Exception:
                pass

        if is_rate_limited:
            use_proxy = True
    except Exception as e:
        print(f"[-] Initial send-code failed, switching to proxy: {e}")
        use_proxy = True

    if use_proxy:
        print("[*] send-code is rate limited or failed. Trying with proxy...")
        working_proxy = find_working_proxy()
        proxies_dict = {"http": working_proxy, "https": working_proxy} if working_proxy else None
        try:
            res = s.post(send_code_url, json={"email": email}, proxies=proxies_dict, timeout=15)
            if res.status_code != 200:
                print(f"[-] send-code failed with proxy: {res.status_code}")
                return None, None
        except Exception as e:
            print(f"[-] send-code proxy request failed: {e}")
            return None, None
    else:
        print("[+] send-code successful without proxy.")

    # 2. Poll temp mail for OTP code
    code = poll_temp_mail_code(mail_session, attempts=25, delay=2)

    if not code:
        print("[-] Verification OTP code not received.")
        return None, None

    # 3. get csrf token
    csrf_token = None
    try:
        csrf_url = "https://www.yolly.ai/api/auth/csrf"
        csrf_res = s.get(csrf_url, timeout=15)
        if csrf_res.status_code == 200:
            csrf_token = csrf_res.json().get("csrfToken")
    except Exception as e:
        print(f"[-] CSRF request failed: {e}")
        return None, None

    if not csrf_token:
        print("[-] CSRF Token not found.")
        return None, None

    # 4. callback verify (with proxy)
    working_proxy = find_working_proxy()
    verify_proxies = {"http": working_proxy, "https": working_proxy} if working_proxy else None

    verify_url = "https://www.yolly.ai/api/auth/callback/verification-code?"
    verify_payload = {
        "email": email,
        "code": code,
        "firstVisitPage": "/",
        "redirect": "false",
        "callbackUrl": "https://www.yolly.ai/",
        "csrfToken": csrf_token
    }
    verify_headers = dict(s.headers)
    verify_headers["content-type"] = "application/x-www-form-urlencoded"

    try:
        verify_res = s.post(
            verify_url,
            data=verify_payload,
            headers=verify_headers,
            proxies=verify_proxies,
            timeout=15
        )
        if verify_res.status_code != 200:
            print(f"[-] Callback verification failed: {verify_res.status_code}")
            return None, None
    except Exception as e:
        print(f"[-] Verify connection exception: {e}")
        return None, None

    # 5. Check credits to ensure account is fresh & active
    try:
        credits_res = s.get("https://www.yolly.ai/api/user/credits", timeout=15)
        if credits_res.status_code == 200:
            credits_data = credits_res.json()
            left_credits = int(credits_data.get("left_credits", 0))
            if left_credits == 0:
                print("[-] Registered account has 0 credits.")
                return None, None
        else:
            print("[-] Credits check failed.")
            return None, None
    except Exception as e:
        print(f"[-] Credits check exception: {e}")
        return None, None

    # 6. Add account to database
    db.add_account(api_key_id, email, "windows700")
    print(f"[+] Successfully registered and saved Yolly account: {email}")

    return s, email

def link_new_account_to_task(api_key_id, email, task_id):
    """Updates database:
    1. Marks the newly registered on-the-fly email as used = 1.
    2. Finds a random unused (used=0) client account from the database (excluding the new on-the-fly email).
    3. Marks that random client account as used = 1 (to consume client quota).
    4. Links the task to this random client account (so if the task fails, the client's quota is refunded).
    Returns the email of the consumed client account (or falls back to the on-the-fly email).
    """
    conn = db.get_connection()
    cursor = conn.cursor()
    consumed_email = email
    try:
        # 1. Mark on-the-fly account as used
        if db.DB_TYPE == 'postgresql':
            cursor.execute(
                'UPDATE accounts SET used = 1 WHERE api_key_id = %s AND email = %s',
                (api_key_id, email)
            )
        else:
            cursor.execute(
                'UPDATE accounts SET used = 1 WHERE api_key_id = ? AND email = ?',
                (api_key_id, email)
            )

        # 2. Find a random unused account for this client
        if db.DB_TYPE == 'postgresql':
            cursor.execute(
                'SELECT email FROM accounts WHERE api_key_id = %s AND used = 0 AND email != %s',
                (api_key_id, email)
            )
        else:
            cursor.execute(
                'SELECT email FROM accounts WHERE api_key_id = ? AND used = 0 AND email != ?',
                (api_key_id, email)
            )

        rows = cursor.fetchall()
        if rows:
            emails = []
            for r in rows:
                if isinstance(r, dict):
                    emails.append(r['email'])
                elif hasattr(r, 'keys') or isinstance(r, tuple) or isinstance(r, list):
                    emails.append(r[0])
                else:
                    emails.append(r['email'])

            if emails:
                chosen_email = random.choice(emails)
                print(f"[QUOTA] Consuming random unused account to decrease client quota: {chosen_email}")

                # 3. Mark the chosen random account as used = 1
                if db.DB_TYPE == 'postgresql':
                    cursor.execute(
                        'UPDATE accounts SET used = 1 WHERE api_key_id = %s AND email = %s',
                        (api_key_id, chosen_email)
                    )
                else:
                    cursor.execute(
                        'UPDATE accounts SET used = 1 WHERE api_key_id = ? AND email = ?',
                        (api_key_id, chosen_email)
                    )
                consumed_email = chosen_email

        # 4. Link the task to the consumed email (so that release_account refunds it on failure)
        if task_id:
            if db.DB_TYPE == 'postgresql':
                cursor.execute(
                    'UPDATE tasks SET account_email = %s WHERE task_id = %s',
                    (consumed_email, task_id)
                )
            else:
                cursor.execute(
                    'UPDATE tasks SET account_email = ? WHERE task_id = ?',
                    (consumed_email, task_id)
                )
        conn.commit()
    except Exception as e:
        print(f"Error linking account and consuming quota: {e}")
        conn.rollback()
    finally:
        conn.close()
    return consumed_email

def login_with_retry_and_link(api_key_id, task_id=None):
    """On-the-fly registration wrapper. Tries creating a Yolly account up to 5 times.
    Links the successful account to the task.
    """
    for _ in range(5):
        session, email = create_yolly_account(api_key_id)
        if session and email:
            consumed_email = link_new_account_to_task(api_key_id, email, task_id)
            return session, {"email": consumed_email}
    return None, None

def upload_image_to_yolly_b64(session, b64_data, filename_ext=".png"):
    """Parses base64 image data and uploads it to Yolly."""
    if "," in b64_data:
        b64_data = b64_data.split(",")[1]

    mime_type = "image/png" if filename_ext == ".png" else "image/jpeg"
    base64_string = f"data:{mime_type};base64,{b64_data}"

    timestamp = int(time.time() * 1000)
    file_name = f"video-input-{timestamp}-0{filename_ext}"

    upload_url = "https://www.yolly.ai/api/kie/upload"
    payload = {"base64Data": base64_string, "fileName": file_name}

    session.headers.update({"referer": "https://www.yolly.ai/video"})

    try:
        res = session.post(upload_url, json=payload, timeout=30)
        if res.status_code == 200:
            data = res.json()
            image_url = data.get("data", {}).get("url")
            if image_url:
                return image_url
    except Exception as e:
        print(f"[-] Image upload network error: {e}")
    return None

# --- Worker Functions ---

def process_image_task(task_id, params, api_key_id):
    try:
        db.update_task_status(task_id, 'running')

        session, account = login_with_retry_and_link(api_key_id, task_id)
        if not session:
            db.update_task_status(task_id, 'failed')
            db.add_task_log(task_id, "Service temporarily unavailable.")
            return

        prompt = params.get('prompt', '')
        model = params.get('model', 'nano-banana-2')
        aspect_ratio = params.get('size', '1:1')
        resolution = params.get('resolution', '1k')

        input_mode = "text"
        reference_images = []

        images = params.get('reference_images', [])
        if images:
            input_mode = "image"
            model_meta = next(
                (m for m in AVAILABLE_MODELS['image'] if m['id'] == model),
                {}
            )
            max_refs = model_meta.get('max_reference_images', 1)
            for img_b64 in images[:max_refs]:
                uploaded_url = upload_image_to_yolly_b64(session, img_b64)
                if not uploaded_url:
                    db.update_task_status(task_id, 'failed')
                    db.add_task_log(task_id, "Reference image upload failed.")
                    db.release_account(api_key_id, account['email'])
                    return
                reference_images.append(uploaded_url)
            db.update_task_reference_urls(task_id, reference_images)

        create_url = "https://www.yolly.ai/api/image/create"
        payload = {
            "model": model,
            "prompt": prompt,
            "referenceImages": reference_images,
            "aspectRatio": aspect_ratio,
            "numberOfImages": 1,
            "activeTab": "text" if input_mode == "text" else "image",
            "isPublic": True,
            "locale": "en"
        }

        model_config = MODELS.get(model, {})
        if resolution and model_config.get("resolutions"):
            payload["resolution"] = resolution

        cookie_json = json.dumps(session.cookies.get_dict())
        db.update_task_token(task_id, cookie_json)

        session.headers.update({"referer": "https://www.yolly.ai/ai-image-generator"})

        res = session.post(create_url, json=payload, timeout=20)
        if "Insufficient credits" in res.text:
            db.update_task_status(task_id, 'failed')
            db.add_task_log(task_id, "Insufficient quota.")
            db.release_account(api_key_id, account['email'])
            return

        if res.status_code != 200:
            db.update_task_status(task_id, 'failed')
            db.add_task_log(task_id, f"Submit error: {res.text}")
            db.release_account(api_key_id, account['email'])
            return

        data = res.json()
        yolly_task_id = data.get("id")

        if not yolly_task_id:
            db.update_task_status(task_id, 'failed')
            db.add_task_log(task_id, "Submit error: Task ID not found.")
            db.release_account(api_key_id, account['email'])
            return

        db.update_task_external_data(task_id, yolly_task_id, cookie_json)
        db.add_task_log(task_id, f"API Task ID: {yolly_task_id}")

        query_url = "https://www.yolly.ai/api/image/query"
        q_params = {"id": yolly_task_id}

        for _ in range(600):
            if _shutdown_event.wait(3):
                return
            try:
                q_res = session.get(query_url, params=q_params, timeout=15)
                if q_res.status_code == 200:
                    q_data = q_res.json()
                    status = q_data.get("status")

                    if status == "completed":
                        image_urls = q_data.get("image_urls", [])
                        result_urls = q_data.get("result", {}).get("imageUrls", [])
                        final_urls = image_urls or result_urls
                        if final_urls:
                            db.update_task_status(task_id, 'completed', final_urls[0])
                            return
                    elif status in ["failed", "error"]:
                        db.update_task_status(task_id, 'failed')
                        db.add_task_log(task_id, f"Submit error: {q_data}")
                        db.release_account(api_key_id, account['email'])
                        return
            except Exception:
                pass

        db.update_task_status(task_id, 'timeout')
        db.release_account(api_key_id, account['email'])

    except Exception as e:
        db.update_task_status(task_id, 'error')
        db.add_task_log(task_id, str(e))
        if 'account' in locals() and account:
            db.release_account(api_key_id, account['email'])

def process_video_task(task_id, params, api_key_id):
    try:
        db.update_task_status(task_id, 'running')

        session, account = login_with_retry_and_link(api_key_id, task_id)
        if not session:
            db.update_task_status(task_id, 'failed')
            db.add_task_log(task_id, "Service temporarily unavailable.")
            return

        prompt = params.get('prompt', '')
        model = params.get('model', 'grok-imagine')
        aspect_ratio = params.get('size', '16:9')
        resolution = params.get('resolution', '480p')
        duration = str(params.get('duration', '6'))

        input_mode = "text"
        images_payload = []

        start_frame_b64 = params.get('start_frame')
        if start_frame_b64:
            input_mode = "image"
            uploaded_start = upload_image_to_yolly_b64(session, start_frame_b64)
            if not uploaded_start:
                db.update_task_status(task_id, 'failed')
                db.add_task_log(task_id, "Start frame upload failed.")
                db.release_account(api_key_id, account['email'])
                return
            images_payload.append(uploaded_start)
            db.update_task_frame_urls(task_id, start_frame_url=uploaded_start, end_frame_url=None)

        end_frame_b64 = params.get('end_frame')
        if end_frame_b64 and start_frame_b64:
            uploaded_end = upload_image_to_yolly_b64(session, end_frame_b64)
            if not uploaded_end:
                db.update_task_status(task_id, 'failed')
                db.add_task_log(task_id, "End frame upload failed.")
                db.release_account(api_key_id, account['email'])
                return
            images_payload.append(uploaded_end)
            db.update_task_frame_urls(task_id, start_frame_url=uploaded_start, end_frame_url=uploaded_end)

        create_url = "https://www.yolly.ai/api/video/create"
        payload = {
            "model": model,
            "prompt": prompt,
            "images": images_payload,
            "inputMode": input_mode,
            "isPublic": True,
            "resolution": resolution,
            "duration": duration,
            "aspectRatio": aspect_ratio,
            "locale": "en"
        }

        model_config = MODELS.get(model, {})
        extra = model_config.get("extra_params", {
            "negativePrompt": "",
            "audioUrl": "",
            "enablePromptExpansion": False,
            "cameraFixed": False,
            "cfgScale": 0.5
        })
        payload.update(extra)

        cookie_json = json.dumps(session.cookies.get_dict())
        db.update_task_token(task_id, cookie_json)

        res = session.post(create_url, json=payload, timeout=20)
        if "Insufficient credits" in res.text:
            db.update_task_status(task_id, 'failed')
            db.add_task_log(task_id, "Insufficient quota.")
            db.release_account(api_key_id, account['email'])
            return

        if res.status_code != 200:
            db.update_task_status(task_id, 'failed')
            db.add_task_log(task_id, f"Submit error: {res.text}")
            db.release_account(api_key_id, account['email'])
            return

        data = res.json()
        yolly_task_id = data.get("id")
        provider = data.get("provider", model)

        if not yolly_task_id:
            db.update_task_status(task_id, 'failed')
            db.add_task_log(task_id, "Submit error: Task ID not found.")
            db.release_account(api_key_id, account['email'])
            return

        db.update_task_external_data(task_id, yolly_task_id, cookie_json)
        db.add_task_log(task_id, f"API Task ID: {yolly_task_id}")

        query_url = "https://www.yolly.ai/api/video/query"
        q_params = {"id": yolly_task_id, "provider": provider}

        for _ in range(600):
            if _shutdown_event.wait(3):
                return
            try:
                q_res = session.get(query_url, params=q_params, timeout=15)
                if q_res.status_code == 200:
                    q_data = q_res.json()
                    task_info = q_data.get("data") if isinstance(q_data.get("data"), dict) else None

                    if task_info:
                        status = task_info.get("status")
                        video_url = task_info.get("videoUrl")
                    else:
                        status = q_data.get("status")
                        video_url = (
                            q_data.get("video_url")
                            or q_data.get("r2_video_url")
                            or (q_data.get("video_urls", [None]) or [None])[0]
                        )

                    if status == "completed":
                        if video_url:
                            db.update_task_status(task_id, 'completed', video_url)
                            return
                    elif status in ["failed", "error"]:
                        db.update_task_status(task_id, 'failed')
                        db.add_task_log(task_id, f"Submit error: {task_info or q_data}")
                        db.release_account(api_key_id, account['email'])
                        return
            except Exception:
                pass

        db.update_task_status(task_id, 'timeout')
        db.release_account(api_key_id, account['email'])

    except Exception as e:
        db.update_task_status(task_id, 'error')
        db.add_task_log(task_id, str(e))
        if 'account' in locals() and account:
            db.release_account(api_key_id, account['email'])

def process_tts_task(task_id, params, api_key_id):
    db.update_task_status(task_id, 'failed')
    db.add_task_log(task_id, "TTS is not supported by this service.")

def process_music_task(task_id, params, api_key_id):
    db.update_task_status(task_id, 'failed')
    db.add_task_log(task_id, "Music is not supported by this service.")

def get_tts_voices(api_key_id):
    return [], "TTS not supported by this service"

def proxy_request(url, range_header=None):
    """Proxy implementation for Yolly."""
    fwd_headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
    }
    if range_header:
        fwd_headers['Range'] = range_header
    r = requests.get(url, headers=fwd_headers, stream=True, timeout=(30, 120))
    excluded = {'content-encoding', 'transfer-encoding', 'connection'}
    resp_headers = [(k, v) for k, v in r.headers.items() if k.lower() not in excluded]
    return r.iter_content(chunk_size=8192), r.status_code, resp_headers

# --- Recovery Logic ---

def check_yolly_for_task(task_id, mode, token_cookies, api_task_id, account_email=None, api_key_id=None):
    """Checks Yolly query endpoint to see if a task completed or failed during recovery."""
    try:
        cookie_dict = json.loads(token_cookies)
        session = make_session()
        requests.utils.cookiejar_from_dict(cookie_dict, session.cookies)

        if mode == 'image':
            query_url = "https://www.yolly.ai/api/image/query"
            q_res = session.get(query_url, params={"id": api_task_id}, timeout=15)
            if q_res.status_code == 200:
                q_data = q_res.json()
                status = q_data.get("status")
                if status == "completed":
                    image_urls = q_data.get("image_urls", [])
                    result_urls = q_data.get("result", {}).get("imageUrls", [])
                    final_urls = image_urls or result_urls
                    if final_urls:
                        db.update_task_status(task_id, 'completed', final_urls[0])
                        return
                elif status in ["failed", "error"]:
                    db.update_task_status(task_id, 'failed')
                    if account_email and api_key_id:
                        db.release_account(api_key_id, account_email)
                    return
            threading.Thread(target=poll_image_recovery, args=(task_id, api_task_id, token_cookies, account_email, api_key_id)).start()

        elif mode == 'video':
            task_data = db.get_task(api_key_id, task_id) if api_key_id else None
            provider = task_data.get('model', 'grok-imagine') if task_data else 'grok-imagine'

            query_url = "https://www.yolly.ai/api/video/query"
            q_res = session.get(query_url, params={"id": api_task_id, "provider": provider}, timeout=15)
            if q_res.status_code == 200:
                q_data = q_res.json()
                task_info = q_data.get("data") if isinstance(q_data.get("data"), dict) else None
                status = task_info.get("status") if task_info else q_data.get("status")
                if status == "completed":
                    video_url = task_info.get("videoUrl") if task_info else (q_data.get("video_url") or q_data.get("r2_video_url") or (q_data.get("video_urls", [None]) or [None])[0])
                    if video_url:
                        db.update_task_status(task_id, 'completed', video_url)
                        return
                elif status in ["failed", "error"]:
                    db.update_task_status(task_id, 'failed')
                    if account_email and api_key_id:
                        db.release_account(api_key_id, account_email)
                    return
            threading.Thread(target=poll_video_recovery, args=(task_id, api_task_id, token_cookies, provider, account_email, api_key_id)).start()

    except Exception as e:
        print(f"[-] Recovery check exception: {e}")
        db.update_task_status(task_id, 'failed')
        if account_email and api_key_id:
            db.release_account(api_key_id, account_email)

def poll_image_recovery(task_id, api_task_id, token_cookies, account_email=None, api_key_id=None):
    try:
        cookie_dict = json.loads(token_cookies)
        session = make_session()
        requests.utils.cookiejar_from_dict(cookie_dict, session.cookies)

        query_url = "https://www.yolly.ai/api/image/query"
        q_params = {"id": api_task_id}

        for _ in range(600):
            if _shutdown_event.wait(5):
                return
            try:
                q_res = session.get(query_url, params=q_params, timeout=15)
                if q_res.status_code == 200:
                    q_data = q_res.json()
                    status = q_data.get("status")
                    if status == "completed":
                        image_urls = q_data.get("image_urls", [])
                        result_urls = q_data.get("result", {}).get("imageUrls", [])
                        final_urls = image_urls or result_urls
                        if final_urls:
                            db.update_task_status(task_id, 'completed', final_urls[0])
                            return
                    elif status in ["failed", "error"]:
                        db.update_task_status(task_id, 'failed')
                        if account_email and api_key_id:
                            db.release_account(api_key_id, account_email)
                        return
            except:
                pass
        db.update_task_status(task_id, 'timeout')
        if account_email and api_key_id:
            db.release_account(api_key_id, account_email)
    except Exception as e:
        db.update_task_status(task_id, 'failed')
        if account_email and api_key_id:
            db.release_account(api_key_id, account_email)

def poll_video_recovery(task_id, api_task_id, token_cookies, provider='grok-imagine', account_email=None, api_key_id=None):
    try:
        cookie_dict = json.loads(token_cookies)
        session = make_session()
        requests.utils.cookiejar_from_dict(cookie_dict, session.cookies)

        query_url = "https://www.yolly.ai/api/video/query"
        q_params = {"id": api_task_id, "provider": provider}

        for _ in range(600):
            if _shutdown_event.wait(5):
                return
            try:
                q_res = session.get(query_url, params=q_params, timeout=15)
                if q_res.status_code == 200:
                    q_data = q_res.json()
                    task_info = q_data.get("data") if isinstance(q_data.get("data"), dict) else None
                    status = task_info.get("status") if task_info else q_data.get("status")
                    if status == "completed":
                        video_url = task_info.get("videoUrl") if task_info else (q_data.get("video_url") or q_data.get("r2_video_url") or (q_data.get("video_urls", [None]) or [None])[0])
                        if video_url:
                            db.update_task_status(task_id, 'completed', video_url)
                            return
                    elif status in ["failed", "error"]:
                        db.update_task_status(task_id, 'failed')
                        if account_email and api_key_id:
                            db.release_account(api_key_id, account_email)
                        return
            except:
                pass
        db.update_task_status(task_id, 'timeout')
        if account_email and api_key_id:
            db.release_account(api_key_id, account_email)
    except Exception as e:
        db.update_task_status(task_id, 'failed')
        if account_email and api_key_id:
            db.release_account(api_key_id, account_email)

def resume_incomplete_tasks():
    print("=" * 50)
    print("[STARTUP] Starting crash recovery for AI service...")
    try:
        recovery_result = db.recover_stale_tasks()
        if recovery_result['failed_count'] > 0:
            print(f"[STARTUP] Marked {recovery_result['failed_count']} tasks as failed (never logged in)")
    except Exception as e:
        print(f"[STARTUP] Error during stale task recovery: {e}")
        recovery_result = {'needs_check': []}

    needs_check = recovery_result.get('needs_check', [])
    for t in needs_check:
        if t.get('token') and t.get('external_task_id'):
            threading.Thread(
                target=check_yolly_for_task,
                args=(t['task_id'], t['mode'], t['token'], t['external_task_id'], t.get('account_email'), t.get('api_key_id'))
            ).start()
        else:
            db.update_task_status(t['task_id'], 'failed')
            if t.get('account_email') and t.get('api_key_id'):
                db.release_account(t['api_key_id'], t['account_email'])

    try:
        tasks = db.get_incomplete_tasks()
        for t in tasks:
            if t.get('token') and t.get('external_task_id'):
                task_id = t['task_id']
                mode = t['mode']
                external_id = t['external_task_id']
                token = t['token']
                account_email = t.get('account_email')
                api_key_id = t.get('api_key_id')

                print(f"  [RESUME] Task {task_id} ({mode}) - External ID: {external_id}")
                if mode == 'image':
                    threading.Thread(
                        target=poll_image_recovery,
                        args=(task_id, external_id, token, account_email, api_key_id)
                    ).start()
                elif mode == 'video':
                    task_data = db.get_task(api_key_id, task_id) if api_key_id else None
                    provider = task_data.get('model', 'grok-imagine') if task_data else 'grok-imagine'
                    threading.Thread(
                        target=poll_video_recovery,
                        args=(task_id, external_id, token, provider, account_email, api_key_id)
                    ).start()
    except Exception as e:
        print(f"[STARTUP] Error during task resume: {e}")
    print("[STARTUP] Crash recovery complete.")
    print("=" * 50)
