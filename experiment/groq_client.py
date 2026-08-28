import time
import random
import re

import requests


class GroqRateLimitClient:

    def __init__(
        self,
        api_key,
        model,
        api_url,
        timeout=60,
        max_retries=4,
        min_request_interval=4.0
    ):

        if not api_key:
            raise ValueError(
                "Groq API key is missing."
            )

        self.api_key = api_key
        self.model = model
        self.api_url = api_url

        self.timeout = timeout
        self.max_retries = max_retries

        self.min_request_interval = (
            min_request_interval
        )

        self.last_request_time = 0.0

    # ========================================================
    # REQUEST SPACING
    # ========================================================

    def _wait_for_request_slot(self):

        now = time.monotonic()

        elapsed = (
            now - self.last_request_time
        )

        if elapsed < self.min_request_interval:

            sleep_time = (
                self.min_request_interval
                - elapsed
            )

            print(
                f"[Groq] Waiting "
                f"{sleep_time:.2f}s "
                f"before next request..."
            )

            time.sleep(sleep_time)

    # ========================================================
    # RETRY DELAY
    # ========================================================

    def _get_retry_delay(
        self,
        response,
        attempt
    ):

        # ----------------------------------------------------
        # 1. Groq explicitly tells us how long to wait
        # ----------------------------------------------------

        retry_after = response.headers.get(
            "retry-after"
        )

        if retry_after:

            try:

                return max(
                    float(retry_after),
                    1.0
                )

            except ValueError:
                pass

        # ----------------------------------------------------
        # 2. Token reset
        # ----------------------------------------------------

        reset_tokens = response.headers.get(
            "x-ratelimit-reset-tokens"
        )

        if reset_tokens:

            seconds = self._parse_duration(
                reset_tokens
            )

            if seconds is not None:

                return max(
                    seconds,
                    1.0
                )

        # ----------------------------------------------------
        # 3. Request reset
        # ----------------------------------------------------

        reset_requests = response.headers.get(
            "x-ratelimit-reset-requests"
        )

        if reset_requests:

            seconds = self._parse_duration(
                reset_requests
            )

            if seconds is not None:

                return max(
                    seconds,
                    1.0
                )

        # ----------------------------------------------------
        # 4. Exponential backoff fallback
        # ----------------------------------------------------

        base_delay = 2 ** attempt

        jitter = random.uniform(
            0.5,
            1.5
        )

        return min(
            base_delay * jitter,
            60
        )

    # ========================================================
    # PARSE GROQ DURATION
    # ========================================================

    @staticmethod
    def _parse_duration(value):

        if not value:
            return None

        value = value.strip()

        total_seconds = 0.0

        # Example:
        #
        # 2m59.56s

        minutes_match = re.search(
            r"(\d+(?:\.\d+)?)m",
            value
        )

        seconds_match = re.search(
            r"(\d+(?:\.\d+)?)s",
            value
        )

        if minutes_match:

            total_seconds += (
                float(
                    minutes_match.group(1)
                ) * 60
            )

        if seconds_match:

            total_seconds += float(
                seconds_match.group(1)
            )

        return (
            total_seconds
            if total_seconds > 0
            else None
        )

    # ========================================================
    # RATE-LIMIT INFORMATION
    # ========================================================

    @staticmethod
    def _get_rate_limit_info(
        response
    ):

        headers = response.headers

        return {

            "remaining_requests":
                headers.get(
                    "x-ratelimit-remaining-requests"
                ),

            "remaining_tokens":
                headers.get(
                    "x-ratelimit-remaining-tokens"
                ),

            "reset_requests":
                headers.get(
                    "x-ratelimit-reset-requests"
                ),

            "reset_tokens":
                headers.get(
                    "x-ratelimit-reset-tokens"
                )
        }

    # ========================================================
    # CHAT COMPLETION
    # ========================================================

    def chat(
        self,
        messages,
        temperature=0.0
    ):

        payload = {

            "model": self.model,

            "messages": messages,

            "temperature": temperature,

            # GPT-OSS supports controlling reasoning.
            # We explicitly disable reasoning content because
            # Experiment A evaluates final answers, not reasoning.
            "include_reasoning": False
        }

        headers = {

            "Authorization":
                f"Bearer {self.api_key}",

            "Content-Type":
                "application/json"
        }

        for attempt in range(
            self.max_retries
        ):

            self._wait_for_request_slot()

            try:

                request_start = (
                    time.perf_counter()
                )

                response = requests.post(

                    self.api_url,

                    json=payload,

                    headers=headers,

                    timeout=self.timeout
                )

                request_latency = (
                    time.perf_counter()
                    - request_start
                )

                # Record time of last actual request
                self.last_request_time = (
                    time.monotonic()
                )

            except requests.RequestException as e:

                if attempt >= (
                    self.max_retries - 1
                ):
                    raise

                delay = min(
                    2 ** attempt,
                    30
                )

                print(
                    "[Groq] Network error: "
                    f"{e}"
                )

                print(
                    f"[Groq] Retrying in "
                    f"{delay}s..."
                )

                time.sleep(delay)

                continue

            # =================================================
            # SUCCESS
            # =================================================

            if response.status_code == 200:

                data = response.json()

                usage = data.get(
                    "usage",
                    {}
                )

                rate_limits = (
                    self._get_rate_limit_info(
                        response
                    )
                )

                return {

                    "data": data,

                    "latency_seconds":
                        request_latency,

                    "usage": {

                        "prompt_tokens":
                            usage.get(
                                "prompt_tokens",
                                0
                            ),

                        "completion_tokens":
                            usage.get(
                                "completion_tokens",
                                0
                            ),

                        "total_tokens":
                            usage.get(
                                "total_tokens",
                                0
                            )
                    },

                    "rate_limits":
                        rate_limits
                }

            # =================================================
            # RATE LIMIT
            # =================================================

            if response.status_code == 429:

                delay = (
                    self._get_retry_delay(
                        response,
                        attempt
                    )
                )

                print(
                    "[Groq] 429 Too Many "
                    "Requests."
                )

                print(
                    "[Groq] Retry delay: "
                    f"{delay:.2f}s"
                )

                time.sleep(delay)

                continue

            # =================================================
            # SERVER ERROR
            # =================================================

            if response.status_code >= 500:

                delay = min(
                    2 ** attempt,
                    30
                )

                print(
                    "[Groq] Server error "
                    f"{response.status_code}."
                )

                print(
                    f"[Groq] Retrying in "
                    f"{delay}s..."
                )

                time.sleep(delay)

                continue

            # =================================================
            # OTHER ERROR
            # =================================================

            raise RuntimeError(

                "Groq API error "

                f"{response.status_code}: "

                f"{response.text}"
            )

        raise RuntimeError(
            "Groq request failed after "
            f"{self.max_retries} attempts."
        )