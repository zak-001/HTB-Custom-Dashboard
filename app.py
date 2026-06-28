import os
import time
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Conservative local safety defaults. HTB does not publish stable public limits for
# all internal endpoints used by the web app, so the dashboard throttles itself
# before every live request and relies on Streamlit caching to avoid repeats.
DEFAULT_MIN_REQUEST_INTERVAL_SEC = float(os.getenv("HTB_MIN_REQUEST_INTERVAL_SEC", "2.5"))
DEFAULT_MAX_REQUESTS_PER_MINUTE = int(os.getenv("HTB_MAX_REQUESTS_PER_MINUTE", "0"))  # 0 = disabled; spacing-only limiter
DEFAULT_MAX_REQUESTS_PER_REFRESH = int(os.getenv("HTB_MAX_REQUESTS_PER_REFRESH", "0"))
DEFAULT_ACTIVITY_PAGES_PER_USER = int(os.getenv("HTB_ACTIVITY_PAGES_PER_USER", "2"))
DEFAULT_MACHINE_PAGES = int(os.getenv("HTB_MACHINE_PAGES", "3"))
########/api/v5/machines?per_page=15&showCompleted=incomplete
DEFAULT_IDS = [2266673, 3769, 4935]
DEFAULT_BASE = "https://labs.hackthebox.com/api/v4"
DEFAULT_BASE_V5 = "https://labs.hackthebox.com/api/v5"
HTB_APP_BASE = "https://app.hackthebox.com"
HTB_WEB_BASE = "https://www.hackthebox.com"
HTB_CHALLENGE_CATEGORY_ASSET_BASE = "https://htb-mp-prod-public-storage.s3.eu-central-1.amazonaws.com/challenge_categories"
HTB_AVATAR_ASSET_BASE = "https://htb-mp-prod-public-storage.s3.eu-central-1.amazonaws.com/avatars"

# The HTB web app changes internal API routes from time to time. Keep endpoint
# candidates here so failed routes are visible in diagnostics instead of breaking
# the dashboard.
ENDPOINTS = {
    "basic": ["user/profile/basic/{id}"],
    "activity": [
        ("v5", "user/profile/activity/{id}"),
        ("v4", "user/profile/owns/{id}"),
        ("v4", "user/profile/content/{id}"),
        ("v4", "user/profile/bloods/{id}"),
    ],
    "challenges_progress": ["user/profile/progress/challenges/{id}"],
    "challenge_active": [
        "challenge/list",
        "challenge/list?state=active",
        "challenge/active",
        "challenge/paginated",
    ],
    "challenge_categories": [
        "challenge/categories",
        "challenge/category/list",
        "challenge/categories/list",
    ],
    "fortress": ["user/profile/progress/fortress/{id}"],
    "prolab": ["user/profile/progress/prolab/{id}"],
    "graph_week": ["user/profile/graph/1W/{id}"],
    "graph_year": ["user/profile/graph/1Y/{id}"],
    "content": ["user/profile/content/{id}"],
    "bloods": ["user/profile/bloods/{id}"],
    "machine_active": ["machine/active"],
    "machine_paginated": ["machine/paginated"],
    "machine_paginated_pages": ["machine/paginated?page={page}", "machine/paginated?per_page=100&page={page}", "machine/paginated?page={page}&per_page=100"],
    # Best-effort v5 machine catalogue candidates. These are used only for the
    # optional free non-seasonal list, and are guarded by the same local rate limiter.
    "machine_free_nonseasonal_pages": [
        ("v5", "machines?per_page=50&page={page}&showCompleted=incomplete&free=true"),
        ("v5", "machines?per_page=50&page={page}&showCompleted=incomplete"),
        ("v5", "machines?per_page=50&page={page}"),
    ],
    "machine_todo": ["machine/todo"],
    "rankings_users": ["rankings/users"],
}


@dataclass
class ApiResult:
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    url: str
    status_code: Optional[int] = None


def clean_base(base: str) -> str:
    return (base or DEFAULT_BASE).rstrip("/")


def auth_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "htb-progress-dashboard/1.6-rate-limited",
    }


def rate_limit_settings() -> Tuple[float, int, int]:
    """Return local API pacing settings.

    This build intentionally does not enforce a local request-count stop.
    The dashboard is paced by sleeping between requests. That means sections
    load more slowly, but they do not fail with a fake local rate-limit error.
    """
    return (
        float(st.session_state.get("htb_min_request_interval_sec", DEFAULT_MIN_REQUEST_INTERVAL_SEC)),
        0,
        0,
    )


def before_live_request(url: str) -> None:
    """Sleep before live HTB requests; never synthesize local API failures."""
    min_interval, _max_per_minute, _max_per_refresh = rate_limit_settings()
    now = time.monotonic()
    state = st.session_state.setdefault(
        "htb_rate_state",
        {"last_request_at": 0.0, "refresh_count": 0},
    )

    cooldown_until = float(state.get("cooldown_until", 0.0) or 0.0)
    if cooldown_until and now < cooldown_until:
        time.sleep(max(0.0, cooldown_until - now))
        now = time.monotonic()
        state["cooldown_until"] = 0.0

    elapsed = now - float(state.get("last_request_at", 0.0))
    wait_for_interval = max(0.0, min_interval - elapsed)
    if wait_for_interval > 0:
        time.sleep(wait_for_interval)
        now = time.monotonic()

    state["last_request_at"] = now
    state["refresh_count"] = int(state.get("refresh_count", 0)) + 1
    state["last_url"] = url

def note_rate_limited_response(status_code: int) -> None:
    if status_code == 429:
        state = st.session_state.setdefault("htb_rate_state", {})
        state["cooldown_until"] = time.monotonic() + 300


@st.cache_data(ttl=300, show_spinner=False)
def fetch_url(base: str, token: str, path: str) -> ApiResult:
    url = f"{clean_base(base)}/{path.lstrip('/')}"
    before_live_request(url)
    try:
        r = requests.get(url, headers=auth_headers(token), timeout=25)
        note_rate_limited_response(r.status_code)
        if r.status_code in (401, 403, 404, 429):
            msg = {
                401: "401 unauthorized: check token",
                403: "403 forbidden: private profile or unavailable endpoint",
                404: "404 not found: endpoint path may have changed",
                429: "429 rate limited: wait and refresh",
            }[r.status_code]
            return ApiResult(None, msg, url, r.status_code)
        r.raise_for_status()
        return ApiResult(r.json(), None, url, r.status_code)
    except requests.RequestException as exc:
        return ApiResult(None, str(exc), url)
    except ValueError:
        return ApiResult(None, "Non-JSON response", url)


def endpoint_base_for(base: str, item: Any) -> Tuple[str, str]:
    """Return (base_url, path_pattern) for either plain v4 paths or (version, path) endpoint entries."""
    if isinstance(item, tuple):
        version, pattern = item
        if version == "v5":
            return DEFAULT_BASE_V5, pattern
        return base, pattern
    return base, item


def fetch_endpoint(base: str, token: str, key: str, user_id: Optional[int] = None) -> ApiResult:
    last = None
    for item in ENDPOINTS[key]:
        endpoint_base, pattern = endpoint_base_for(base, item)
        path = pattern.format(id=user_id) if user_id is not None else pattern
        result = fetch_url(endpoint_base, token, path)
        last = result
        if result.data is not None:
            return result
    return last or ApiResult(None, "No endpoint candidate tried", "")


def fetch_all_candidates(base: str, token: str, key: str, user_id: Optional[int] = None) -> List[Tuple[str, ApiResult]]:
    results = []
    for item in ENDPOINTS[key]:
        endpoint_base, pattern = endpoint_base_for(base, item)
        path = pattern.format(id=user_id) if user_id is not None else pattern
        label = f"{endpoint_base.rstrip('/')}/{path.lstrip('/')}"
        results.append((label, fetch_url(endpoint_base, token, path)))
    return results



def fetch_paginated_endpoint(
    base: str,
    token: str,
    key: str,
    max_pages: int = 10,
    user_id: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, Optional[str], str, Optional[int]]]]:
    """Fetch a paginated list from several candidate query-string shapes.

    HTB list endpoints often default to a small page size. This keeps trying
    pages until no new object IDs/names appear or a page returns fewer rows.
    It returns the best candidate shape by row count plus diagnostics.
    """
    best_rows: List[Dict[str, Any]] = []
    diagnostics: List[Tuple[str, Optional[str], str, Optional[int]]] = []
    patterns = ENDPOINTS.get(key, [])

    for pattern in patterns:
        endpoint_base, pattern_text = endpoint_base_for(base, pattern)
        rows: List[Dict[str, Any]] = []
        seen = set()
        last_count = 0
        for page in range(1, max_pages + 1):
            try:
                path = pattern_text.format(id=user_id, page=page)
            except Exception:
                path = pattern_text
            result = fetch_url(endpoint_base, token, path)
            diagnostics.append((path, result.error, result.url, result.status_code))
            if result.data is None:
                break
            page_rows = first_list(result.data)
            last_count = len(page_rows)
            if not page_rows:
                break

            new_count = 0
            for idx, row in enumerate(page_rows):
                # Prefer id, but include fallback keys because some HTB payloads
                # use machine_id/challenge_id or only a name in nested lists.
                key_value = (
                    row.get("id")
                    or row.get("machine_id")
                    or row.get("challenge_id")
                    or row.get("name")
                    or f"{page}:{idx}"
                )
                if key_value in seen:
                    continue
                seen.add(key_value)
                rows.append(row)
                new_count += 1

            if new_count == 0:
                break
            # Default HTB page size is often 15. If we got less than that, there
            # probably is no next page for this candidate shape.
            if last_count < 15:
                break

        if len(rows) > len(best_rows):
            best_rows = rows

    return best_rows, diagnostics

def parse_ids(raw: str) -> List[int]:
    ids = []
    for part in raw.replace("\n", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            st.warning(f"Ignoring invalid user id: {part}")
    return list(dict.fromkeys(ids))


def env_ids() -> List[int]:
    raw = os.getenv("HTB_USER_IDS", "")
    return parse_ids(raw) if raw.strip() else DEFAULT_IDS


def profile_payload(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not data:
        return {}
    for key in ["profile", "info", "data"]:
        val = data.get(key)
        if isinstance(val, dict):
            return val
    return data if isinstance(data, dict) else {}


def first_list(data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not data:
        return []
    for key in ["info", "message", "data", "activity", "items", "machines", "rankings", "challenges"]:
        val = data.get(key)
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)]
    for val in data.values():
        if isinstance(val, list):
            return [x for x in val if isinstance(x, dict)]
        if isinstance(val, dict):
            nested = first_list(val)
            if nested:
                return nested
    return []


def get_nested_list(data: Optional[Dict[str, Any]], names: List[str]) -> List[Dict[str, Any]]:
    if not data:
        return []
    roots = [data, profile_payload(data)]
    for root in roots:
        if not isinstance(root, dict):
            continue
        for name in names:
            val = root.get(name)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
            if isinstance(val, dict):
                nested = first_list(val)
                if nested:
                    return nested
    return first_list(data)


def pct(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "—"


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def num(value: Any) -> Any:
    return value if value not in (None, "") else "—"


def normalize_asset_url(value: Any) -> Optional[str]:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.startswith("http://") or s.startswith("https://"):
        return s
    if s.startswith("//"):
        return "https:" + s
    if s.startswith("/"):
        return urljoin(HTB_WEB_BASE, s)
    return urljoin(HTB_WEB_BASE + "/", s)


def normalize_htb_avatar_url(value: Any) -> Optional[str]:
    """Normalize HTB avatar values to the public S3 avatar URL used by the web app.

    Machine/profile activity payloads may return a full URL, an /avatars/... path,
    a /storage/avatars/... path, or just the PNG filename/hash. Streamlit
    ImageColumn needs the final public URL, for example:
    https://htb-mp-prod-public-storage.s3.eu-central-1.amazonaws.com/avatars/<file>.png
    """
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.startswith("//"):
        s = "https:" + s
    if s.startswith("http://") or s.startswith("https://"):
        if "htb-mp-prod-public-storage.s3" in s and "/avatars/" in s:
            return s
        # Convert old/app/web avatar URLs to the stable public S3 form.
        if "/avatars/" in s:
            filename = s.rsplit("/avatars/", 1)[-1].split("?", 1)[0].split("#", 1)[0]
            if filename:
                return f"{HTB_AVATAR_ASSET_BASE}/{filename}"
        return s
    # Common relative variants: avatars/hash.png, /avatars/hash.png, /storage/avatars/hash.png
    cleaned = s.split("?", 1)[0].split("#", 1)[0].strip("/")
    if "/avatars/" in cleaned:
        filename = cleaned.rsplit("/avatars/", 1)[-1]
    elif cleaned.startswith("avatars/"):
        filename = cleaned.split("/", 1)[-1]
    else:
        filename = cleaned.rsplit("/", 1)[-1]
    if filename:
        return f"{HTB_AVATAR_ASSET_BASE}/{filename}"
    return None


def avatar_url(profile: Dict[str, Any]) -> Optional[str]:
    for key in ["avatar", "avatar_url", "profile_avatar", "image", "photo"]:
        if "avatar" in key:
            url = normalize_htb_avatar_url(profile.get(key))
        else:
            url = normalize_asset_url(profile.get(key))
        if url:
            return url
    return None


RAW_ASSET_COLUMNS = {
    "avatar",
    "avatar_url",
    "avatar_thumb",
    "image_url",
    "logo",
    "logo_url",
    "cover",
    "thumbnail",
    "thumb",
    "icon",
}


# Best-effort fallback labels. The live category endpoint, when available,
# overrides these values. Unknown IDs still display as "Category <id>" so
# filters remain useful even when the active-challenge payload only exposes
# challenge_category_id.
CHALLENGE_CATEGORY_FALLBACK_NAMES = {
    5: "Web",
    6: "Misc",
    7: "Forensics",
    8: "Mobile",
    9: "OSINT",
    15: "Secure Coding",
}


def clean_category_id(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def challenge_category_asset_url(category_id: Any) -> Optional[str]:
    cid = clean_category_id(category_id)
    if cid is None:
        return None
    digest = hashlib.md5(str(cid).encode()).hexdigest()
    return f"{HTB_CHALLENGE_CATEGORY_ASSET_BASE}/{digest}.svg"



def display_df(
    rows: List[Dict[str, Any]],
    preferred_cols: List[str],
    empty: str,
    image_cols: Optional[List[str]] = None,
    hide_cols: Optional[List[str]] = None,
) -> None:
    if not rows:
        st.info(empty)
        return
    df = pd.DataFrame(rows)

    hidden = set(hide_cols or [])
    # If we render a normalized image column, suppress the original raw URL fields
    # so the table shows the picture instead of both picture + URL.
    if image_cols:
        hidden.update(c for c in RAW_ASSET_COLUMNS if c not in image_cols)
    hidden = {c for c in hidden if c in df.columns and c not in preferred_cols}
    if hidden:
        df = df.drop(columns=sorted(hidden), errors="ignore")

    cols = [c for c in preferred_cols if c in df.columns]
    show_all = bool(st.session_state.get("show_raw_columns", False))
    if cols and not show_all:
        df = df[cols]
    elif cols:
        df = df[cols + [c for c in df.columns if c not in cols]]
    column_config = {}
    for col in image_cols or []:
        if col in df.columns:
            column_config[col] = st.column_config.ImageColumn(col, width="small")
    st.dataframe(df, use_container_width=True, hide_index=True, column_config=column_config)


def machine_points(row: Dict[str, Any]) -> int:
    return as_int(row.get("points") or row.get("static_points"), 0)


def is_user_owned(row: Dict[str, Any]) -> bool:
    return bool(row.get("authUserInUserOwns") or row.get("authUserFirstUserTime"))


def is_root_owned(row: Dict[str, Any]) -> bool:
    return bool(row.get("authUserInRootOwns") or row.get("authUserFirstRootTime"))


def is_completed(row: Dict[str, Any]) -> bool:
    return is_user_owned(row) and is_root_owned(row)


def is_free(row: Dict[str, Any]) -> bool:
    free_keys = ["free", "isFree", "is_free", "free_to_play"]
    if any(row.get(k) is True or str(row.get(k)).lower() == "true" for k in free_keys):
        return True
    # Some HTB rows expose VIP rather than free. Treat explicit non-VIP as free.
    vip_keys = ["vip", "isVip", "is_vip", "vip_only"]
    if any(row.get(k) is True or str(row.get(k)).lower() == "true" for k in vip_keys):
        return False
    if any(row.get(k) is False or str(row.get(k)).lower() == "false" for k in vip_keys):
        return True
    return False


def boolish(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def is_seasonal(row: Dict[str, Any]) -> bool:
    for key in ["seasonal", "isSeasonal", "is_seasonal", "season_machine", "seasonal_machine"]:
        value = boolish(row.get(key))
        if value is not None:
            return value
    for key in ["season_id", "seasonId", "season", "season_name", "seasonName"]:
        value = row.get(key)
        if value not in (None, "", 0, "0"):
            return True
    machine_type = str(row.get("type") or row.get("machine_type") or row.get("state") or "").lower()
    return "season" in machine_type


def is_retired_or_nonseasonal_candidate(row: Dict[str, Any]) -> bool:
    # These keys vary across HTB payloads. Prefer explicit retired/free flags when present,
    # otherwise keep the row as a candidate as long as it is free and not seasonal.
    explicit_active = None
    for key in ["active", "isActive", "is_active", "currently_active"]:
        value = boolish(row.get(key))
        if value is not None:
            explicit_active = value
            break
    if explicit_active is True:
        return False
    machine_type = str(row.get("type") or row.get("machine_type") or row.get("state") or row.get("status") or "").lower()
    if any(word in machine_type for word in ["active", "seasonal"]):
        return False
    return True


def free_nonseasonal_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        if is_completed(row) or not is_free(row) or is_seasonal(row):
            continue
        if not is_retired_or_nonseasonal_candidate(row):
            continue
        rr = dict(row)
        rr["owned_user"] = is_user_owned(row)
        rr["owned_root"] = is_root_owned(row)
        rr["free_machine"] = True
        rr["seasonal"] = False
        rr["estimated_points_available"] = machine_points(row)
        out.append(rr)
    return sorted(out, key=lambda x: (as_int(x.get("difficulty")), machine_points(x), str(x.get("name"))), reverse=True)


def candidate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if is_completed(r):
            continue
        rr = dict(r)
        rr["estimated_points_available"] = machine_points(r)
        rr["owned_user"] = is_user_owned(r)
        rr["owned_root"] = is_root_owned(r)
        rr["free_machine"] = is_free(r)
        out.append(rr)
    return sorted(out, key=lambda x: (bool(x.get("free_machine")), as_int(x.get("difficulty")), machine_points(x)), reverse=True)


def parse_challenge_progress(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cp = profile_payload(data)
    owns = cp.get("challenge_owns", {}) if isinstance(cp, dict) else {}
    out: Dict[str, Any] = {}
    if isinstance(owns, dict):
        out["challenge_owns"] = owns.get("solved") or owns.get("owns") or owns.get("owned")
        out["challenge_total"] = owns.get("total")
    for key in ["challenge_owns", "challenges_owned", "challenge_solved", "challenges_solved"]:
        if out.get("challenge_owns") in (None, "") and cp.get(key) not in (None, ""):
            out["challenge_owns"] = cp.get(key)
    return out


def challenge_solved(row: Dict[str, Any]) -> bool:
    for key in ["authUserSolve", "authUserSolved", "auth_user_solved", "solved", "owned", "isOwned"]:
        if row.get(key) is True or str(row.get(key)).lower() == "true":
            return True
    return False


def normalize_activity_rows(data: Optional[Dict[str, Any]], name: str, uid: int, source: str) -> List[Dict[str, Any]]:
    rows = get_nested_list(data, ["activity", "owns", "content", "bloods", "items", "data"])
    normalized = []
    for row in rows:
        rr = dict(row)
        rr["user"] = name
        rr["user_id"] = uid
        rr["source"] = source
        normalized.append(rr)
    return normalized



TIMESTAMP_KEYS = [
    "date",
    "created_at",
    "createdAt",
    "completed_at",
    "completedAt",
    "solved_at",
    "solvedAt",
    "submitted_at",
    "submittedAt",
    "ownDate",
    "own_date",
    "owned_at",
    "ownedAt",
    "authUserFirstUserTime",
    "authUserFirstRootTime",
    "authUserSolveTime",
    "authUserSolvedAt",
    "first_user_time",
    "first_root_time",
]


def first_timestamp(row: Dict[str, Any]) -> Any:
    for key in TIMESTAMP_KEYS:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def isoish(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (int, float)):
        # HTB may return UNIX seconds or milliseconds. Keep impossible values as strings.
        try:
            if value > 10_000_000_000:
                value = value / 1000
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        except Exception:
            return str(value)
    s = str(value).strip()
    return s


def human_datetime(value: Any) -> str:
    """Return a readable local-ish UTC timestamp for API dates."""
    if value in (None, ""):
        return ""
    try:
        if isinstance(value, (int, float)):
            ts = value / 1000 if value > 10_000_000_000 else value
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            s = str(value).strip()
            # HTB commonly returns Zulu ISO strings like 2026-06-20T13:19:15.000Z.
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(value)


def row_image_url(row: Dict[str, Any]) -> Optional[str]:
    for key in [
        "avatar",
        "avatar_thumb",
        "image",
        "image_url",
        "logo",
        "logo_url",
        "cover",
        "thumbnail",
        "thumb",
        "icon",
    ]:
        if key.startswith("avatar"):
            url = normalize_htb_avatar_url(row.get(key))
        else:
            url = normalize_asset_url(row.get(key))
        if url:
            return url

    # Some HTB challenge/list payloads keep category assets nested.
    for parent_key in ["category", "challenge_category", "categoryInfo", "category_info", "prolab", "fortress"]:
        parent = row.get(parent_key)
        if isinstance(parent, dict):
            url = row_image_url(parent)
            if url:
                return url
    return None


def nested_display_name(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        for key in ["name", "title", "display_name", "categoryName", "category_name", "text", "label"]:
            candidate = value.get(key)
            if candidate not in (None, ""):
                return str(candidate)
        return None
    return str(value)


def challenge_category_id(row: Dict[str, Any]) -> Optional[int]:
    for key in [
        "challenge_category_id",
        "challengeCategoryId",
        "category_id",
        "categoryId",
        "challenge_category",
        "category",
    ]:
        value = row.get(key)
        if isinstance(value, dict):
            for nested_key in ["id", "category_id", "challenge_category_id"]:
                cid = clean_category_id(value.get(nested_key))
                if cid is not None:
                    return cid
        else:
            cid = clean_category_id(value)
            if cid is not None:
                return cid
    return None


def challenge_category_name(row: Dict[str, Any], category_lookup: Optional[Dict[int, Dict[str, Any]]] = None) -> Optional[str]:
    for key in [
        "categoryName",
        "category_name",
        "challenge_category_name",
        "challengeCategoryName",
        "category",
        "challenge_category",
        "categoryInfo",
        "category_info",
    ]:
        value = row.get(key)
        # Do not treat a bare numeric category id as a display name. Resolve it below.
        if not isinstance(value, dict) and clean_category_id(value) is not None:
            continue
        name = nested_display_name(value)
        if name:
            return name

    cid = challenge_category_id(row)
    if cid is not None:
        if category_lookup and cid in category_lookup:
            name = category_lookup[cid].get("name")
            if name:
                return str(name)
        return CHALLENGE_CATEGORY_FALLBACK_NAMES.get(cid, f"Category {cid}")
    return None


def challenge_category_image(row: Dict[str, Any], category_lookup: Optional[Dict[int, Dict[str, Any]]] = None) -> Optional[str]:
    cid = challenge_category_id(row)
    if cid is not None and category_lookup and cid in category_lookup:
        url = category_lookup[cid].get("image")
        if url:
            return str(url)
    return challenge_category_asset_url(cid)


def fetch_challenge_category_lookup(base: str, token: str) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {}
    for path, result in fetch_all_candidates(base, token, "challenge_categories"):
        if result.data is None:
            continue
        for row in first_list(result.data):
            cid = challenge_category_id(row) or clean_category_id(row.get("id"))
            if cid is None:
                continue
            name = nested_display_name(row) or row.get("name") or CHALLENGE_CATEGORY_FALLBACK_NAMES.get(cid) or f"Category {cid}"
            image = row_image_url(row) or challenge_category_asset_url(cid)
            lookup[cid] = {"name": str(name), "image": image, "source": path}
        if lookup:
            break
    for cid, name in CHALLENGE_CATEGORY_FALLBACK_NAMES.items():
        lookup.setdefault(cid, {"name": name, "image": challenge_category_asset_url(cid), "source": "fallback"})
    return lookup


def challenge_difficulty_name(row: Dict[str, Any]) -> Optional[str]:
    for key in ["difficultyText", "difficulty_text", "difficulty_name", "difficultyName", "difficulty", "level"]:
        name = nested_display_name(row.get(key))
        if name:
            return name
    return None


def decorate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add display-friendly image/date columns while preserving raw API columns."""
    decorated: List[Dict[str, Any]] = []
    date_keys = [
        "release",
        "release_date",
        "released_at",
        "created_at",
        "expires_at",
        "authUserFirstUserTime",
        "authUserFirstRootTime",
        "ownDate",
    ]
    for row in rows:
        rr = dict(row)
        for asset_col in RAW_ASSET_COLUMNS:
            if asset_col in rr and rr.get(asset_col) not in (None, ""):
                if asset_col.startswith("avatar"):
                    rr[asset_col] = normalize_htb_avatar_url(rr.get(asset_col)) or rr.get(asset_col)
                else:
                    rr[asset_col] = normalize_asset_url(rr.get(asset_col)) or rr.get(asset_col)

        img = row_image_url(rr)
        if img:
            rr["image"] = img
            # For machine tables, render the real avatar column itself as an image
            # rather than adding a separate image column and leaving avatar as text.
            if rr.get("avatar") in (None, ""):
                rr["avatar"] = img

        for key in date_keys:
            if key in rr and rr.get(key) not in (None, ""):
                rr[f"{key}_readable"] = human_datetime(rr.get(key))
        decorated.append(rr)
    return decorated


def event_sort_key(value: Any) -> str:
    return isoish(value)


def add_event(events: List[Dict[str, Any]], user: str, user_id: int, event_type: str, target: Any, timestamp: Any, source: str, points: Any = None) -> None:
    if timestamp in (None, ""):
        return
    events.append({
        "date": isoish(timestamp),
        "user": user,
        "user_id": user_id,
        "event_type": event_type,
        "target": target or "—",
        "points": points,
        "source": source,
    })


def machine_submission_events(rows: List[Dict[str, Any]], user: str, user_id: int, source: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for row in rows:
        name = row.get("name") or row.get("machine") or row.get("machine_name") or row.get("id")
        add_event(events, user, user_id, "machine user own", name, row.get("authUserFirstUserTime") or row.get("first_user_time"), source, row.get("points") or row.get("static_points"))
        add_event(events, user, user_id, "machine root own", name, row.get("authUserFirstRootTime") or row.get("first_root_time"), source, row.get("points") or row.get("static_points"))
    return events


def generic_submission_events(rows: List[Dict[str, Any]], user: str, user_id: int, source: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for row in rows:
        ts = first_timestamp(row)
        target = row.get("name") or row.get("machine") or row.get("challenge") or row.get("title") or row.get("id")
        kind = row.get("object_type") or row.get("type") or row.get("category") or "submission"
        add_event(events, user, user_id, str(kind), target, ts, source, row.get("points"))
    return events


##### fix rate limit
def fetch_v5_activity_pages(token: str, user_id: int, max_pages: int = 1) -> Tuple[List[Dict[str, Any]], List[Tuple[str, Optional[str], str, Optional[int]]]]:
    """Fetch all pages from /api/v5/user/profile/activity/{user}.

    Observed payload:
      {"data": [...], "meta": {"page": 1, "lastPage": 14, "totalItems": 199}}

    The endpoint currently defaults to 15 rows per page. This function follows
    meta.lastPage and de-duplicates by (type, id, ownDate, points).
    """
    rows: List[Dict[str, Any]] = []
    diagnostics: List[Tuple[str, Optional[str], str, Optional[int]]] = []
    seen = set()
    last_page: Optional[int] = None

    for page in range(1, max_pages + 1):
        path = f"user/profile/activity/{user_id}?page={page}"
        result = fetch_url(DEFAULT_BASE_V5, token, path)
        diagnostics.append((path, result.error, result.url, result.status_code))
        if result.data is None:
            break

        page_rows = result.data.get("data") if isinstance(result.data, dict) else None
        if not isinstance(page_rows, list):
            page_rows = first_list(result.data)
        page_rows = [x for x in page_rows if isinstance(x, dict)]
        if not page_rows:
            break

        meta = result.data.get("meta", {}) if isinstance(result.data, dict) else {}
        if isinstance(meta, dict):
            try:
                last_page = int(meta.get("lastPage") or meta.get("last_page") or last_page or page)
            except Exception:
                last_page = None

        new_count = 0
        for row in page_rows:
            row_key = (
                row.get("type"),
                row.get("id"),
                row.get("ownDate"),
                row.get("points"),
                row.get("name"),
            )
            if row_key in seen:
                continue
            seen.add(row_key)
            rows.append(row)
            new_count += 1

        if new_count == 0:
            break
        if last_page is not None and page >= last_page:
            break

    return rows, diagnostics


def normalize_v5_activity_rows(rows: List[Dict[str, Any]], name: str, uid: int) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        event_type = str(row.get("type") or "activity")
        category = row.get("categoryName") or row.get("category")
        blood = row.get("blood")
        label_parts = []
        if event_type:
            label_parts.append(event_type)
        if category:
            label_parts.append(str(category))
        event_label = " / ".join(label_parts) if label_parts else "activity"
        normalized.append({
            "date": human_datetime(row.get("ownDate")),
            "raw_date": isoish(row.get("ownDate")),
            "user": name,
            "user_id": uid,
            "event_type": event_label,
            "type": event_type,
            "target": row.get("name") or row.get("id") or "—",
            "target_id": row.get("id"),
            "category": category,
            "points": row.get("points"),
            "blood": blood,
            "avatar": normalize_asset_url(row.get("avatar")),
            "source": "api/v5 user/profile/activity",
            "profile_activity_url": f"{HTB_APP_BASE}/users/{uid}?tab=activity",
        })
    return normalized

def save_activity_snapshot(events: List[Dict[str, Any]], path: str = "htb_activity_snapshot.csv") -> None:
    """Persist the current derived activity list locally.

    Streamlit reruns often; this file gives you a simple local audit trail even
    when HTB does not expose a stable feed endpoint. It stores only rows that
    already have a timestamp.
    """
    if not events:
        return
    new_df = pd.DataFrame(events)
    if os.path.exists(path):
        try:
            old_df = pd.read_csv(path)
            new_df = pd.concat([old_df, new_df], ignore_index=True)
        except Exception:
            pass
    subset = [c for c in ["date", "user_id", "event_type", "target"] if c in new_df.columns]
    if subset:
        new_df = new_df.drop_duplicates(subset=subset, keep="last")
    new_df.to_csv(path, index=False)

def progress_bar(label: str, solved: Any, total: Any, percent: Optional[Any] = None) -> None:
    try:
        if percent is None:
            percent = 100 * float(solved) / float(total) if float(total) else 0
        percent_f = max(0.0, min(100.0, float(percent)))
    except Exception:
        percent_f = 0.0
    st.write(f"**{label}** — {num(solved)} / {num(total)} ({percent_f:.1f}%)")
    st.progress(percent_f / 100)


st.set_page_config(page_title="HTB Progress Dashboard", layout="wide")
st.title("Hack The Box Progress Dashboard")
st.caption("Dashboard build: v1.7-no-local-skip — local limiter sleeps only; it never returns a fake per-minute failure.")

with st.sidebar:
    st.header("Configuration")
    token = os.getenv("HTB_API_TOKEN", "")
    token_input = st.text_input("HTB API token", value=token, type="password")
    base = st.text_input("API base", value=os.getenv("HTB_API_BASE", DEFAULT_BASE))
    ids_raw = st.text_area("User IDs", value=",".join(map(str, env_ids())), help="First ID is you; others are friends.")
    graph_range = st.radio("History range", ["1W", "1Y"], index=1, horizontal=True)
    st.divider()
    st.subheader("API safety")
    st.session_state["htb_min_request_interval_sec"] = st.number_input(
        "Minimum seconds between live API requests",
        min_value=0.5,
        max_value=30.0,
        value=float(st.session_state.get("htb_min_request_interval_sec", DEFAULT_MIN_REQUEST_INTERVAL_SEC)),
        step=0.5,
        help="Keep this conservative. Cached responses do not consume live requests.",
    )
    st.info("Local request-count limits are disabled in this build. The dashboard only sleeps between live API requests, so tables should not fail locally.")
    activity_pages_per_user = st.number_input(
        "Activity pages per user",
        min_value=1,
        max_value=50,
        value=int(os.getenv("HTB_ACTIVITY_PAGES_PER_USER", DEFAULT_ACTIVITY_PAGES_PER_USER)),
        step=1,
        help="Lower is safer. Increase only when you intentionally want deeper history.",
    )
    machine_pages_to_fetch = st.number_input(
        "Machine catalogue pages",
        min_value=1,
        max_value=20,
        value=int(os.getenv("HTB_MACHINE_PAGES", DEFAULT_MACHINE_PAGES)),
        step=1,
        help="Used for active/free machine lists. Lower is safer.",
    )
    show_raw_columns = st.checkbox("Show raw/debug columns in tables", value=False)
    refresh = st.button("Refresh cache")

if refresh:
    st.cache_data.clear()
    st.session_state["htb_rate_state"] = {"last_request_at": 0.0, "refresh_count": 0}

st.session_state["show_raw_columns"] = bool(show_raw_columns)

# Reset only the per-rerun hard budget. Keep last_request/window/cooldown in
# session state so reruns are still paced safely.
_rate_state = st.session_state.setdefault("htb_rate_state", {"last_request_at": 0.0, "refresh_count": 0})
_rate_state["refresh_count"] = 0

api_token = token_input.strip()
user_ids = parse_ids(ids_raw)
if not api_token:
    st.error("Set HTB_API_TOKEN in `.env`, your shell, or the sidebar.")
    st.stop()
if not user_ids:
    st.error("Add at least one HTB user ID.")
    st.stop()

min_interval, max_per_minute, max_per_refresh = rate_limit_settings()
st.caption(f"API safety: ≥{min_interval:.1f}s between live requests. No local per-minute/request-count failures are generated in this build.")

profiles: Dict[int, Dict[str, Any]] = {}
challenge_progress_by_user: Dict[int, Dict[str, Any]] = {}
endpoint_warnings: List[Tuple[str, str, str]] = []
with st.spinner("Loading HTB profile summaries..."):
    for uid in user_ids:
        result = fetch_endpoint(base, api_token, "basic", uid)
        if result.error:
            endpoint_warnings.append((str(uid), result.error, result.url))
        profiles[uid] = profile_payload(result.data)

        ch_result = fetch_endpoint(base, api_token, "challenges_progress", uid)
        if ch_result.error:
            endpoint_warnings.append((f"{uid} challenges", ch_result.error, ch_result.url))
        challenge_progress_by_user[uid] = parse_challenge_progress(ch_result.data)

if endpoint_warnings:
    with st.expander("API warnings", expanded=True):
        for label, err, url in endpoint_warnings:
            st.warning(f"{label}: {err} ({url})")

summary_rows = []
for i, uid in enumerate(user_ids):
    p = profiles.get(uid, {})
    team = p.get("team") if isinstance(p.get("team"), dict) else {}
    ch = challenge_progress_by_user.get(uid, {})
    summary_rows.append({
        "avatar": avatar_url(p),
        "role": "me" if i == 0 else "friend",
        "id": uid,
        "profile": f"{HTB_APP_BASE}/users/{uid}",
        "name": p.get("name") or p.get("username") or "unknown",
        "rank": p.get("rank") or p.get("rank_name"),
        "next_rank": p.get("next_rank"),
        "global_ranking": p.get("ranking"),
        "points": p.get("points"),
        "rank_ownership": p.get("rank_ownership"),
        "rank_requirement": p.get("rank_requirement"),
        "user_owns": p.get("user_owns"),
        "root_owns": p.get("system_owns"),
        "challenge_owns": ch.get("challenge_owns"),
        "challenge_total": ch.get("challenge_total"),
        "user_bloods": p.get("user_bloods"),
        "root_bloods": p.get("system_bloods"),
        "respects": p.get("respects") or p.get("respect"),
        "team": team.get("name") if team else None,
        "country": p.get("country_name"),
    })
summary_df = pd.DataFrame(summary_rows)

st.subheader("Comparison")
summary_visible_cols = ["avatar", "role", "id", "profile", "name", "rank", "next_rank", "global_ranking", "points", "user_owns", "root_owns", "challenge_owns"]
if not st.session_state.get("show_raw_columns", False):
    summary_df = summary_df[[c for c in summary_visible_cols if c in summary_df.columns]]
st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "avatar": st.column_config.ImageColumn("avatar", width="small"),
        "profile": st.column_config.LinkColumn("profile"),
    },
)

me_id = user_ids[0]
me = profiles.get(me_id, {})
me_name = me.get("name") or me.get("username") or str(me_id)

st.subheader(f"Overview: {me_name}")
cols = st.columns(8)
metrics = [
    ("Global rank", me.get("ranking")),
    ("Points", me.get("points")),
    ("Rank", me.get("rank") or me.get("rank_name")),
    ("Next rank", me.get("next_rank")),
    ("Ownership", pct(me.get("rank_ownership"))),
    ("User owns", me.get("user_owns")),
    ("Root owns", me.get("system_owns")),
    ("Challenge owns", challenge_progress_by_user.get(me_id, {}).get("challenge_owns")),
]
for col, (label, value) in zip(cols, metrics):
    col.metric(label, num(value))

# Machine lists are auth-user scoped; they show the logged-in token owner's current state.
active_result = fetch_endpoint(base, api_token, "machine_active")
active_machine = profile_payload(active_result.data)
playable_result = fetch_endpoint(base, api_token, "machine_paginated")
playable_machines = first_list(playable_result.data)
paginated_machines, machine_page_diagnostics = fetch_paginated_endpoint(base, api_token, "machine_paginated_pages", max_pages=int(machine_pages_to_fetch))
if len(paginated_machines) > len(playable_machines):
    playable_machines = paginated_machines
todo_result = fetch_endpoint(base, api_token, "machine_todo")
todo_machines = first_list(todo_result.data)
free_catalogue_machines, free_catalogue_diagnostics = fetch_paginated_endpoint(
    base, api_token, "machine_free_nonseasonal_pages", max_pages=int(machine_pages_to_fetch)
)

st.subheader("Active and free machines")
if active_result.error:
    st.warning(f"Current active machine endpoint failed: {active_result.error} ({active_result.url})")

if active_machine and active_machine.get("id"):
    st.markdown("#### Currently spawned machine")
    display_df(decorate_rows([active_machine]), ["avatar", "id", "name", "type", "expires_at_readable", "lab_server"], "No active machine.", image_cols=["avatar"])
else:
    st.info("No currently spawned machine returned for your token.")

machine_rows = decorate_rows([dict(r, owned_user=is_user_owned(r), owned_root=is_root_owned(r), free_machine=is_free(r)) for r in playable_machines])
needed_rows = decorate_rows(candidate_rows(playable_machines))
free_rows = [r for r in machine_rows if r.get("free_machine")]
nonseasonal_free_rows = decorate_rows(free_nonseasonal_rows(free_catalogue_machines))
owned_rows = [r for r in machine_rows if r.get("owned_user") or r.get("owned_root")]

if playable_result.error:
    st.warning(f"Active machine list endpoint failed: {playable_result.error} ({playable_result.url})")
st.caption(f"Fetched {len(playable_machines)} active machine rows and {len(nonseasonal_free_rows)} free non-seasonal candidate rows under the configured request budget.")
with st.expander("Active machine pagination diagnostics"):
    if machine_page_diagnostics:
        for path, err, url, status in machine_page_diagnostics:
            st.write(f"`{path}` → status={status}; error={err}; url={url}")
    else:
        st.write("No active-machine pagination candidates were tried.")
    if free_catalogue_diagnostics:
        st.markdown("**Free/non-seasonal catalogue diagnostics**")
        for path, err, url, status in free_catalogue_diagnostics:
            st.write(f"`{path}` → status={status}; error={err}; url={url}")

machine_tabs = st.tabs(["Free active", "Free non-seasonal", "Needed active", "All active", "Owned active", "To-do"])
with machine_tabs[0]:
    display_df(free_rows, ["avatar", "id", "name", "os", "difficultyText", "points", "owned_user", "owned_root", "release_readable"], "No free active machines were returned by the API.", image_cols=["avatar"])
with machine_tabs[1]:
    display_df(nonseasonal_free_rows, ["avatar", "id", "name", "os", "difficultyText", "points", "owned_user", "owned_root", "release_readable", "stars", "rating"], "No free non-seasonal machines were returned by the API under the current page/request limits.", image_cols=["avatar"])
with machine_tabs[2]:
    display_df(needed_rows, ["avatar", "id", "name", "os", "difficultyText", "points", "estimated_points_available", "free_machine", "owned_user", "owned_root", "release_readable"], "No unsolved active machines were returned.", image_cols=["avatar"])
with machine_tabs[3]:
    display_df(machine_rows, ["avatar", "id", "name", "os", "difficultyText", "points", "free_machine", "owned_user", "owned_root", "release_readable"], "No active machines were returned by the API.", image_cols=["avatar"])
with machine_tabs[4]:
    display_df(owned_rows, ["avatar", "id", "name", "os", "difficultyText", "points", "owned_user", "owned_root", "authUserFirstUserTime_readable", "authUserFirstRootTime_readable"], "No active owned machine rows returned.", image_cols=["avatar"])
with machine_tabs[5]:
    display_df(decorate_rows(todo_machines), ["avatar", "id", "name", "os", "difficultyText", "points", "authUserInUserOwns", "authUserInRootOwns", "stars"], "Your HTB machine ToDo list is empty.", image_cols=["avatar"])

st.subheader("Active challenges")
category_lookup = fetch_challenge_category_lookup(base, api_token)
challenge_results = fetch_all_candidates(base, api_token, "challenge_active")
active_challenge_rows: List[Dict[str, Any]] = []
challenge_diagnostics = []
for path, result in challenge_results:
    if result.data is not None:
        active_challenge_rows = first_list(result.data)
        if active_challenge_rows:
            st.caption(f"Active challenge data source: `{path}`")
            break
    challenge_diagnostics.append((path, result.error, result.url, result.status_code))

if active_challenge_rows:
    normalized_challenges = []
    for row in active_challenge_rows:
        rr = dict(row)
        rr["solved"] = challenge_solved(row)
        # Normalize category/difficulty names for filtering across payload variants.
        # HTB currently may return only challenge_category_id, so resolve that
        # through the categories endpoint, then fallback labels/assets.
        rr["category_id"] = challenge_category_id(rr)
        rr["category_display"] = challenge_category_name(rr, category_lookup) or "Uncategorized"
        rr["difficulty_display"] = challenge_difficulty_name(rr) or "—"
        cat_img = challenge_category_image(rr, category_lookup)
        if cat_img and rr.get("avatar") in (None, "") and rr.get("image") in (None, ""):
            rr["avatar"] = cat_img
        normalized_challenges.append(rr)
    normalized_challenges = decorate_rows(normalized_challenges)

    chall_df = pd.DataFrame(normalized_challenges)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        status_filter = st.selectbox("Challenge status", ["Unsolved", "All", "Solved"], index=0)
    with f2:
        category_values = sorted([str(x) for x in chall_df.get("category_display", pd.Series(dtype=str)).dropna().unique() if str(x)])
        category_filter = st.selectbox("Challenge category", ["All"] + category_values)
    with f3:
        difficulty_values = sorted([str(x) for x in chall_df.get("difficulty_display", pd.Series(dtype=str)).dropna().unique() if str(x)])
        difficulty_filter = st.selectbox("Challenge difficulty", ["All"] + difficulty_values)
    with f4:
        challenge_search = st.text_input("Challenge search", value="")

    filtered_challenges = chall_df.copy()
    if status_filter == "Unsolved" and "solved" in filtered_challenges.columns:
        filtered_challenges = filtered_challenges[~filtered_challenges["solved"].astype(bool)]
    elif status_filter == "Solved" and "solved" in filtered_challenges.columns:
        filtered_challenges = filtered_challenges[filtered_challenges["solved"].astype(bool)]
    if category_filter != "All" and "category_display" in filtered_challenges.columns:
        filtered_challenges = filtered_challenges[filtered_challenges["category_display"].astype(str) == category_filter]
    if difficulty_filter != "All" and "difficulty_display" in filtered_challenges.columns:
        filtered_challenges = filtered_challenges[filtered_challenges["difficulty_display"].astype(str) == difficulty_filter]
    if challenge_search.strip() and "name" in filtered_challenges.columns:
        filtered_challenges = filtered_challenges[filtered_challenges["name"].astype(str).str.contains(challenge_search.strip(), case=False, na=False)]

    st.caption(f"Showing {len(filtered_challenges)} of {len(chall_df)} active challenge rows after filters.")
    challenge_cols = ["avatar", "id", "name", "category_id", "category_display", "difficulty_display", "points", "solved", "likes", "dislikes", "release_date_readable"]
    visible_challenge_cols = [c for c in challenge_cols if c in filtered_challenges.columns]
    challenge_view = (
        filtered_challenges.copy()
        if st.session_state.get("show_raw_columns", False)
        else filtered_challenges[visible_challenge_cols].copy()
    )
    st.dataframe(
        challenge_view,
        use_container_width=True,
        hide_index=True,
        column_config={"avatar": st.column_config.ImageColumn("avatar", width="small")},
    )
else:
    st.info("No active challenge list returned. The profile challenge progress is still shown in the comparison table.")
    with st.expander("Challenge endpoint diagnostics"):
        for path, err, url, status in challenge_diagnostics:
            st.write(f"`{path}` → status={status}; error={err}; url={url}")

st.subheader("Recent activity")
st.caption("Uses `/api/v5/user/profile/activity/{user}` and follows `meta.lastPage`, so it can fetch all activity pages instead of only the first 15 rows.")
activity_rows: List[Dict[str, Any]] = []
activity_warnings = []
activity_page_diagnostics: List[Tuple[int, str, Optional[str], str, Optional[int]]] = []

for uid in user_ids:
    name = profiles.get(uid, {}).get("name") or profiles.get(uid, {}).get("username") or str(uid)
    raw_rows, diagnostics = fetch_v5_activity_pages(api_token, uid, max_pages=int(activity_pages_per_user))
    for path, err, url, status in diagnostics:
        activity_page_diagnostics.append((uid, path, err, url, status))
    if raw_rows:
        activity_rows.extend(normalize_v5_activity_rows(raw_rows, name, uid))
    else:
        activity_warnings.append((uid, "user/profile/activity", "no rows returned from v5 activity", f"{DEFAULT_BASE_V5}/user/profile/activity/{uid}", None))
        activity_rows.append({
            "date": "",
            "user": name,
            "user_id": uid,
            "event_type": "profile link",
            "type": "link",
            "target": "Open HTB activity tab",
            "target_id": None,
            "category": None,
            "points": None,
            "blood": None,
            "avatar": None,
            "source": "browser profile fallback",
            "profile_activity_url": f"{HTB_APP_BASE}/users/{uid}?tab=activity",
        })

if activity_rows:
    act_df = pd.DataFrame(activity_rows)
    if "date" in act_df.columns:
        act_df["_sort_date"] = pd.to_datetime(act_df.get("raw_date", act_df["date"]), errors="coerce", utc=True)
        act_df = act_df.sort_values("_sort_date", ascending=False, na_position="last").drop(columns=["_sort_date"])

    c1, c2, c3, c4 = st.columns(4)
    timestamped_df = act_df[act_df.get("date", "").astype(str) != ""] if "date" in act_df.columns else act_df
    c1.metric("Activity rows", len(timestamped_df))
    c2.metric("Challenge owns", int((timestamped_df.get("type") == "challenge").sum()) if "type" in timestamped_df.columns else 0)
    c3.metric("User owns", int((timestamped_df.get("type") == "user").sum()) if "type" in timestamped_df.columns else 0)
    c4.metric("Root owns", int((timestamped_df.get("type") == "root").sum()) if "type" in timestamped_df.columns else 0)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        type_options = ["All"] + sorted([str(x) for x in act_df.get("type", pd.Series(dtype=str)).dropna().unique()])
        type_filter = st.selectbox("Activity type", type_options)
    with col_b:
        user_options = ["All"] + sorted([str(x) for x in act_df.get("user", pd.Series(dtype=str)).dropna().unique()])
        user_filter = st.selectbox("Activity user", user_options)
    with col_c:
        max_rows = st.number_input("Rows to show", min_value=25, max_value=2000, value=25, step=25)

    filtered = act_df.copy()
    if type_filter != "All" and "type" in filtered.columns:
        filtered = filtered[filtered["type"].astype(str) == type_filter]
    if user_filter != "All" and "user" in filtered.columns:
        filtered = filtered[filtered["user"].astype(str) == user_filter]

    snapshot_enabled = st.checkbox("Save/update local activity snapshot CSV", value=True, help="Stores v5 activity events in ./htb_activity_snapshot.csv for your own long-term tracker.")
    if snapshot_enabled:
        save_activity_snapshot([r for r in activity_rows if r.get("date")])

    preferred = ["avatar", "date", "user", "type", "target", "category", "points", "blood", "target_id", "profile_activity_url", "source"]
    display_cols = [c for c in preferred if c in filtered.columns] or list(filtered.columns)[:10]
    st.dataframe(
        filtered[display_cols].head(int(max_rows)),
        use_container_width=True,
        hide_index=True,
        column_config={
            "avatar": st.column_config.ImageColumn("avatar", width="small"),
            "profile_activity_url": st.column_config.LinkColumn("profile_activity_url"),
        },
    )
else:
    st.info("No activity rows returned by the v5 activity endpoint.")

with st.expander("Activity endpoint diagnostics"):
    st.write("The v5 activity endpoint returns rows under `data` and pagination under `meta.lastPage`. This dashboard requests each page as `?page=N` and de-duplicates by type/id/ownDate/points/name.")
    for uid, path, err, url, status in activity_page_diagnostics:
        st.write(f"{uid}: `{path}` → status={status}; error={err}; url={url}")
    for uid, path, err, url, status in activity_warnings:
        st.write(f"{uid}: `{path}` → status={status}; note/error={err}; url={url}")

st.subheader("Progress history")
graph_key = "graph_week" if graph_range == "1W" else "graph_year"
series_rows = []
graph_warnings = []
for uid in user_ids:
    p = profiles.get(uid, {})
    name = p.get("name") or str(uid)
    result = fetch_endpoint(base, api_token, graph_key, uid)
    if result.error:
        graph_warnings.append((uid, result.error, result.url))
        continue
    payload = profile_payload(result.data)
    gd = payload.get("graphData") or payload.get("graph_data") or payload.get("data") or {}
    if isinstance(gd, list):
        for i, item in enumerate(gd):
            if isinstance(item, dict):
                for metric, value in item.items():
                    if metric not in ("date", "timestamp"):
                        series_rows.append({"user": name, "id": uid, "metric": metric, "step": item.get("date") or item.get("timestamp") or str(i), "value": value})
    elif isinstance(gd, dict):
        for metric, values in gd.items():
            if isinstance(values, dict):
                iterable = sorted(values.items(), key=lambda x: str(x[0]))
            elif isinstance(values, list):
                iterable = list(enumerate(values))
            else:
                continue
            for step, value in iterable:
                series_rows.append({"user": name, "id": uid, "metric": metric, "step": str(step), "value": value})
if graph_warnings:
    with st.expander("Graph endpoint diagnostics"):
        for uid, err, url in graph_warnings:
            st.write(f"{uid}: {err} — {url}")
if series_rows:
    graph_df = pd.DataFrame(series_rows)
    metric_choice = st.selectbox("Metric", sorted(graph_df["metric"].unique()), index=0)
    filtered = graph_df[graph_df["metric"] == metric_choice]
    fig = px.line(filtered, x="step", y="value", color="user", markers=True, title=f"{metric_choice} over {graph_range}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No graph data returned by the API for the selected users/range.")

left, right = st.columns(2)
with left:
    st.subheader("Fortresses")
    r = fetch_endpoint(base, api_token, "fortress", me_id)
    rows = get_nested_list(r.data, ["fortresses"])
    if rows:
        display_df(decorate_rows(rows), ["image", "id", "name", "completion_percentage", "owned_flags", "total_flags", "progress", "points"], "No fortress progress returned.", image_cols=["image"])
    else:
        st.info("No fortress progress returned.")
        if r.error:
            st.caption(f"Endpoint: {r.error} — {r.url}")
with right:
    st.subheader("Pro Labs")
    r = fetch_endpoint(base, api_token, "prolab", me_id)
    rows = get_nested_list(r.data, ["prolabs"])
    if rows:
        display_df(decorate_rows(rows), ["image", "id", "name", "completion_percentage", "owned_flags", "total_flags", "progress", "points"], "No pro lab progress returned.", image_cols=["image"])
    else:
        st.info("No pro lab progress returned.")
        if r.error:
            st.caption(f"Endpoint: {r.error} — {r.url}")

with st.expander("Content submitted, bloods, and raw endpoint diagnostics"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Content")
        r = fetch_endpoint(base, api_token, "content", me_id)
        content = profile_payload(r.data).get("content", {})
        if isinstance(content, dict) and content:
            for name, rows in content.items():
                st.markdown(f"#### {name.title()}")
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.write("None")
        else:
            st.info("No content data returned.")
            if r.error:
                st.caption(f"Endpoint: {r.error} — {r.url}")
    with c2:
        st.markdown("### Bloods")
        r = fetch_endpoint(base, api_token, "bloods", me_id)
        bloods = profile_payload(r.data).get("bloods", {})
        if isinstance(bloods, dict) and bloods:
            for name, rows in bloods.items():
                st.markdown(f"#### {name.title()}")
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.write("None")
        else:
            st.info("No blood data returned.")
            if r.error:
                st.caption(f"Endpoint: {r.error} — {r.url}")
    st.markdown("### Raw endpoint tester")
    raw_api_version = st.radio("Raw endpoint API version", ["v4", "v5"], horizontal=True)
    raw_path = st.text_input("Path under selected API base", value=f"user/profile/basic/{me_id}")
    if st.button("Test raw endpoint"):
        raw_base = DEFAULT_BASE_V5 if raw_api_version == "v5" else base
        rr = fetch_url(raw_base, api_token, raw_path)
        st.write(f"URL: {rr.url}")
        st.write(f"Status: {rr.status_code}; Error: {rr.error}")
        st.json(rr.data or {})

st.caption("Data is cached for 5 minutes. Use Refresh cache after new owns. Machine pagination and v5/derived activity tracking are best-effort because HTB API payloads vary by endpoint and account visibility.")
