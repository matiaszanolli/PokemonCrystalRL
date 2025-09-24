"""
Authentication and Security for Pokemon Crystal RL REST API

Provides basic authentication, rate limiting, and security features
for production deployment of the REST API.
"""

import time
import hashlib
import hmac
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class ApiKey:
    """API key with metadata"""
    key_id: str
    key_hash: str
    name: str
    permissions: list
    created_at: float
    last_used: float = 0.0
    request_count: int = 0
    rate_limit: int = 1000  # requests per hour


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
        self.lock = threading.Lock()

    def is_allowed(self, key: str) -> Tuple[bool, Dict[str, any]]:
        """Check if request is allowed under rate limit"""
        current_time = time.time()

        with self.lock:
            # Clean old requests
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if current_time - req_time < self.window_seconds
            ]

            # Check rate limit
            request_count = len(self.requests[key])
            if request_count >= self.max_requests:
                return False, {
                    "error": "Rate limit exceeded",
                    "limit": self.max_requests,
                    "window": self.window_seconds,
                    "retry_after": int(self.window_seconds - (current_time - self.requests[key][0]))
                }

            # Add current request
            self.requests[key].append(current_time)

            return True, {
                "requests_remaining": self.max_requests - request_count - 1,
                "reset_time": int(current_time + self.window_seconds)
            }


class ApiAuthenticator:
    """API authentication and authorization system"""

    def __init__(self, enable_auth: bool = False):
        self.enable_auth = enable_auth
        self.api_keys: Dict[str, ApiKey] = {}
        self.rate_limiters = {
            "default": RateLimiter(100, 3600),  # 100 requests per hour
            "training": RateLimiter(20, 3600),   # 20 training operations per hour
            "plugins": RateLimiter(50, 3600),    # 50 plugin operations per hour
        }

        # Default development key (only if auth disabled)
        if not enable_auth:
            self._create_development_key()

    def _create_development_key(self):
        """Create a default development API key"""
        dev_key = ApiKey(
            key_id="dev",
            key_hash="dev_key_hash",
            name="Development Key",
            permissions=["*"],  # All permissions
            created_at=time.time(),
            rate_limit=10000  # High limit for development
        )
        self.api_keys["dev"] = dev_key

    def create_api_key(self, name: str, permissions: list = None) -> Tuple[str, ApiKey]:
        """Create a new API key"""
        if permissions is None:
            permissions = ["read"]

        # Generate key ID and secret
        key_id = f"pk_{int(time.time())}"
        key_secret = hashlib.sha256(f"{key_id}{time.time()}".encode()).hexdigest()[:32]
        key_hash = hashlib.sha256(key_secret.encode()).hexdigest()

        api_key = ApiKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            permissions=permissions,
            created_at=time.time()
        )

        self.api_keys[key_id] = api_key

        # Return the full key (ID + secret) - only shown once
        full_key = f"{key_id}.{key_secret}"
        return full_key, api_key

    def authenticate_request(self, auth_header: Optional[str]) -> Tuple[bool, Optional[ApiKey], Dict[str, any]]:
        """Authenticate API request"""
        if not self.enable_auth:
            # Development mode - allow all requests
            return True, self.api_keys.get("dev"), {}

        if not auth_header:
            return False, None, {"error": "Missing authentication header"}

        if not auth_header.startswith("Bearer "):
            return False, None, {"error": "Invalid authentication format"}

        # Extract key
        full_key = auth_header[7:]  # Remove "Bearer "
        if "." not in full_key:
            return False, None, {"error": "Invalid API key format"}

        key_id, key_secret = full_key.split(".", 1)

        # Find API key
        if key_id not in self.api_keys:
            return False, None, {"error": "Invalid API key"}

        api_key = self.api_keys[key_id]

        # Verify key
        key_hash = hashlib.sha256(key_secret.encode()).hexdigest()
        if not hmac.compare_digest(api_key.key_hash, key_hash):
            return False, None, {"error": "Invalid API key"}

        # Update usage
        api_key.last_used = time.time()
        api_key.request_count += 1

        return True, api_key, {}

    def check_permission(self, api_key: ApiKey, permission: str) -> bool:
        """Check if API key has required permission"""
        if not api_key:
            return False

        # Wildcard permission
        if "*" in api_key.permissions:
            return True

        # Specific permission
        return permission in api_key.permissions

    def check_rate_limit(self, api_key: ApiKey, endpoint_type: str = "default") -> Tuple[bool, Dict[str, any]]:
        """Check rate limit for API key and endpoint type"""
        if not api_key:
            return False, {"error": "No API key provided"}

        # Get appropriate rate limiter
        rate_limiter = self.rate_limiters.get(endpoint_type, self.rate_limiters["default"])

        # Use key ID for rate limiting
        return rate_limiter.is_allowed(api_key.key_id)

    def get_api_key_info(self, key_id: str) -> Optional[Dict[str, any]]:
        """Get API key information (without sensitive data)"""
        if key_id not in self.api_keys:
            return None

        api_key = self.api_keys[key_id]
        return {
            "key_id": api_key.key_id,
            "name": api_key.name,
            "permissions": api_key.permissions,
            "created_at": api_key.created_at,
            "last_used": api_key.last_used,
            "request_count": api_key.request_count,
            "rate_limit": api_key.rate_limit
        }

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key"""
        if key_id in self.api_keys:
            del self.api_keys[key_id]
            return True
        return False

    def list_api_keys(self) -> list:
        """List all API keys (without sensitive data)"""
        return [self.get_api_key_info(key_id) for key_id in self.api_keys.keys()]


def create_auth_middleware():
    """Create authentication middleware for the web server"""
    authenticator = ApiAuthenticator(enable_auth=False)  # Default: development mode

    def authenticate_request(request_headers: dict, endpoint_path: str) -> Tuple[bool, Dict[str, any]]:
        """Middleware function to authenticate requests"""
        try:
            # Get authorization header
            auth_header = request_headers.get('Authorization') or request_headers.get('authorization')

            # Authenticate
            is_auth, api_key, auth_error = authenticator.authenticate_request(auth_header)
            if not is_auth:
                return False, auth_error

            # Check permissions based on endpoint
            required_permission = _get_required_permission(endpoint_path)
            if not authenticator.check_permission(api_key, required_permission):
                return False, {"error": f"Insufficient permissions. Required: {required_permission}"}

            # Check rate limits
            endpoint_type = _get_endpoint_type(endpoint_path)
            is_allowed, rate_info = authenticator.check_rate_limit(api_key, endpoint_type)
            if not is_allowed:
                return False, rate_info

            return True, {
                "api_key": api_key.key_id if api_key else None,
                "rate_limit_info": rate_info
            }

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False, {"error": "Authentication system error"}

    return authenticate_request, authenticator


def _get_required_permission(endpoint_path: str) -> str:
    """Determine required permission based on endpoint path"""
    if "/training/sessions" in endpoint_path and endpoint_path.endswith("/control"):
        return "training:control"
    elif "/training/sessions" in endpoint_path:
        return "training:read"
    elif "/agents" in endpoint_path and "/control" in endpoint_path:
        return "agents:control"
    elif "/agents" in endpoint_path:
        return "agents:read"
    elif "/plugins" in endpoint_path and "/control" in endpoint_path:
        return "plugins:control"
    elif "/plugins" in endpoint_path:
        return "plugins:read"
    elif endpoint_path.endswith("/docs"):
        return "docs:read"
    else:
        return "read"


def _get_endpoint_type(endpoint_path: str) -> str:
    """Determine endpoint type for rate limiting"""
    if "/training/" in endpoint_path:
        return "training"
    elif "/plugins/" in endpoint_path:
        return "plugins"
    else:
        return "default"