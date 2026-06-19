from __future__ import annotations

import time
import urllib.parse
import urllib.error
import urllib.robotparser
import urllib.request
from dataclasses import dataclass, field


@dataclass
class RobotsCache:
    user_agent: str
    ttl_seconds: int = 3600
    _cache: dict[str, tuple[float, urllib.robotparser.RobotFileParser]] = field(default_factory=dict)

    def allowed(self, url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        origin = f"{parsed.scheme}://{parsed.netloc}"
        now = time.time()
        cached = self._cache.get(origin)
        if cached and now - cached[0] < self.ttl_seconds:
            parser = cached[1]
        else:
            parser = urllib.robotparser.RobotFileParser()
            robots_url = urllib.parse.urljoin(origin, "/robots.txt")
            parser.set_url(robots_url)
            request = urllib.request.Request(robots_url, headers={"User-Agent": self.user_agent})
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    status = getattr(response, "status", 200)
                    if status in {401, 403}:
                        parser.disallow_all = True
                    elif status >= 400:
                        parser.allow_all = True
                    else:
                        lines = response.read().decode("utf-8", errors="ignore").splitlines()
                        parser.parse(lines)
            except urllib.error.HTTPError as exc:
                if exc.code in {401, 403}:
                    parser.disallow_all = True
                else:
                    parser.allow_all = True
            except Exception:
                parser.allow_all = True
            self._cache[origin] = (now, parser)
        return parser.can_fetch(self.user_agent, url)
