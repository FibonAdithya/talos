"""Advisory GHCR version discovery. Reads metadata only; never changes a build pin."""
from __future__ import annotations

import http.client
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

from talos.challenges import CHALLENGES, DEV_IMAGE_TAG, dev_image

CHECK_TIMEOUT_S = 5.0
MAX_TAG_PAGES = 20
_VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


class UpdateCheckError(RuntimeError):
    pass


def _version(tag: str) -> tuple[int, ...] | None:
    match = _VERSION.fullmatch(tag)
    return tuple(map(int, match.groups())) if match else None


@dataclass(frozen=True)
class ImageUpdate:
    challenge: str
    current: str
    latest: str

    @property
    def available(self) -> bool:
        return _version(self.latest) > _version(self.current)


def _fetch_json(url: str, headers: dict, timeout: float) -> tuple[dict, str]:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response), response.headers.get("Link", "")


def check_update(challenge: str, fetch=None, clock=time.monotonic) -> ImageUpdate:
    """Newest published X.Y.Z tag for one challenge, excluding aliases and prereleases.

    GHCR paginates tags in lexical order, so scan all pages and compare versions numerically.
    A partial or failed lookup is unknown, never an assertion that the pin is current.
    The deadline is shared by token acquisition and every page request.
    """
    if challenge not in CHALLENGES:
        raise ValueError(f"unknown challenge {challenge!r}")
    if _version(DEV_IMAGE_TAG) is None:
        raise UpdateCheckError("the configured image tag is not an X.Y.Z version")
    fetch = fetch or _fetch_json
    deadline = clock() + CHECK_TIMEOUT_S

    def get(url, headers):
        remaining = deadline - clock()
        if remaining <= 0:
            raise UpdateCheckError("GHCR update check timed out")
        try:
            doc, link = fetch(url, headers, remaining)
        except (OSError, http.client.HTTPException, ValueError):
            raise UpdateCheckError("could not read GHCR image metadata") from None
        if not isinstance(doc, dict):
            raise UpdateCheckError("invalid GHCR image metadata")
        return doc, link

    name = dev_image(challenge).removeprefix("ghcr.io/").rsplit(":", 1)[0]
    token_doc, _ = get(f"https://ghcr.io/token?scope=repository:{name}:pull", {})
    token = token_doc.get("token")
    if not isinstance(token, str) or not token:
        raise UpdateCheckError("could not obtain a GHCR pull token")
    headers = {"Authorization": f"Bearer {token}"}
    endpoint = f"https://ghcr.io/v2/{name}/tags/list"
    url = endpoint + "?n=100"
    seen, versions = set(), set()
    while url:
        if url in seen or len(seen) >= MAX_TAG_PAGES:
            raise UpdateCheckError("GHCR tag listing did not finish")
        seen.add(url)
        doc, link = get(url, headers)
        tags = doc.get("tags")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise UpdateCheckError("invalid GHCR tag listing")
        versions.update(tag for tag in tags if _version(tag) is not None)
        next_link = re.search(r'<([^>]+)>[^,]*;\s*rel="?next"?', link)
        url = urllib.parse.urljoin(url, next_link[1]) if next_link else ""
        if url:
            # A registry pagination link must never forward the pull token to another host
            # or repository. Query parameters may change; the endpoint must stay identical.
            parsed = urllib.parse.urlsplit(url)
            if urllib.parse.urlunsplit(parsed._replace(query="", fragment="")) != endpoint:
                raise UpdateCheckError("unexpected GHCR pagination endpoint")
    if not versions:
        raise UpdateCheckError("no stable X.Y.Z image tags published")
    return ImageUpdate(challenge, DEV_IMAGE_TAG, max(versions, key=_version))
