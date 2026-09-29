import json
import urllib.error

import pytest

from talos import cli, image_updates
from talos.challenges import CHALLENGES
from talos.image_updates import ImageUpdate, UpdateCheckError, check_update


@pytest.fixture(autouse=True)
def _fixed_version(monkeypatch):
    monkeypatch.setattr(image_updates, "DEV_IMAGE_TAG", "0.0.8")


def registry(pages):
    calls = []
    pages = iter(pages)

    def fetch(url, headers, timeout):
        calls.append((url, headers, timeout))
        assert 0 < timeout <= image_updates.CHECK_TIMEOUT_S
        if url.startswith("https://ghcr.io/token?"):
            assert headers == {}
            return {"token": "pull-token"}, ""
        assert headers == {"Authorization": "Bearer pull-token"}
        page = next(pages)
        if isinstance(page, Exception):
            raise page
        return page

    return fetch, calls


def test_checks_numeric_versions_across_pages_and_ignores_aliases_and_prereleases():
    fetch, calls = registry([
        ({"tags": ["latest", "0.0.10", "0.0.99-rc1", "0.0.99-arm64"]},
         '</v2/tig-foundation/tig-monorepo/knapsack/dev/tags/list?n=100&last=0.0.10>; '
         'rel="next"'),
        ({"tags": ["0.0.8", "0.0.9", "0.1", "main", "01.0.0"]}, ""),
    ])
    update = check_update("knapsack", fetch=fetch)
    assert update == ImageUpdate("knapsack", "0.0.8", "0.0.10")
    assert update.available
    assert calls[0][0] == ("https://ghcr.io/token?scope=repository:"
                           "tig-foundation/tig-monorepo/knapsack/dev:pull")
    assert calls[1][0].endswith("/knapsack/dev/tags/list?n=100")
    assert calls[2][0].endswith("last=0.0.10")


@pytest.mark.parametrize("tag", ["0.0.7", "0.0.8"])
def test_equal_or_older_release_is_not_an_update(tag):
    fetch, _ = registry([({"tags": [tag]}, "")])
    assert not check_update("knapsack", fetch=fetch).available


@pytest.mark.parametrize("doc", [None, [], {}, {"tags": None}, {"tags": [8]},
                                 {"tags": []}, {"tags": ["latest", "0.0.9-amd64"]}])
def test_bad_or_empty_listing_is_unknown_not_up_to_date(doc):
    fetch, _ = registry([(doc, "")])
    with pytest.raises(UpdateCheckError):
        check_update("knapsack", fetch=fetch)


@pytest.mark.parametrize("doc", [[], {}, {"token": None}, {"token": ""}])
def test_invalid_token_response_stops_before_listing(doc):
    with pytest.raises(UpdateCheckError):
        check_update("knapsack", fetch=lambda *args: (doc, ""))


def test_failure_on_later_page_does_not_report_a_partial_version():
    fetch, _ = registry([
        ({"tags": ["0.0.9"]}, '<?n=100&last=0.0.9>; rel="next"'),
        urllib.error.URLError("offline"),
    ])
    with pytest.raises(UpdateCheckError):
        check_update("knapsack", fetch=fetch)


@pytest.mark.parametrize("link", [
    "https://other.example/tags/list", "http://ghcr.io/v2/x/tags/list",
    "/v2/tig-foundation/tig-monorepo/hypergraph/dev/tags/list",
])
def test_pagination_cannot_forward_the_pull_token_to_another_endpoint(link):
    fetch, calls = registry([({"tags": ["0.0.9"]}, f'<{link}>; rel="next"')])
    with pytest.raises(UpdateCheckError, match="pagination endpoint"):
        check_update("knapsack", fetch=fetch)
    assert len(calls) == 2


def test_cyclic_pagination_is_unknown():
    fetch, _ = registry([({"tags": ["0.0.9"]}, '<?n=100>; rel="next"')])
    with pytest.raises(UpdateCheckError, match="did not finish"):
        check_update("knapsack", fetch=fetch)


def test_timeout_budget_is_shared_across_requests():
    fetch, calls = registry([({"tags": ["0.0.9"]}, '<?last=0.0.9>; rel="next"')])
    ticks = iter([0.0, 0.0, 4.0, 6.0])
    with pytest.raises(UpdateCheckError, match="timed out"):
        check_update("knapsack", fetch=fetch, clock=lambda: next(ticks))
    assert [call[2] for call in calls] == [5.0, 1.0]


@pytest.mark.parametrize("failure", [TimeoutError(), urllib.error.URLError("offline"),
                                    json.JSONDecodeError("invalid", "", 0)])
def test_unreadable_metadata_is_unknown(failure):
    def fetch(*args):
        raise failure
    with pytest.raises(UpdateCheckError):
        check_update("knapsack", fetch=fetch)


def test_manual_check_needs_no_config_and_reports_each_challenge(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(image_updates, "check_update", lambda name:
                        ImageUpdate(name, "0.0.8", "0.0.10" if name == "knapsack" else "0.0.8"))
    assert cli.main(["check-updates"]) == 0
    out = capsys.readouterr().out
    for name in CHALLENGES:
        assert f"{name}: configured 0.0.8," in out
    assert "newest published 0.0.10 (update available)" in out
    assert "MONOREPO_REF" in out and "talos setup" in out
    assert not list(tmp_path.iterdir())


def test_manual_failure_is_reported_as_unknown(monkeypatch, capsys):
    def fail(name):
        raise UpdateCheckError("offline")
    monkeypatch.setattr(image_updates, "check_update", fail)
    assert cli.main(["check-updates", "--challenge", "knapsack"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "knapsack: unable to check GHCR; update status unknown" in captured.err


def test_startup_advisory_keeps_the_pin_and_ignores_registry_failure(monkeypatch, capsys):
    monkeypatch.setattr(image_updates, "check_update", lambda name:
                        ImageUpdate(name, "0.0.8", "0.0.10"))
    cli.recommend_image_update("knapsack")
    err = capsys.readouterr().err
    assert "0.0.8 -> 0.0.10" in err and "This job will use the pinned image 0.0.8" in err
    assert image_updates.DEV_IMAGE_TAG == "0.0.8"
    def fail(name):
        raise UpdateCheckError("offline")
    monkeypatch.setattr(image_updates, "check_update", fail)
    cli.recommend_image_update("knapsack")
    assert capsys.readouterr().err == ""
