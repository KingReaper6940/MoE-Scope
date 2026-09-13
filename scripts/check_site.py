"""Check static internal links, downloadable artifacts, and journal coverage."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote
import os

ROOT = Path(__file__).resolve().parents[1] / "web/dist"
BASE = os.environ.get("SITE_BASE", "/").rstrip("/")


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.hrefs.append(attrs["href"])


def main():
    pages = {}
    for path in ROOT.rglob("*.html"):
        parser = Links()
        parser.feed(path.read_text(encoding="utf-8"))
        pages[path.resolve()] = parser
    assert len(list((ROOT / "writing").glob("*/index.html"))) == 10, "expected ten journal chapters"
    failures = []
    for page, parsed in pages.items():
        for href in parsed.hrefs:
            url = urlsplit(href)
            if url.scheme or url.netloc:
                continue
            if url.path.startswith("/") and BASE:
                if not url.path.startswith(BASE + "/"):
                    failures.append(f"{page.relative_to(ROOT)}: link escapes site base: {href}")
                    continue
                url = url._replace(path=url.path[len(BASE):])
            target = ((ROOT / unquote(url.path).lstrip("/")) if url.path.startswith("/")
                      else page.parent / unquote(url.path)) if url.path else page
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                failures.append(f"{page.relative_to(ROOT)}: missing {href}")
            elif url.fragment and target.resolve() in pages and unquote(url.fragment) not in pages[target.resolve()].ids:
                failures.append(f"{page.relative_to(ROOT)}: missing anchor {href}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified internal links and anchors across {len(pages)} pages; ten journal chapters")


if __name__ == "__main__":
    main()
