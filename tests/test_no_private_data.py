"""No file in the repository carries an e-mail address or a path of the machine it was made on."""
import re
import subprocess

from cholec4d.data import REPO

AT = "@"
PATTERNS = [
    re.compile(r"[A-Za-z0-9._%+-]+" + AT + r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile("/" + "home/|/" + "var/autofs|/" + "Users/|/" + "work/"),
]
# Package pins name their source by URL, which is not an address.
ALLOWED = ("git+https://",)


def tracked_files():
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [REPO / p for p in out.splitlines()]


def test_no_address_or_local_path():
    found = []
    for path in tracked_files():
        try:
            text = path.read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if any(a in line for a in ALLOWED):
                continue
            if any(p.search(line) for p in PATTERNS):
                found.append(f"{path.relative_to(REPO)}:{n}: {line.strip()[:80]}")
    assert not found, "\n".join(found)
