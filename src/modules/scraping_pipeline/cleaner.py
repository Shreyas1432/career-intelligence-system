import html
import re
from html.parser import HTMLParser


class HTMLTextExtractor(HTMLParser):
    """
    Standard library-based HTML parser to extract clean visible text.
    """

    def __init__(self) -> None:
        super().__init__()
        self.reset()
        self.fed: list[str] = []
        self.ignore_tags = {
            "script",
            "style",
            "noscript",
            "svg",
            "form",
            "nav",
            "iframe",
            "header",
            "footer",
            "aside",
        }
        self.current_ignore_depth = 0
        self.tag_stack: list[str] = []

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self.tag_stack.append(tag)
        if tag in self.ignore_tags:
            self.current_ignore_depth += 1
        # Add spacing/line breaks for structural block elements
        if tag in {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}:
            if self.current_ignore_depth == 0:
                self.fed.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.tag_stack:
            if tag in self.tag_stack:
                while self.tag_stack:
                    popped = self.tag_stack.pop()
                    if popped in self.ignore_tags:
                        self.current_ignore_depth -= 1
                    if popped == tag:
                        break
            else:
                popped = self.tag_stack.pop()
                if popped in self.ignore_tags:
                    self.current_ignore_depth -= 1
        # Add spacing/line breaks for structural block elements
        if tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}:
            if self.current_ignore_depth == 0:
                self.fed.append("\n")

    def handle_data(self, data: str) -> None:
        if self.current_ignore_depth == 0:
            self.fed.append(data)

    def get_data(self) -> str:
        return "".join(self.fed)


def clean_html(html_content: str) -> str:
    """
    Extracts visible text from HTML content, discarding script/style/nav blocks
    and compressing whitespace.
    """
    if not html_content:
        return ""

    unescaped = html.unescape(html_content)

    parser = HTMLTextExtractor()
    try:
        parser.feed(unescaped)
        text = parser.get_data()
    except Exception:
        # Fallback to simple regex tag stripping if parsing fails
        text = re.sub(r"<[^>]+>", " ", unescaped)

    return clean_text(text)


def clean_text(text: str) -> str:
    """
    Normalizes whitespaces, resolves double spaces, and cleans up paragraph transitions.
    """
    if not text:
        return ""

    lines = []
    for line in text.splitlines():
        cleaned_line = line.strip()
        if cleaned_line:
            # Compress multiple spaces/tabs within a single line
            cleaned_line = re.sub(r"[ \t]+", " ", cleaned_line)
            lines.append(cleaned_line)

    # Rejoin with standard single newlines, ensuring we don't have large empty gaps
    # but still separating paragraphs cleanly.
    return "\n".join(lines)
