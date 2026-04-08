#!/usr/bin/env python3
"""
nyt_recipe_to_pdf.py
────────────────────
Batch-download NYT Cooking recipes and save each one as a beautifully
formatted PDF.

Requires:
    pip install nyt-recipe reportlab requests beautifulsoup4 lxml Pillow

Usage examples
──────────────
# From a plain-text file of recipe URLs (one per line):
    python nyt_recipe_to_pdf.py --urls-file my_recipes.txt

# Pass URLs directly on the command line:
    python nyt_recipe_to_pdf.py \
        https://cooking.nytimes.com/recipes/1020044-vegetable-paella \
        https://cooking.nytimes.com/recipes/1018639-chocolate-chip-cookies

# Choose a custom output folder:
    python nyt_recipe_to_pdf.py --urls-file my_recipes.txt --output ~/RecipePDFs

Authentication
──────────────
NYT Cooking requires a subscription.  Supply your session cookie so the
scraper can access paywalled content.  The easiest way:

  1. Log in at https://cooking.nytimes.com in your browser.
  2. Open DevTools → Application → Cookies → https://cooking.nytimes.com
  3. Copy the value of the cookie named  NYT-S  (or  nyt-s)
  4. Pass it:
       --nyt-cookie  "YOUR_COOKIE_VALUE"
     or set the environment variable:
       export NYT_COOKIE="YOUR_COOKIE_VALUE"
"""

import argparse
import os
import re
import sys
import textwrap
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

# ── third-party ──────────────────────────────────────────────────────────────
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("Missing dependencies.  Run:  pip install requests beautifulsoup4 lxml")

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable,
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
except ImportError:
    sys.exit("Missing reportlab.  Run:  pip install reportlab")

try:
    from PIL import Image as PILImage  # for image resizing
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ── colour palette ────────────────────────────────────────────────────────────
NYT_RED    = colors.HexColor("#D0021B")
NYT_BLACK  = colors.HexColor("#121212")
NYT_GREY   = colors.HexColor("#6B6B6B")
NYT_LGREY  = colors.HexColor("#F2F2F2")
NYT_BORDER = colors.HexColor("#DEDEDE")
WHITE      = colors.white


# ── styles ────────────────────────────────────────────────────────────────────
def build_styles():
    base = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "RecipeTitle",
        parent=base["Title"],
        fontName="Times-Bold",
        fontSize=26,
        leading=30,
        textColor=NYT_BLACK,
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    byline_style = ParagraphStyle(
        "Byline",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=NYT_GREY,
        spaceAfter=2,
        alignment=TA_CENTER,
    )
    meta_style = ParagraphStyle(
        "Meta",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=NYT_GREY,
        spaceAfter=0,
        alignment=TA_CENTER,
    )
    description_style = ParagraphStyle(
        "Description",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=16,
        textColor=NYT_BLACK,
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    section_header_style = ParagraphStyle(
        "SectionHeader",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=NYT_BLACK,
        spaceBefore=14,
        spaceAfter=6,
        borderPad=0,
        leftIndent=0,
    )
    ingredient_style = ParagraphStyle(
        "Ingredient",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=NYT_BLACK,
        leftIndent=12,
        spaceAfter=1,
    )
    step_label_style = ParagraphStyle(
        "StepLabel",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=NYT_RED,
        leading=14,
    )
    step_text_style = ParagraphStyle(
        "StepText",
        parent=base["Normal"],
        fontName="Times-Roman",
        fontSize=10,
        leading=15,
        textColor=NYT_BLACK,
        spaceAfter=6,
    )
    note_style = ParagraphStyle(
        "Note",
        parent=base["Normal"],
        fontName="Times-Italic",
        fontSize=10,
        leading=14,
        textColor=NYT_GREY,
        leftIndent=12,
        spaceAfter=4,
    )
    footer_style = ParagraphStyle(
        "Footer",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=8,
        textColor=NYT_GREY,
        alignment=TA_CENTER,
    )

    return dict(
        title=title_style,
        byline=byline_style,
        meta=meta_style,
        description=description_style,
        section_header=section_header_style,
        ingredient=ingredient_style,
        step_label=step_label_style,
        step_text=step_text_style,
        note=note_style,
        footer=footer_style,
    )


# ── scraping ──────────────────────────────────────────────────────────────────
class RecipeScraper:
    """Scrapes a single NYT Cooking recipe page into a structured dict."""

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def __init__(self, cookie: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        if cookie:
            self.session.cookies.set("NYT-S", cookie, domain=".nytimes.com")

    def fetch(self, url: str) -> dict:
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return self._parse(resp.text, url)

    def _parse(self, html: str, url: str) -> dict:
        soup = BeautifulSoup(html, "lxml")

        data = {
            "url": url,
            "title": "",
            "byline": "",
            "description": "",
            "time_total": "",
            "time_prep": "",
            "time_cook": "",
            "yield": "",
            "rating": "",
            "image_url": "",
            "ingredient_groups": [],   # list of {"name": str, "items": [str]}
            "steps": [],               # list of str
            "notes": [],               # list of str
        }

        # ── title ─────────────────────────────────────────────────────────────
        h1 = soup.find("h1")
        if h1:
            data["title"] = h1.get_text(strip=True)

        # ── byline / author ───────────────────────────────────────────────────
        for sel in [
            '[class*="byline"]',
            '[class*="author"]',
            '[itemprop="author"]',
        ]:
            tag = soup.select_one(sel)
            if tag:
                data["byline"] = tag.get_text(strip=True)
                break

        # ── description ───────────────────────────────────────────────────────
        for sel in [
            '[class*="topnote"]',
            '[class*="recipe-introduction"]',
            '[class*="recipeYield"] ~ p',
            '[itemprop="description"]',
        ]:
            tag = soup.select_one(sel)
            if tag:
                data["description"] = tag.get_text(" ", strip=True)
                break

        # ── timing & yield ────────────────────────────────────────────────────
        for tag in soup.select('[class*="recipeYield"],[itemprop="recipeYield"]'):
            text = tag.get_text(strip=True)
            if text:
                data["yield"] = text
                break

        for tag in soup.select('[class*="recipeTotalTime"],[itemprop="totalTime"]'):
            data["time_total"] = tag.get_text(strip=True)
            break
        for tag in soup.select('[class*="recipePrepTime"],[itemprop="prepTime"]'):
            data["time_prep"] = tag.get_text(strip=True)
            break
        for tag in soup.select('[class*="recipeCookTime"],[itemprop="cookTime"]'):
            data["time_cook"] = tag.get_text(strip=True)
            break

        # ── hero image ────────────────────────────────────────────────────────
        for sel in [
            'img[class*="recipeImage"]',
            'img[class*="recipe-image"]',
            '[class*="media-container"] img',
            'figure img',
        ]:
            img = soup.select_one(sel)
            if img and img.get("src", "").startswith("http"):
                data["image_url"] = img["src"]
                break
            if img and img.get("data-src", "").startswith("http"):
                data["image_url"] = img["data-src"]
                break

        # ── ingredients ───────────────────────────────────────────────────────
        # Try structured ingredient groups first
        for group_el in soup.select(
            '[class*="ingredientGroup"],[class*="ingredient-group"]'
        ):
            name_el = group_el.select_one(
                '[class*="ingredientGroupName"],[class*="group-name"]'
            )
            name = name_el.get_text(strip=True) if name_el else ""
            items = [
                li.get_text(" ", strip=True)
                for li in group_el.select("li,[class*='ingredient']")
                if li.get_text(strip=True)
            ]
            if items:
                data["ingredient_groups"].append({"name": name, "items": items})

        # Fallback: flat list
        if not data["ingredient_groups"]:
            flat = [
                li.get_text(" ", strip=True)
                for li in soup.select(
                    '[class*="ingredient"] li, [itemprop="recipeIngredient"]'
                )
                if li.get_text(strip=True)
            ]
            if flat:
                data["ingredient_groups"].append({"name": "", "items": flat})

        # ── steps ─────────────────────────────────────────────────────────────
        for step_el in soup.select(
            '[class*="step"] p, [class*="preparation-step"] p, '
            '[itemprop="recipeInstructions"] li, '
            '[class*="instructions"] li'
        ):
            text = step_el.get_text(" ", strip=True)
            if text and len(text) > 10:
                data["steps"].append(text)

        # ── notes ─────────────────────────────────────────────────────────────
        for note_el in soup.select('[class*="tip"] p, [class*="note"] p'):
            text = note_el.get_text(" ", strip=True)
            if text and len(text) > 5:
                data["notes"].append(text)

        return data


# ── image helper ──────────────────────────────────────────────────────────────
def fetch_image(url: str, session: requests.Session, max_w: float, max_h: float):
    """Download image and return a ReportLab Image flowable, or None."""
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        buf = BytesIO(resp.content)
        if HAS_PIL:
            pil = PILImage.open(buf)
            w, h = pil.size
            scale = min(max_w / w, max_h / h, 1.0)
            return Image(BytesIO(resp.content), width=w * scale, height=h * scale)
        else:
            return Image(BytesIO(resp.content), width=max_w, height=max_h)
    except Exception:
        return None


# ── PDF builder ───────────────────────────────────────────────────────────────
def build_pdf(recipe: dict, out_path: Path, styles: dict) -> None:
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        rightMargin=0.85 * inch,
        leftMargin=0.85 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.75 * inch,
        title=recipe["title"],
    )

    story = []
    S = styles  # shorthand

    # ── masthead rule ─────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=2, color=NYT_RED, spaceAfter=10))

    # ── title ─────────────────────────────────────────────────────────────────
    if recipe["title"]:
        story.append(Paragraph(recipe["title"], S["title"]))
        story.append(Spacer(1, 4))

    # ── byline ────────────────────────────────────────────────────────────────
    if recipe["byline"]:
        story.append(Paragraph(recipe["byline"], S["byline"]))
        story.append(Spacer(1, 2))

    # ── meta row (time / yield) ───────────────────────────────────────────────
    meta_parts = []
    if recipe["time_total"]:
        meta_parts.append(f"Total: {recipe['time_total']}")
    if recipe["time_prep"]:
        meta_parts.append(f"Prep: {recipe['time_prep']}")
    if recipe["time_cook"]:
        meta_parts.append(f"Cook: {recipe['time_cook']}")
    if recipe["yield"]:
        meta_parts.append(f"Yield: {recipe['yield']}")
    if meta_parts:
        story.append(Paragraph("  ·  ".join(meta_parts), S["meta"]))
        story.append(Spacer(1, 6))

    story.append(HRFlowable(width="100%", thickness=0.5, color=NYT_BORDER, spaceAfter=10))

    # ── hero image ────────────────────────────────────────────────────────────
    if recipe.get("image_url"):
        scraper_session = requests.Session()
        scraper_session.headers.update(RecipeScraper.HEADERS)
        img = fetch_image(recipe["image_url"], scraper_session, 5 * inch, 3.5 * inch)
        if img:
            img.hAlign = "CENTER"
            story.append(img)
            story.append(Spacer(1, 8))

    # ── description ───────────────────────────────────────────────────────────
    if recipe["description"]:
        story.append(Paragraph(recipe["description"], S["description"]))
        story.append(Spacer(1, 4))

    # ── ingredients ───────────────────────────────────────────────────────────
    if recipe["ingredient_groups"]:
        story.append(Paragraph("INGREDIENTS", S["section_header"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=NYT_BORDER, spaceAfter=4))
        for group in recipe["ingredient_groups"]:
            if group["name"]:
                story.append(Paragraph(group["name"], S["section_header"]))
            for item in group["items"]:
                # bullet point via leading em-dash
                story.append(Paragraph(f"— {item}", S["ingredient"]))
        story.append(Spacer(1, 8))

    # ── preparation steps ─────────────────────────────────────────────────────
    if recipe["steps"]:
        story.append(Paragraph("PREPARATION", S["section_header"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=NYT_BORDER, spaceAfter=6))
        for i, step in enumerate(recipe["steps"], 1):
            # Step number + text in a small two-column table for tight alignment
            label_para = Paragraph(f"<b>{i}.</b>", S["step_label"])
            text_para  = Paragraph(step, S["step_text"])
            t = Table(
                [[label_para, text_para]],
                colWidths=[0.3 * inch, None],
                hAlign="LEFT",
            )
            t.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 4),
                ("LEFTPADDING", (1, 0), (1, 0), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t)

    # ── notes / tips ──────────────────────────────────────────────────────────
    if recipe["notes"]:
        story.append(Spacer(1, 6))
        story.append(Paragraph("NOTES", S["section_header"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=NYT_BORDER, spaceAfter=4))
        for note in recipe["notes"]:
            story.append(Paragraph(note, S["note"]))

    # ── footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=NYT_BORDER, spaceAfter=4))
    story.append(Paragraph(recipe["url"], S["footer"]))

    doc.build(story)


# ── helpers ───────────────────────────────────────────────────────────────────
def safe_filename(title: str) -> str:
    """Convert a recipe title to a safe filename."""
    name = title.lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s]+", "_", name.strip())
    return name[:80] or "recipe"


def safe_folder_name(name: str) -> str:
    """Convert category text to a filesystem-safe folder name."""
    folder = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "", name).strip()
    folder = re.sub(r"\s+", "_", folder)
    return folder[:80] or "uncategorized"


def load_url_entries_from_file(path: str) -> list[tuple[str, Optional[str]]]:
    """Load URL entries and optional category headers from a text file.

    Supported category header formats:
      [Dinner]
      Category: Dinner
    """
    entries: list[tuple[str, Optional[str]]] = []
    current_category: Optional[str] = None

    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        bracket_header = re.fullmatch(r"\[(.+)\]", line)
        named_header = re.fullmatch(r"(?i)category\s*[:=-]\s*(.+)", line)
        if bracket_header:
            current_category = bracket_header.group(1).strip() or None
            continue
        if named_header:
            current_category = named_header.group(1).strip() or None
            continue

        entries.append((line, current_category))

    return entries


def category_from_position(index: int, category_size: int, prefix: str) -> str:
    """Return category name based on 1-based position in the input list."""
    category_num = ((index - 1) // category_size) + 1
    return f"{prefix}_{category_num:02d}"


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Batch-download NYT Cooking recipes and save as PDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python nyt_recipe_to_pdf.py --urls-file my_recipes.txt
              python nyt_recipe_to_pdf.py https://cooking.nytimes.com/recipes/1020044
              python nyt_recipe_to_pdf.py --urls-file box.txt --output ~/RecipePDFs
        """),
    )
    parser.add_argument(
        "urls",
        nargs="*",
        metavar="URL",
        help="One or more NYT Cooking recipe URLs.",
    )
    parser.add_argument(
        "--urls-file", "-f",
        metavar="FILE",
        help=(
            "Path to a text file with recipe URLs. Optional category headers are "
            "supported, e.g. [Dinner] or Category: Dinner."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        metavar="DIR",
        default=str(Path.home() / "recipes"),
        help="Output directory for PDFs (default: ~/recipes).",
    )
    parser.add_argument(
        "--nyt-cookie",
        metavar="COOKIE",
        default=os.environ.get("NYT_COOKIE", ""),
        help="Value of the NYT-S session cookie (or set NYT_COOKIE env var).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        metavar="SECONDS",
        help="Pause between requests to avoid rate-limiting (default: 1.5).",
    )
    parser.add_argument(
        "--category-size",
        type=int,
        default=0,
        metavar="N",
        help=(
            "If > 0, create output subfolders by URL position from --urls-file. "
            "Example: --category-size 20 creates category_01 for items 1-20, "
            "category_02 for 21-40, etc."
        ),
    )
    parser.add_argument(
        "--category-prefix",
        default="category",
        metavar="NAME",
        help="Prefix for position-based category folders (default: category).",
    )
    args = parser.parse_args()

    if args.category_size < 0:
        parser.error("--category-size must be 0 or greater")

    # ── collect URLs ──────────────────────────────────────────────────────────
    file_entries: list[tuple[str, Optional[str]]] = []
    if args.urls_file:
        file_entries = load_url_entries_from_file(args.urls_file)

    url_entries: list[tuple[str, Optional[str]]] = [(u, None) for u in args.urls] + file_entries
    if not url_entries:
        parser.error("Provide at least one URL or a --urls-file.")

    # ── output dir ────────────────────────────────────────────────────────────
    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    # ── build styles once ─────────────────────────────────────────────────────
    styles = build_styles()
    scraper = RecipeScraper(cookie=args.nyt_cookie or None)

    ok, failed = 0, []

    for i, (url, file_category) in enumerate(url_entries, 1):
        print(f"[{i}/{len(url_entries)}] {url}")
        try:
            recipe = scraper.fetch(url)
            title  = recipe.get("title") or f"recipe_{i}"
            fname  = safe_filename(title) + ".pdf"

            target_dir = out_dir
            if file_category:
                target_dir = out_dir / safe_folder_name(file_category)
                target_dir.mkdir(parents=True, exist_ok=True)
            elif args.category_size > 0 and i > len(args.urls):
                file_pos = i - len(args.urls)
                category_name = category_from_position(
                    file_pos,
                    args.category_size,
                    args.category_prefix,
                )
                target_dir = out_dir / category_name
                target_dir.mkdir(parents=True, exist_ok=True)

            out    = target_dir / fname
            build_pdf(recipe, out, styles)
            print(f"  ✓  Saved: {out}")
            ok += 1
        except requests.HTTPError as e:
            msg = f"HTTP {e.response.status_code}"
            if e.response.status_code == 401:
                msg += " – check your NYT_COOKIE / --nyt-cookie"
            elif e.response.status_code == 404:
                msg += " – recipe not found"
            print(f"  ✗  {msg}")
            failed.append((url, msg))
        except Exception as e:
            print(f"  ✗  {e}")
            failed.append((url, str(e)))

        if i < len(url_entries):
            time.sleep(args.delay)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Done. {ok} succeeded, {len(failed)} failed.")
    if failed:
        print("\nFailed URLs:")
        for url, reason in failed:
            print(f"  {url}  →  {reason}")


if __name__ == "__main__":
    main()
