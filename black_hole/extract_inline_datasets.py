import re
from pathlib import Path


def main() -> int:
    paper_path = Path(__file__).with_name("paper.tex")
    text = paper_path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n")

    # Matches *inline* table blocks like:
    # \\pgfplotstableread[col sep=space]{
    # time ...
    # 0.00 ...
    # }\\datatableName
    #
    # Important: we require a newline immediately after the opening '{' so we
    # don't accidentally match the loader macro's file-reading form
    #   \\pgfplotstableread[col sep=space]{#1}#2
    pattern = re.compile(
        r"\\pgfplotstableread\[col sep=space\]\{\s*\n(.*?)\n\}\s*\\(datatable[A-Za-z0-9_]+)",
        re.DOTALL,
    )

    matches = list(pattern.finditer(text))
    if not matches:
        raise SystemExit("No inline pgfplotstable datasets found.")

    out_dir = paper_path.parent
    written = []
    for m in matches:
        data = m.group(1).strip("\n")
        name = m.group(2)
        out_path = out_dir / f"{name}.dat"
        out_path.write_text(data.rstrip() + "\n", encoding="utf-8")
        written.append(out_path.name)

    print(f"Wrote {len(written)} dataset files:")
    for fn in written:
        print(f"- {fn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
