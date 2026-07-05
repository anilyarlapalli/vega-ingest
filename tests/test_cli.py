"""CLI console entry point — `vega info` and `vega ingest`."""

import json
import re

from vega.cli import main


def _duration_line(path):
    return rf"processed {re.escape(str(path))} in \d+\.\d{{3}}s"


def test_info_reports_backend_and_languages(capsys):
    rc = main(["info"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "auto-selected OCR backend:" in out
    assert "CUDA GPU available:" in out
    # Every contract language is listed.
    for iso in ("en", "te", "hi", "ta", "ml", "or"):
        assert f"  {iso}  " in out


def test_ingest_stdout_emits_jsonl_records(born_digital_pdf, capsys):
    rc = main(["ingest", str(born_digital_pdf), "--ocr", "none"])
    assert rc == 0
    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.strip()]
    assert lines
    rec = json.loads(lines[0])
    assert set(rec) == {"chunk_id", "text", "metadata"}
    assert rec["metadata"]["doc_type"] == "pdf"
    err_lines = [l for l in captured.err.splitlines() if l.strip()]
    assert len(err_lines) == 1
    assert re.fullmatch(_duration_line(born_digital_pdf), err_lines[0])


def test_ingest_corrupt_pdf_does_not_emit_duration(tmp_path, capsys):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.5\nnot really a pdf\n")

    rc = main(["ingest", str(bad), "--ocr", "none"])
    assert rc == 0
    captured = capsys.readouterr()
    assert not re.search(_duration_line(bad), captured.err)


def test_ingest_successful_text_file_emits_duration(tmp_path, capsys):
    note = tmp_path / "note.txt"
    note.write_text(
        "=== Overview ===\n"
        "This plain text note parses successfully and emits ordinary chunks.\n",
        encoding="utf-8",
    )

    rc = main(["ingest", str(note), "--ocr", "none"])
    assert rc == 0
    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.strip()]
    assert lines
    assert json.loads(lines[0])["metadata"]["doc_type"] == "txt"
    err_lines = [l for l in captured.err.splitlines() if l.strip()]
    assert len(err_lines) == 1
    assert re.fullmatch(_duration_line(note), err_lines[0])


def test_ingest_directory_emits_duration(born_digital_pdf, capsys):
    rc = main(["ingest", str(born_digital_pdf.parent), "--ocr", "none"])
    assert rc == 0
    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.strip()]
    assert lines
    err_lines = [l for l in captured.err.splitlines() if l.strip()]
    assert len(err_lines) == 1
    assert re.fullmatch(_duration_line(born_digital_pdf.parent), err_lines[0])


def test_ingest_writes_jsonl_file(born_digital_pdf, tmp_path):
    out = tmp_path / "out.jsonl"
    rc = main(["ingest", str(born_digital_pdf), "--ocr", "none",
               "--out", str(out)])
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines
    assert all("chunk_id" in json.loads(l) for l in lines)


def test_ingest_json_document_dump(born_digital_pdf, tmp_path):
    doc = tmp_path / "doc.json"
    rc = main(["ingest", str(born_digital_pdf), "--ocr", "none",
               "--out", str(tmp_path / "c.jsonl"), "--json", str(doc)])
    assert rc == 0
    payload = json.loads(doc.read_text(encoding="utf-8"))
    assert payload["doc_type"] == "pdf"
    assert "elements" in payload


def test_missing_path_returns_error_code():
    assert main(["ingest", "/no/such/file.pdf", "--ocr", "none"]) == 2
