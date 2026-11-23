"""Clinical image storage.

Images are part of the medical record, so they are handled carefully:

* Stored outside the web root (``uploads/`` or ``$HMS_UPLOADS``) and served only
  through an authenticated route that checks the clinic, never from /static.
* Every upload is decoded with Pillow and re-encoded. That rejects files that are
  not really images, drops EXIF metadata (GPS location, device, owner name),
  applies the camera rotation, and caps the size.
* File names are content hashes, so a name never leaks patient details and the
  same photo uploaded twice for one patient is stored once.
* Removing an image hides it (soft delete); the file stays for the record and can
  be purged by an administrator with ``purge_deleted``.

The AI pipeline (handoff/code) accepts up to 4 images per case. The doctor's
written finding is what the ranker and the explanation read; an automatic caption
is added only when the optional captioning model is installed (see caption()).
"""

from __future__ import annotations

import hashlib
import io
import os
import threading
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .db import ROOT

UPLOAD_DIR = Path(os.environ.get("HMS_UPLOADS", ROOT / "uploads"))
MAX_BYTES = int(float(os.environ.get("HMS_MAX_UPLOAD_MB", "15")) * 1024 * 1024)
MAX_SIDE = 3000          # longest side kept on disk
THUMB_SIDE = 360
MAX_PER_CASE = 4         # same limit as the handoff pipeline (MAX_IMAGES_PER_CASE)
MAX_PER_UPLOAD = 8
Image.MAX_IMAGE_PIXELS = 60_000_000   # decompression-bomb guard

ACCEPT = "image/jpeg,image/png,image/webp,image/gif,image/bmp,image/tiff"
FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF", "MPO"}

# (key, label). Labels are what the AI pipeline sees as the image "modality".
MODALITIES = [
    ("clinical_photo", "Clinical photograph"),
    ("skin", "Skin / rash photograph"),
    ("dermoscopy", "Dermoscopy"),
    ("fundus", "Fundus photograph"),
    ("eye", "Eye photograph"),
    ("oral", "Mouth / throat photograph"),
    ("xray", "X-ray"),
    ("ct", "CT scan"),
    ("mri", "MRI"),
    ("ultrasound", "Ultrasound"),
    ("echo", "Echocardiogram"),
    ("ecg", "ECG"),
    ("histology", "Histology / pathology slide"),
    ("lab_report", "Lab report"),
    ("other", "Other image"),
]
MODALITY_LABEL = dict(MODALITIES)
PATIENT_MODALITIES = ["clinical_photo", "skin", "eye", "oral", "lab_report", "other"]


class ImageError(ValueError):
    """Upload rejected; the message is safe to show to the user."""


def patient_dir(patient_id: int) -> Path:
    return UPLOAD_DIR / f"p{int(patient_id)}"


def process(data: bytes, filename: str = "") -> dict:
    """Validate and re-encode an upload. Returns the bytes to store plus metadata."""
    if not data:
        raise ImageError(f"{filename or 'The file'} is empty.")
    if len(data) > MAX_BYTES:
        raise ImageError(f"{filename or 'The file'} is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()
        if fmt not in FORMATS:
            raise ImageError(f"{filename or 'The file'}: {fmt} images are not supported. Use JPEG, PNG or WebP.")
        im = Image.open(io.BytesIO(data))
        im.seek(0)            # first frame of GIF/TIFF/MPO
        im = ImageOps.exif_transpose(im)
        im.load()
    except Image.DecompressionBombError:
        raise ImageError(f"{filename or 'The file'} has too many pixels.")
    except ImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ImageError(f"{filename or 'The file'} is not a readable image.")

    # photos stay JPEG; scans, screenshots and anything with transparency become PNG
    lossless = fmt in {"PNG", "GIF", "BMP", "TIFF"} or im.mode in {"RGBA", "LA", "P", "I;16", "I", "1"}
    if im.mode in {"I;16", "I", "F"}:
        im = ImageOps.autocontrast(im.convert("I").point(lambda v: v / 256).convert("L"))
    if lossless:
        im = im.convert("RGBA" if "A" in im.getbands() or im.mode == "P" else ("L" if im.mode in {"L", "1"} else "RGB"))
    else:
        im = im.convert("L" if im.mode == "L" else "RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

    out = io.BytesIO()
    if lossless:
        im.save(out, "PNG", optimize=True)
        ext, mime = "png", "image/png"
    else:
        im.save(out, "JPEG", quality=90, optimize=True)   # no exif= argument: metadata dropped
        ext, mime = "jpg", "image/jpeg"
    stored = out.getvalue()

    th = im.convert("RGB") if im.mode != "RGB" else im.copy()
    th.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.LANCZOS)
    tb = io.BytesIO()
    th.save(tb, "JPEG", quality=80)

    return {"bytes": stored, "thumb": tb.getvalue(), "ext": ext, "mime": mime,
            "width": im.size[0], "height": im.size[1], "size": len(stored),
            "sha256": hashlib.sha256(stored).hexdigest(), "original_format": fmt}


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def store(patient_id: int, processed: dict) -> tuple[str, str]:
    """Write the image and its thumbnail; returns (stored_name, thumb_name)."""
    name = f"{processed['sha256'][:40]}.{processed['ext']}"
    thumb = f"{processed['sha256'][:40]}_t.jpg"
    d = patient_dir(patient_id)
    if not (d / name).exists():
        _write(d / name, processed["bytes"])
    if not (d / thumb).exists():
        _write(d / thumb, processed["thumb"])
    return name, thumb


def file_path(img, thumb: bool = False) -> Path | None:
    name = img.thumb_name if thumb else img.stored_name
    if not name or "/" in name or "\\" in name or ".." in name:
        return None
    p = patient_dir(img.patient_id) / name
    return p if p.exists() else None


def purge_deleted(db) -> int:
    """Delete files of images removed from the record (admin maintenance)."""
    from .db import PatientImage
    n = 0
    for img in db.query(PatientImage).filter(PatientImage.deleted_at.isnot(None)).all():
        still_used = db.query(PatientImage).filter(
            PatientImage.patient_id == img.patient_id, PatientImage.sha256 == img.sha256,
            PatientImage.deleted_at.is_(None)).count()
        if not still_used:
            for name in (img.stored_name, img.thumb_name):
                p = patient_dir(img.patient_id) / name
                if p.exists():
                    p.unlink()
                    n += 1
    return n


def save_uploads(db, patient_id: int, uploads: list[tuple[str, bytes]], *, source: str = "staff",
                 uploaded_by_id: int | None = None, modality: str = "clinical_photo", body_site: str = "",
                 finding: str = "", taken_on=None, intake_id: int | None = None,
                 appointment_id: int | None = None) -> tuple[list, list[str]]:
    """Validate, store and record uploads. Returns (images, error messages).

    A photo already in this patient's record (same content) is not stored twice;
    the existing record is returned instead (and restored if it had been removed).
    """
    from datetime import datetime

    from .db import PatientImage
    modality = modality if modality in MODALITY_LABEL else "other"
    saved, errors = [], []
    for filename, data in uploads[:MAX_PER_UPLOAD]:
        try:
            pr = process(data, filename)
        except ImageError as exc:
            errors.append(str(exc))
            continue
        img = db.query(PatientImage).filter(PatientImage.patient_id == patient_id,
                                            PatientImage.sha256 == pr["sha256"]).first()
        if img:
            if img.deleted_at:
                img.deleted_at, img.deleted_by_id = None, None
            if finding and not img.finding:
                img.finding, img.finding_source = finding, ("patient" if source == "patient" else "clinician")
            if intake_id and not img.intake_id:
                img.intake_id = intake_id
            saved.append(img)
            continue
        stored, thumb = store(patient_id, pr)
        img = PatientImage(
            patient_id=patient_id, source=source, uploaded_by_id=uploaded_by_id, intake_id=intake_id,
            appointment_id=appointment_id, modality=modality, body_site=body_site[:120],
            finding=finding.strip(), finding_source="patient" if source == "patient" else "clinician",
            taken_on=taken_on, original_name=(filename or "")[-255:], stored_name=stored, thumb_name=thumb,
            mime=pr["mime"], size_bytes=pr["size"], width=pr["width"], height=pr["height"],
            sha256=pr["sha256"], created_at=datetime.now())
        cap = caption(patient_dir(patient_id) / stored)
        if cap:
            img.caption, img.caption_model = cap, CAPTION_MODEL
        db.add(img)
        saved.append(img)
    if len(uploads) > MAX_PER_UPLOAD:
        errors.append(f"Only the first {MAX_PER_UPLOAD} files were uploaded.")
    db.commit()
    return saved, errors


def as_json(img) -> dict:
    return {"id": img.id, "url": f"/images/{img.id}", "thumb": f"/images/{img.id}?thumb=1",
            "modality": img.modality, "modality_label": MODALITY_LABEL.get(img.modality, "Image"),
            "label": label(img), "body_site": img.body_site, "finding": img.finding,
            "finding_source": img.finding_source, "source": img.source,
            "created": img.created_at.strftime("%d %b %Y") if img.created_at else "",
            "width": img.width, "height": img.height}


# ----------------------------------------------------------------------------
# optional automatic caption (same model family as the handoff pipeline)
# ----------------------------------------------------------------------------
CAPTION_MODEL = os.environ.get("HMS_CAPTION_MODEL", "WafaaFraih/blip-roco-radiology-captioning")
_cap = {"model": None, "processor": None, "failed": False}
_cap_lock = threading.Lock()


def captions_enabled() -> bool:
    if os.environ.get("HMS_IMAGE_CAPTIONS", "0") != "1" or _cap["failed"]:
        return False
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def caption(path: Path) -> str | None:
    """Machine caption, only when HMS_IMAGE_CAPTIONS=1 and torch + transformers are
    installed. The pipeline labels it "machine-generated, may be wrong"."""
    if not captions_enabled():
        return None
    try:
        with _cap_lock:
            if _cap["model"] is None:
                import torch
                from transformers import BlipForConditionalGeneration, BlipProcessor
                _cap["processor"] = BlipProcessor.from_pretrained(CAPTION_MODEL)
                _cap["model"] = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL).eval()
                _cap["torch"] = torch
            with Image.open(path) as im:
                inp = _cap["processor"](im.convert("RGB"), return_tensors="pt")
            with _cap["torch"].no_grad():
                out = _cap["model"].generate(**inp, max_new_tokens=36, num_beams=1)
            return _cap["processor"].decode(out[0], skip_special_tokens=True).strip() or None
    except Exception:
        _cap["failed"] = True
        return None


# ----------------------------------------------------------------------------
# AI pipeline glue
# ----------------------------------------------------------------------------
def label(img) -> str:
    base = MODALITY_LABEL.get(img.modality, "Image")
    return f"{base} ({img.body_site})" if img.body_site else base


def finding_sentence(img) -> str:
    """How an image enters the case text the ranker and realist read."""
    if not (img.finding or "").strip():
        return ""
    f = img.finding.strip().rstrip(".")
    who = " (patient's description)" if img.finding_source == "patient" else ""
    return f"{label(img)}{who}: {f}."


def pipeline_meta(images) -> list[dict]:
    """Image metadata in the format handoff/code expects (fmt_case_for_plan)."""
    out = []
    for i, img in enumerate(list(images)[:MAX_PER_CASE], 1):
        sub = label(img)
        if (img.finding or "").strip():
            who = "patient-reported" if img.finding_source == "patient" else "clinician-reported"
            sub += f"; {who} finding: {img.finding.strip()}"
        out.append({"image_id": f"I{i}", "modality": MODALITY_LABEL.get(img.modality, "image"),
                    "subtype": sub, "file": f"image-{img.id}", "caption": img.caption or None,
                    "caption_model": img.caption_model or None, "db_id": img.id})
    return out


def image_findings(meta: list[dict], start: int) -> list[dict]:
    """Imaging findings, appended after the text findings (C-ids continue), as in
    handoff/code/live_engine.py."""
    out = []
    for im in meta:
        txt = im["subtype"]
        if im.get("caption"):
            txt += f"; automatic caption: {im['caption']}"
        has_finding = "-reported finding" in im["subtype"]
        out.append({"id": f"C{start + len(out) + 1}", "text": txt, "type": "imaging",
                    "quote": im.get("caption") or "", "quote_verified": has_finding or bool(im.get("caption")),
                    "source": "image", "image_id": im["image_id"], "db_id": im["db_id"]})
    return out
