from __future__ import annotations
from flask import Blueprint, render_template

bp = Blueprint("realtime", __name__, url_prefix="/realtime")

@bp.get("/")
def realtime_page():
    return render_template("realtime.html")
