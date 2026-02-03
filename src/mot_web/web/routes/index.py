from __future__ import annotations
from flask import Blueprint, render_template

bp = Blueprint("index", __name__)

@bp.get("/")
def home():
    return render_template("index.html")
