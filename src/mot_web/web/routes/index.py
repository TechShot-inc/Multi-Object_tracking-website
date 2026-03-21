from __future__ import annotations
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return request.app.state.templates.TemplateResponse("index.html", {"request": request})
