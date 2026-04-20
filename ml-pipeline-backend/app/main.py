from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import json
import uuid
import io

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from app.pipeline import start_pipeline, get_job, list_jobs

app = FastAPI(title="ANN Regression Pipeline API")

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

UPLOAD_DIR = "/tmp/pipeline_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/api/template")
async def download_template(
    n_labels: int = Query(default=7, ge=0, le=50, description="Number of label columns"),
    n_inputs: int = Query(default=10, ge=1, le=100, description="Number of input columns"),
    output_name: str = Query(default="Output", description="Name of the output/target column"),
):
    """Generate and download a standardized Excel template.

    Row 1: Category headers (Label / Input / Output) - color-coded
    Row 2: Column names (user fills in the actual variable names)
    Row 3+: Data rows (user fills in)
    """
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Data"

    total_cols = n_labels + n_inputs + 1  # +1 for output

    # Color definitions
    label_fill = PatternFill(start_color="D6EAF8", end_color="D6EAF8", fill_type="solid")  # light blue
    input_fill = PatternFill(start_color="D5F5E3", end_color="D5F5E3", fill_type="solid")  # light green
    output_fill = PatternFill(start_color="FADBD8", end_color="FADBD8", fill_type="solid")  # light red
    cat_font = Font(bold=True, size=11, color="FFFFFF")
    label_cat_fill = PatternFill(start_color="2E86C1", end_color="2E86C1", fill_type="solid")  # dark blue
    input_cat_fill = PatternFill(start_color="27AE60", end_color="27AE60", fill_type="solid")  # dark green
    output_cat_fill = PatternFill(start_color="E74C3C", end_color="E74C3C", fill_type="solid")  # dark red
    name_font = Font(bold=True, size=10)
    name_fill_label = PatternFill(start_color="EBF5FB", end_color="EBF5FB", fill_type="solid")
    name_fill_input = PatternFill(start_color="EAFAF1", end_color="EAFAF1", fill_type="solid")
    name_fill_output = PatternFill(start_color="FDEDEC", end_color="FDEDEC", fill_type="solid")
    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin"),
    )
    center = Alignment(horizontal="center", vertical="center")

    # --- Row 1: Category headers ---
    for col_idx in range(1, total_cols + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = cat_font
        cell.alignment = center
        cell.border = thin_border
        if col_idx <= n_labels:
            cell.value = "Label"
            cell.fill = label_cat_fill
        elif col_idx <= n_labels + n_inputs:
            cell.value = "Input"
            cell.fill = input_cat_fill
        else:
            cell.value = "Output"
            cell.fill = output_cat_fill

    # --- Row 2: Column names (placeholder for user to fill in) ---
    for col_idx in range(1, total_cols + 1):
        cell = ws.cell(row=2, column=col_idx)
        cell.font = name_font
        cell.alignment = center
        cell.border = thin_border
        if col_idx <= n_labels:
            cell.value = f"Label_{col_idx}"
            cell.fill = name_fill_label
        elif col_idx <= n_labels + n_inputs:
            cell.value = f"Input_{col_idx - n_labels}"
            cell.fill = name_fill_input
        else:
            cell.value = output_name.strip() or "Output"
            cell.fill = name_fill_output

    # --- Rows 3-5: Example data rows ---
    for row_idx in range(3, 6):
        for col_idx in range(1, total_cols + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.alignment = center
            cell.border = thin_border
            if col_idx <= n_labels:
                cell.value = f"sample_{row_idx - 2}"
            else:
                cell.value = 0.0

    # Set column widths
    for col_idx in range(1, total_cols + 1):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = 18

    # Freeze panes: freeze rows 1-2 so they stay visible when scrolling
    ws.freeze_panes = "A3"

    # Set row heights
    ws.row_dimensions[1].height = 25
    ws.row_dimensions[2].height = 22

    # --- Instructions sheet ---
    ws_info = wb.create_sheet("Instructions")
    instructions = [
        ["ANN Regression Pipeline - Data Template"],
        [""],
        ["Row Structure:"],
        ["  Row 1: Column category (Label / Input / Output) - DO NOT MODIFY"],
        ["  Row 2: Column names - Replace with your actual variable names"],
        ["  Row 3+: Data - Replace example rows with your actual data"],
        [""],
        ["Column Categories:"],
        [f"  Label (blue):  {n_labels} columns - Identifiers/metadata (not used as model inputs)"],
        [f"  Input (green): {n_inputs} columns - Features used by the ANN model"],
        [f"  Output (red):  1 column - Target variable to predict"],
        [""],
        ["Instructions:"],
        ["  1. Keep Row 1 (categories) unchanged - the pipeline uses it to identify column types"],
        ["  2. In Row 2, replace placeholder names with your actual variable names"],
        ["  3. Delete the example data rows (3-5) and paste your own data starting at Row 3"],
        ["  4. Input and Output columns must contain numeric values"],
        ["  5. Label columns can contain text or numbers"],
        ["  6. Do not leave empty rows in the middle of your data"],
        [""],
        ["Supported file formats for upload: .xlsx, .csv, .tsv"],
    ]
    for row_idx, row_data in enumerate(instructions, 1):
        for col_idx, val in enumerate(row_data, 1):
            cell = ws_info.cell(row=row_idx, column=col_idx, value=val)
            if row_idx == 1:
                cell.font = Font(bold=True, size=14)
            elif row_idx in [3, 8, 13]:
                cell.font = Font(bold=True, size=12)
    ws_info.column_dimensions["A"].width = 80

    tmp_path = os.path.join(UPLOAD_DIR, f"template_{uuid.uuid4().hex[:8]}.xlsx")
    wb.save(tmp_path)

    return FileResponse(
        tmp_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"pipeline_template_{n_labels}labels_{n_inputs}inputs.xlsx",
    )


@app.get("/api/template/prediction")
async def download_prediction_template(
    n_labels: int = Query(default=7, ge=0, le=50, description="Number of label columns"),
    n_inputs: int = Query(default=10, ge=1, le=100, description="Number of input columns"),
):
    """Generate and download a standardized Excel template for prediction samples (no output column).

    Row 1: Category headers (Label / Input) - color-coded
    Row 2: Column names (user fills in the actual variable names)
    Row 3+: Data rows (user fills in)
    """
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Data"

    total_cols = n_labels + n_inputs  # No output column

    # Color definitions
    cat_font = Font(bold=True, size=11, color="FFFFFF")
    label_cat_fill = PatternFill(start_color="2E86C1", end_color="2E86C1", fill_type="solid")
    input_cat_fill = PatternFill(start_color="27AE60", end_color="27AE60", fill_type="solid")
    name_font = Font(bold=True, size=10)
    name_fill_label = PatternFill(start_color="EBF5FB", end_color="EBF5FB", fill_type="solid")
    name_fill_input = PatternFill(start_color="EAFAF1", end_color="EAFAF1", fill_type="solid")
    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin"),
    )
    center = Alignment(horizontal="center", vertical="center")

    # --- Row 1: Category headers ---
    for col_idx in range(1, total_cols + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = cat_font
        cell.alignment = center
        cell.border = thin_border
        if col_idx <= n_labels:
            cell.value = "Label"
            cell.fill = label_cat_fill
        else:
            cell.value = "Input"
            cell.fill = input_cat_fill

    # --- Row 2: Column names (placeholder for user to fill in) ---
    for col_idx in range(1, total_cols + 1):
        cell = ws.cell(row=2, column=col_idx)
        cell.font = name_font
        cell.alignment = center
        cell.border = thin_border
        if col_idx <= n_labels:
            cell.value = f"Label_{col_idx}"
            cell.fill = name_fill_label
        else:
            cell.value = f"Input_{col_idx - n_labels}"
            cell.fill = name_fill_input

    # --- Rows 3-5: Example data rows ---
    for row_idx in range(3, 6):
        for col_idx in range(1, total_cols + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.alignment = center
            cell.border = thin_border
            if col_idx <= n_labels:
                cell.value = f"sample_{row_idx - 2}"
            else:
                cell.value = 0.0

    # Set column widths
    for col_idx in range(1, total_cols + 1):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = 18

    # Freeze panes
    ws.freeze_panes = "A3"
    ws.row_dimensions[1].height = 25
    ws.row_dimensions[2].height = 22

    # --- Instructions sheet ---
    ws_info = wb.create_sheet("Instructions")
    instructions = [
        ["ANN Regression Pipeline - Prediction Samples Template"],
        [""],
        ["Row Structure:"],
        ["  Row 1: Column category (Label / Input) - DO NOT MODIFY"],
        ["  Row 2: Column names - Replace with your actual variable names"],
        ["  Row 3+: Data - Replace example rows with your actual data"],
        [""],
        ["Column Categories:"],
        [f"  Label (blue):  {n_labels} columns - Identifiers/metadata (not used as model inputs)"],
        [f"  Input (green): {n_inputs} columns - Features (must match training data inputs)"],
        [""],
        ["Important:"],
        ["  - No Output column: this template is for new samples to predict"],
        ["  - Column names in Row 2 must match the training data column names exactly"],
        ["  - Input columns must contain numeric values"],
        ["  - Label columns can contain text or numbers"],
        ["  - Do not leave empty rows in the middle of your data"],
        [""],
        ["Supported file formats for upload: .xlsx, .csv, .tsv"],
    ]
    for row_idx, row_data in enumerate(instructions, 1):
        for col_idx, val in enumerate(row_data, 1):
            cell = ws_info.cell(row=row_idx, column=col_idx, value=val)
            if row_idx == 1:
                cell.font = Font(bold=True, size=14)
            elif row_idx in [3, 8, 12]:
                cell.font = Font(bold=True, size=12)
    ws_info.column_dimensions["A"].width = 80

    tmp_path = os.path.join(UPLOAD_DIR, f"prediction_template_{uuid.uuid4().hex[:8]}.xlsx")
    wb.save(tmp_path)

    return FileResponse(
        tmp_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"prediction_template_{n_labels}labels_{n_inputs}inputs.xlsx",
    )


@app.post("/api/upload-table")
async def upload_table_data(body: dict):
    """Accept table data as JSON and save as Excel file with template format.

    Expected body:
    {
        "file_type": "training" | "new_data",
        "categories": ["Label", "Label", "Input", "Input", "Output"],
        "columns": ["Name1", "Name2", "Feat1", "Feat2", "Target"],
        "rows": [["a", "b", 1.0, 2.0, 3.0], ...]
    }
    """
    from openpyxl import Workbook

    file_type = body.get("file_type", "training")
    categories = body.get("categories", [])
    columns = body.get("columns", [])
    rows = body.get("rows", [])

    if not columns:
        raise HTTPException(status_code=400, detail="No columns provided")
    if not rows:
        raise HTTPException(status_code=400, detail="No data rows provided")

    wb = Workbook()
    ws = wb.active
    ws.title = "Data"

    # Row 1: categories
    for col_idx, cat in enumerate(categories, 1):
        ws.cell(row=1, column=col_idx, value=cat)

    # Row 2: column names
    for col_idx, col_name in enumerate(columns, 1):
        ws.cell(row=2, column=col_idx, value=col_name)

    # Row 3+: data
    for row_idx, row_data in enumerate(rows, 3):
        for col_idx, val in enumerate(row_data, 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            # Try to convert numeric strings to numbers
            if isinstance(val, str):
                try:
                    val = float(val)
                    if val == int(val):
                        val = int(val)
                except (ValueError, TypeError):
                    pass
            cell.value = val

    file_id = str(uuid.uuid4())
    file_dir = os.path.join(UPLOAD_DIR, file_id)
    os.makedirs(file_dir, exist_ok=True)
    file_path = os.path.join(file_dir, f"{file_type}.xlsx")
    wb.save(file_path)

    return {
        "file_id": file_id,
        "file_type": file_type,
        "path": file_path,
        "n_rows": len(rows),
        "n_cols": len(columns),
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), file_type: str = Form("training")):
    """Upload a dataset file (training or new_data). Returns a file_id."""
    file_id = str(uuid.uuid4())
    file_dir = os.path.join(UPLOAD_DIR, file_id)
    os.makedirs(file_dir, exist_ok=True)

    ext = os.path.splitext(file.filename or "data.tsv")[1]
    file_path = os.path.join(file_dir, f"{file_type}{ext}")

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "file_type": file_type,
        "path": file_path,
        "size_bytes": len(content),
    }


@app.post("/api/train")
async def train(
    data_file_id: str = Form(...),
    new_data_file_id: str = Form(None),
    config: str = Form("{}"),
):
    """Start a training job. Returns job_id for polling."""
    file_dir = os.path.join(UPLOAD_DIR, data_file_id)
    if not os.path.isdir(file_dir):
        raise HTTPException(status_code=404, detail="Training data file not found. Upload first.")

    data_path = None
    for fname in os.listdir(file_dir):
        if fname.startswith("training"):
            data_path = os.path.join(file_dir, fname)
            break

    if data_path is None:
        raise HTTPException(status_code=404, detail="Training data file not found in upload directory.")

    new_data_path = None
    if new_data_file_id:
        nd_dir = os.path.join(UPLOAD_DIR, new_data_file_id)
        if os.path.isdir(nd_dir):
            for fname in os.listdir(nd_dir):
                if fname.startswith("new_data"):
                    new_data_path = os.path.join(nd_dir, fname)
                    break

    try:
        cfg = json.loads(config)
    except json.JSONDecodeError:
        cfg = {}

    job_id = start_pipeline(data_path, cfg, new_data_path)
    return {"job_id": job_id, "status": "started"}


@app.get("/api/jobs")
async def get_jobs():
    """List all jobs."""
    return list_jobs()


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job status, progress, metrics, and available plots."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "metrics": job["metrics"],
        "plots": list(job["plots"].keys()),
        "has_results_zip": job["results_zip"] is not None,
        "has_predictions_zip": job["predictions_zip"] is not None,
        "error": job["error"],
        "config": job.get("config", {}),
    }


@app.get("/api/job/{job_id}/logs")
async def get_job_logs(job_id: str, since: int = 0):
    """Get job logs starting from index 'since'."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    logs = job["logs"][since:]
    return {
        "logs": logs,
        "total": len(job["logs"]),
        "since": since,
    }


@app.get("/api/job/{job_id}/plot/{plot_name}")
async def get_plot(job_id: str, plot_name: str):
    """Get a specific plot as base64 PNG."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    plot_data = job["plots"].get(plot_name)
    if plot_data is None:
        raise HTTPException(status_code=404, detail=f"Plot '{plot_name}' not found")

    return {"plot_name": plot_name, "data": plot_data, "format": "png"}


@app.get("/api/job/{job_id}/download/results")
async def download_results(job_id: str):
    """Download training results ZIP."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["results_zip"] is None or not os.path.exists(job["results_zip"]):
        raise HTTPException(status_code=404, detail="Results ZIP not available yet")

    return FileResponse(
        job["results_zip"],
        media_type="application/zip",
        filename="training_results.zip",
    )


@app.get("/api/job/{job_id}/download/predictions")
async def download_predictions(job_id: str):
    """Download new data predictions ZIP."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["predictions_zip"] is None or not os.path.exists(job["predictions_zip"]):
        raise HTTPException(status_code=404, detail="Predictions ZIP not available yet")

    return FileResponse(
        job["predictions_zip"],
        media_type="application/zip",
        filename="new_data_predictions.zip",
    )
