#!/usr/bin/env python3
"""Build the standalone PDF summary without LaTeX."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "report" / "boltzmann3d_numerical_report.pdf"


def sci(value: str | float) -> str:
    return f"{float(value):.4e}"


def page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(0.65 * inch, 0.42 * inch, "3D Boltzmann FNO/C-FNO reference")
    canvas.drawRightString(7.85 * inch, 0.42 * inch, f"Page {document.page}")
    canvas.restoreState()


def styled_table(data, widths, header=True, font_size=8.5):
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2F8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17324D")),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#A8B3BF")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    table.setStyle(TableStyle(commands))
    return table


def main() -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=colors.HexColor("#17324D"),
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["Normal"],
            fontSize=11,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#52606D"),
            spaceAfter=24,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionBlue",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#17324D"),
            spaceBefore=10,
            spaceAfter=8,
        )
    )
    body = ParagraphStyle(
        "BodyCompact", parent=styles["BodyText"], fontSize=9.5, leading=14, spaceAfter=7
    )
    note = ParagraphStyle(
        "Note", parent=body, fontSize=8.5, leading=12, textColor=colors.HexColor("#52606D")
    )

    metrics = {}
    with (PROJECT / "results" / "bkw_metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metrics[row["method"]] = row
    timing = json.loads((PROJECT / "results" / "computing_time_summary.json").read_text())

    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.62 * inch,
        title="3D Boltzmann equation: FNO and C-FNO numerical reference",
        author="Numerical experiment package",
    )
    story = []
    story.append(Paragraph("3D Boltzmann equation with FNO and C-FNO", styles["ReportTitle"]))
    story.append(
        Paragraph(
            "A reproducible numerical reference for cutoff Maxwell molecules, the positive BKW benchmark, and online computing cost",
            styles["Subtitle"],
        )
    )
    story.append(Paragraph("Experiment at a glance", styles["SectionBlue"]))
    setup = [
        ["Quantity", "Reported configuration"],
        ["Equation", "Spatially homogeneous Boltzmann equation in 3 velocity dimensions"],
        ["Collision kernel", "Cutoff Maxwell molecules, B = 1/(4 pi)"],
        ["Velocity discretization", "32 x 32 x 32 cell-centered grid"],
        ["Reference solver", "Fast spectral method; Nrho = Nsph = Nsphpre = 32"],
        ["Time map", "One physical time unit; RK4 with dt = 0.1"],
        ["Neural models", "FNO and C-FNO; 8 Fourier modes per velocity axis"],
        ["Training set", "50 endpoint pairs with family counts 16 / 17 / 17"],
    ]
    story.append(styled_table(setup, [1.55 * inch, 5.55 * inch], font_size=9))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Hybrid BKW data design", styles["SectionBlue"]))
    story.append(
        Paragraph(
            "The 17 BKW training members contain nine exact positive BKW states at distinct start times and eight bounded, mass-neutral perturbations near f<sub>BKW</sub>(5.5). The exact evaluation pair f<sub>BKW</sub>(5.5) to f<sub>BKW</sub>(6.5) is excluded from training and model selection. All targets are produced by the same one-unit fast-spectral RK4 solver.",
            body,
        )
    )
    story.append(
        Paragraph(
            "The paper-time convention is K(t) = 1 - exp(-t/6). Positivity in three dimensions requires t >= 6 log(5/2), so t = 5.5 is the first practical positive benchmark time used here.",
            body,
        )
    )
    story.append(Paragraph("Training refinement", styles["SectionBlue"]))
    story.append(
        Paragraph(
            "Both models predict the standardized endpoint increment and use a small random final-layer initialization with BKW-balanced batches. C-FNO adds nondimensionalized mass, momentum, and energy penalties during training; its inference graph is identical to FNO.",
            body,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Positive BKW benchmark: t = 5.5 to t = 6.5", styles["SectionBlue"]))
    profile = PROJECT / "figures" / "bkw_profiles_t1_manuscript.png"
    story.append(Image(str(profile), width=7.1 * inch, height=2.84 * inch))
    story.append(Spacer(1, 8))
    error_rows = [["Method", "L1", "L2", "Linf", "Relative L2"]]
    for method in ("C-FNO", "FNO", "SM"):
        row = metrics[method]
        error_rows.append(
            [method, sci(row["L1"]), sci(row["L2"]), sci(row["Linf"]), sci(row["relative_L2"])]
        )
    story.append(styled_table(error_rows, [0.9 * inch] + [1.55 * inch] * 4))
    story.append(Spacer(1, 10))
    moment_rows = [["Method", "Density error", "Bulk-velocity error", "Energy error"]]
    for method in ("C-FNO", "FNO", "SM"):
        row = metrics[method]
        moment_rows.append(
            [method, sci(row["density_error"]), sci(row["bulk_velocity_error"]), sci(row["energy_error"])]
        )
    story.append(styled_table(moment_rows, [1.0 * inch, 2.0 * inch, 2.1 * inch, 2.0 * inch]))
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "SM has relative L2 error 4.5981e-3 against the analytic BKW state. FNO and C-FNO attain 4.9332e-3 and 4.8018e-3, respectively. C-FNO also reduces all three macroscopic errors relative to FNO. Because the neural targets come from SM, neural error against the analytic solution contains both reference discretization error and operator-learning error.",
            body,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Computing-time comparison", styles["SectionBlue"]))
    online = timing["online_cpu_seconds"]
    speed = timing["derived"]
    online_rows = [
        ["Method", "Online task", "Mean +/- std (s)", "Speedup"],
        ["SM", "One RK4 step, dt = 0.1", f"{online['sm_one_rk4_step']['mean']:.4f} +/- {online['sm_one_rk4_step']['standard_deviation']:.4f}", "-"],
        ["SM", "Complete ten-step map", f"{online['sm_complete_map']['mean']:.4f} +/- {online['sm_complete_map']['standard_deviation']:.4f}", "1.0x"],
        ["FNO", "One endpoint inference", f"{online['fno_complete_map']['mean']:.5f} +/- {online['fno_complete_map']['standard_deviation']:.5f}", f"{speed['cpu_speedup_sm_over_fno']:.1f}x"],
        ["C-FNO", "One endpoint inference", f"{online['cfno_complete_map']['mean']:.5f} +/- {online['cfno_complete_map']['standard_deviation']:.5f}", f"{speed['cpu_speedup_sm_over_cfno']:.1f}x"],
    ]
    story.append(styled_table(online_rows, [0.8 * inch, 2.7 * inch, 2.15 * inch, 1.1 * inch], font_size=8.5))
    story.append(Spacer(1, 12))
    training = timing["offline_training_seconds"]
    training_rows = [
        ["Method", "Optimizer steps", "Training time (s)", "Training time (min)"],
        ["FNO", "3000", f"{training['fno']:.1f}", f"{training['fno']/60:.2f}"],
        ["C-FNO", "3000", f"{training['cfno']:.1f}", f"{training['cfno']/60:.2f}"],
    ]
    story.append(styled_table(training_rows, [1.0 * inch, 1.7 * inch, 1.9 * inch, 2.0 * inch], font_size=9))
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            "The fair online task is the full endpoint map: the spectral solver must take ten RK4 steps, whereas each trained neural operator evaluates the map once. The primary timings use the same Apple M4 CPU. Spectral weights are resident and cache loading is excluded; neural timings start from an assembled normalized tensor and exclude file I/O and preprocessing.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Training is reported separately as an offline cost. With the dataset already available, the measured training time is amortized after about 24 endpoint queries; including the recorded data-generation time raises the estimate to about 93 queries.",
            body,
        )
    )
    story.append(Paragraph("Reproducibility notes", styles["SectionBlue"]))
    story.append(
        Paragraph(
            "The package includes frozen data, selected checkpoints, configuration, source code, tests, raw CSV/JSON metrics, and SHA-256 data validation. Results use one deterministic optimization seed and should not be interpreted as a multi-seed uncertainty study. Wall-clock timings are hardware- and software-dependent.",
            note,
        )
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
