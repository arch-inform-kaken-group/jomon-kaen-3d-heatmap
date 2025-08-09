from pathlib import Path
import sys
import os

import io
from reportlab.graphics.shapes import Rect, Drawing  #
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from PIL import Image as PILImage  #

import numpy as np  #
import pandas as pd  #

import matplotlib  #

matplotlib.use('Agg')
import matplotlib.pyplot as plt

### PDF Report | AI Generated Visualization Functions ###

# ====================================================================
# ADDITIONS FOR JAPANESE CHARACTER SUPPORT START HERE
# ====================================================================
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 1. Register a Japanese font.
#    Replace 'ipaexg.ttf' with the path to your chosen font file.
#    You need to have this font file accessible to your script.
#    This font will be used for all text to ensure it can render Japanese.
try:
    font_path = Path(__file__).parent / 'ipaexg.ttf' # Assumes font is in the same directory
    pdfmetrics.registerFont(TTFont('JapaneseFont', font_path))
    # We can also register a bold version if a separate file exists.
    # For simplicity, we'll just use the same font file for both regular and bold.
    pdfmetrics.registerFont(TTFont('JapaneseFont-Bold', font_path))
    JAPANESE_FONT = 'JapaneseFont'
    JAPANESE_FONT_BOLD = 'JapaneseFont-Bold'
    print(f"Successfully registered Japanese font: {JAPANESE_FONT}")
except Exception as e:
    print(f"Failed to register Japanese font. Japanese characters may not render correctly: {e}", file=sys.stderr)
    # Fallback to a default font that may or may not support Japanese.
    # 'Helvetica' is the default and will not, but it's a good fallback for Latin characters.
    JAPANESE_FONT = 'Helvetica'
    JAPANESE_FONT_BOLD = 'Helvetica-Bold'
# ====================================================================
# ADDITIONS FOR JAPANESE CHARACTER SUPPORT END HERE
# ====================================================================


def _create_cmap_image(cmap, width=1.5 * inch, height=0.2 * inch):
    """Generates an image of a matplotlib colormap in an in-memory buffer."""
    fig, ax = plt.subplots(figsize=(width / inch, height / inch))
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf


def _generate_analysis_plots(
    data: np.ndarray,
    tracking_sheet_path: str = None,
):
    """
    Uses pandas and matplotlib to generate quantitative analysis plots.
    Returns a list of in-memory image buffers.
    """
    if data.size == 0:
        return []

    try:
        df = pd.DataFrame(list(data))
    except Exception:
        return []

    plots = []

    # ====================================================================
    # ADDITION: Configure Matplotlib to use the Japanese font for plots
    # This is crucial for plot titles, labels, and ticks to display
    # Japanese characters if they exist in your data.
    # ====================================================================
    plt.rcParams['font.family'] = JAPANESE_FONT

    # --- Plot 1: Count of Pottery ---
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(11, 4))
        id_counts = df['ID'].value_counts().sort_index()
        id_counts.plot(kind='bar', ax=ax, color='steelblue', zorder=2)
        ax.set_title('Count of Data Instances per Pottery ID', fontsize=14)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_xlabel('Pottery ID', fontsize=10)
        plt.xticks(rotation=-90, fontsize=8)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plots.append({'title': 'Instance Count Distribution', 'buffer': buf})
        plt.close(fig)
    except Exception:
        pass

    # --- Plot 2: Average Data Sizes ---
    try:
        fig, ax = plt.subplots(figsize=(11, 4))
        avg_sizes = df.groupby('ID')[['POINTCLOUD_SIZE_KB',
                                      'QA_SIZE_KB']].mean().round(2)
        avg_sizes.plot(kind='bar', ax=ax, zorder=2)
        ax.set_title('Average Data Size per Pottery ID', fontsize=14)
        ax.set_ylabel('Average Size (KB)', fontsize=10)
        ax.set_xlabel('Pottery ID', fontsize=10)
        ax.legend(['PointCloud', 'Q&A'])
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plots.append({
            'title': 'Average Data Size Distribution',
            'buffer': buf
        })
        plt.close(fig)
    except Exception:
        pass

    # --- Plot 3: Average Voice Quality ---
    if tracking_sheet_path and Path(tracking_sheet_path).exists():
        try:
            tracking_df = pd.read_csv(tracking_sheet_path)
            df['SESSION_KEY'] = df['GROUP'] + '_' + df[
                'SESSION_ID'] + '_' + df['ID']
            tracking_df[
                'SESSION_KEY'] = tracking_df['GROUP'] + '_' + tracking_df[
                    'SESSION_ID'] + '_' + tracking_df['ID']
            merged_df = pd.merge(
                df,
                tracking_df[['SESSION_KEY', 'VOICE_QUALITY_0_TO_5']],
                on='SESSION_KEY',
                how='left')

            if not merged_df['VOICE_QUALITY_0_TO_5'].dropna().empty:
                fig, ax = plt.subplots(figsize=(11, 4))
                voice_quality = merged_df.groupby(
                    'ID')['VOICE_QUALITY_0_TO_5'].mean().round(2)
                voice_quality.plot(kind='bar', ax=ax, color='teal', zorder=2)
                ax.set_title('Average Voice Quality per Pottery ID',
                             fontsize=14)
                ax.set_ylabel('Average Quality (0-5)', fontsize=10)
                ax.set_xlabel('Pottery ID', fontsize=10)
                plt.xticks(rotation=90, fontsize=8)
                plt.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150)
                buf.seek(0)
                plots.append({
                    'title': 'Average Voice Quality Distribution',
                    'buffer': buf
                })
                plt.close(fig)
        except Exception as e:
            print(f"Could not generate 'Voice Quality' plot: {e}",
                  file=sys.stderr)

    return plots


def generate_filtered_dataset_report(
    errors: dict,
    mode: int,
    hololens_2_spatial_error: float,
    base_color: list,
    cmap,
    groups: list = [],
    session_ids: list = [],
    pottery_ids: list = [],
    min_pointcloud_size: float = 0.0,
    min_qa_size: float = 0.0,
    min_voice_quality: float = 0.0,
    min_emotion_count: int = 1,
    max_emotion_count: int = 5,
    emotion_type: list = [],
    limit: int = 9,
    from_tracking_sheet: bool = False,
    tracking_sheet_path: str = "",
    n_filtered_from_tracking_sheet: int = 0,
    n_filtered_from_arguments: int = 0,
    n_valid_data: int = 0,
    filtered_data: list[dict] = [],
    output_dir: str = ".",
):
    """
    Generates a PDF report summarizing the data filtering process,
    including quantitative analysis plots.
    """
    report_path = os.path.join(output_dir,
                               "filtering_and_processing_report.pdf")
    doc = SimpleDocTemplate(report_path)
    story = []
    styles = getSampleStyleSheet()

    # --- Define Styles ---
    title_style = ParagraphStyle('Title',
                                 parent=styles['h1'],
                                 fontSize=18,
                                 alignment=TA_CENTER,
                                 spaceAfter=18,
                                 fontName=JAPANESE_FONT_BOLD) # CHANGE HERE
    heading_style = ParagraphStyle('Heading2',
                                   parent=styles['h2'],
                                   fontSize=14,
                                   spaceAfter=12,
                                   fontName=JAPANESE_FONT) # CHANGE HERE
    body_style = ParagraphStyle('Body',
                                parent=styles['Normal'],
                                fontName=JAPANESE_FONT) # CHANGE HERE

    # --- 1. Title ---
    story.append(Paragraph("Data Filtering and Processing Report",
                           title_style))
    story.append(Spacer(1, 0.2 * inch))

    # --- 2. Summary & Parameters (Combined for brevity) ---
    # --- Summary Section ---
    story.append(Paragraph("1. Summary", heading_style))
    final_count = n_valid_data - n_filtered_from_tracking_sheet - n_filtered_from_arguments
    summary_data = [
        ['Initial Datasets Found:',
         str(n_valid_data)],
        ['Filtered by Tracking Sheet:',
         str(n_filtered_from_tracking_sheet)],
        ['Filtered by Arguments:',
         str(n_filtered_from_arguments)],
        ['Final Datasets for Processing:',
         str(final_count)],
    ]
    summary_table = Table(summary_data, colWidths=[2.5 * inch, 2.5 * inch])
    summary_table.setStyle(
        TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, -1), JAPANESE_FONT),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 3), (0, 3), JAPANESE_FONT_BOLD),
            ('FONTNAME', (1, 3), (1, 3), JAPANESE_FONT_BOLD),
        ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.25 * inch))

    # --- Filtering Parameters Section ---
    story.append(Paragraph("2. Filtering Parameters", heading_style))

    def format_list(items):
        # This helper function will now be rendered with the Japanese-compatible font
        return ", ".join(map(str, items)) if items else "Not specified"

    # Create a small colored box for the base_color
    base_color_box = Drawing(20, 10)
    base_color_box.add(
        Rect(0,
             0,
             40,
             10,
             fillColor=colors.Color(*base_color),
             strokeColor=None))
    base_color_display = Table(
        [[Paragraph(str(base_color), body_style), base_color_box]],
        colWidths=[1.7 * inch, None])
    base_color_display.setStyle(
        TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))

    # Create an image of the colormap
    cmap_image_buffer = _create_cmap_image(cmap)
    cmap_display = Table([[
        Paragraph(f"'{cmap.name}'", body_style),
        Image(cmap_image_buffer, width=1.5 * inch, height=0.2 * inch)
    ]],
                          colWidths=[0.7 * inch, 1.6 * inch])
    cmap_display.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))

    params_data = [
        ['Mode:', 'Linient' if mode == 1 else 'Strict'],
        ['HoloLens 2 Spatial Error:', f"{hololens_2_spatial_error}°"],
        ['Base Color:', base_color_display],  # <-- Visual element
        ['Colormap:', cmap_display],  # <-- Visual element
        ['Filter by Tracking Sheet:', 'Yes' if from_tracking_sheet else 'No'],
        [
            'Tracking Sheet Path:',
            Paragraph(tracking_sheet_path, body_style)
            if from_tracking_sheet else 'N/A'
        ],
        [
            'Min Voice Quality (0-5):',
            str(min_voice_quality) if from_tracking_sheet else 'N/A'
        ],
        ['Min Point Cloud Size (KB):',
         str(min_pointcloud_size)],
        ['Min Q&A Size (KB):', str(min_qa_size)],
        ['Min Emotion Count:', str(min_emotion_count)],
        ['Max Emotion Count:', str(max_emotion_count)],
        ['Emotion Type:',
         Paragraph(format_list(emotion_type), body_style)],
        ['Max Pottery Count:', str(limit)],
        ['Groups:', Paragraph(format_list(groups), body_style)],
        ['Session IDs:',
         Paragraph(format_list(session_ids), body_style)],
        ['Pottery IDs:',
         Paragraph(format_list(pottery_ids), body_style)],
    ]

    params_table = Table(params_data, colWidths=[2.0 * inch, 3.0 * inch])
    params_table.setStyle(
        TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, -1), JAPANESE_FONT),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
    story.append(params_table)
    story.append(Spacer(1, 0.25 * inch))
    story.append(PageBreak())

    # --- 3. Quantitative Analysis Plots ---
    story.append(Paragraph("2. Quantitative Analysis", heading_style))

    analysis_plots = _generate_analysis_plots(filtered_data,
                                              tracking_sheet_path)

    if not analysis_plots:
        story.append(
            Paragraph("No analysis plots could be generated.", body_style))
        story.append(PageBreak())
    else:
        plot_title_style = ParagraphStyle('PlotTitle',
                                          parent=styles['h3'],
                                          spaceBefore=12,
                                          spaceAfter=4,
                                          fontName=JAPANESE_FONT) # CHANGE HERE
        for plot_info in analysis_plots:
            story.append(Paragraph(plot_info['title'], plot_title_style))

            # 1. Get the original image buffer
            original_buffer = plot_info['buffer']

            # 2. Open the image with Pillow
            pil_image = PILImage.open(original_buffer)

            # 3. Rotate it 90 degrees. `expand=True` ensures the canvas resizes.
            rotated_image = pil_image.rotate(90,
                                             expand=True,
                                             resample=PILImage.BICUBIC)

            # 4. Save the new, rotated image into a new buffer
            rotated_buffer = io.BytesIO()
            rotated_image.save(rotated_buffer, format='PNG')
            rotated_buffer.seek(0)

            # 5. Add the rotated image to the PDF
            img = Image(rotated_buffer, width=5 * inch, height=8 * inch)
            img.hAlign = 'CENTER'
            story.append(img)
            story.append(Spacer(1, 0.2 * inch))
            story.append(PageBreak())

    # --- 4. Error Report ---
    story.append(Paragraph("3. Error Report", heading_style))

    if not errors:
        story.append(
            Paragraph("No errors were encountered during the process.",
                      body_style))
    else:
        # Define a new style for the error titles
        error_title_style = ParagraphStyle(
            'ErrorTitle',
            parent=body_style,
            fontName=JAPANESE_FONT_BOLD, # CHANGE HERE
            fontSize=11,
            spaceAfter=6
        )

        for error_type, details in errors.items():
            count = details['count']
            paths = sorted(list(details['paths']))

            # 1. Create the bold title with the count
            title_text = f"{error_type} (Occurrences: {count})"
            error_title = Paragraph(title_text, error_title_style)
            story.append(error_title)

            # 2. Create a simple, single-column table for the paths
            path_data = [[Paragraph(p, body_style)] for p in paths]

            path_table = Table(path_data, colWidths=[5.5 * inch])

            path_table.setStyle(
                TableStyle([
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('FONTNAME', (0, 0), (-1, -1), JAPANESE_FONT), # CHANGE HERE
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))

            story.append(path_table)
            story.append(PageBreak())

    # --- Build PDF ---
    try:
        doc.build(story)
        print(f"\nSuccessfully generated report at: {report_path}")
    except Exception as e:
        print(f"\nFailed to generate PDF report: {e}", file=sys.stderr)