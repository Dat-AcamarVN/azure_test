import requests
import json
from pathlib import Path
from typing import Dict, Tuple
import logging
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)


def read_scanned_pdf_ocr_space(
        file_path: str,
        api_key: str,
        output_dir: str = None,
        language: str = "eng",
        detect_checkboxes: bool = True,
        dpi: int = 450
) -> Tuple[Dict[str, any], bool]:
    """
    Reads a scanned PDF using OCR.space API and PyMuPDF to extract text from first page only.

    Args:
        file_path (str): Path to the scanned PDF file.
        api_key (str): OCR.space API key.
        output_dir (str, optional): Directory to save output files.
        language (str): Language code for OCR (default: 'eng').
        detect_checkboxes (bool): Enable checkbox detection.
        dpi (int): Resolution for converting PDF to images (default: 600).

    Returns:
        Tuple[Dict[str, any], bool]: Structured result (text and checkboxes) and success status.
    """
    # Check if PDF file exists
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        _log.error(f"File {file_path} does not exist.")
        return {}, False

    try:
        # Open PDF with PyMuPDF
        _log.info(f"Opening PDF {file_path} with PyMuPDF...")
        pdf_document = fitz.open(pdf_path)

        # Check if PDF has at least one page
        if pdf_document.page_count == 0:
            _log.error("PDF has no pages.")
            pdf_document.close()
            return {}, False

        result = {
            "text_content": [],
            "checkboxes": []
        }
        success = False

        # Process only the first page (index 0)
        page_num = 0
        _log.info(f"Processing first page only (page {page_num + 1} of {pdf_document.page_count})")

        # Convert page to image
        page = pdf_document[page_num]
        zoom = dpi / 72  # Convert DPI to zoom factor (PDF default is 72 DPI)
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        temp_image_path = f"temp_page_{page_num}.png"
        pix.save(temp_image_path)

        # Prepare API request
        url = "https://apipro1.ocr.space/parse/image"
        payload = {
            "apikey": api_key,
            "language": language,
            "isOverlayRequired": True,  # Get bounding box for text
            "isCreateSearchablePdf": False,
            "issearchablepdfhidetextlayer": False,
            "detectCheckbox": str(detect_checkboxes).lower()  # Use correct parameter
        }
        files = {"file": open(temp_image_path, "rb")}

        try:
            # Send request to OCR.space API
            _log.info(f"Processing page {page_num + 1} with OCR.space API...")
            response = requests.post(url, files=files, data=payload)
            files["file"].close()  # Close file after sending

            # Check response
            if response.status_code != 200:
                _log.error(f"API request failed for page {page_num + 1}: {response.text}")
            else:
                response_data = response.json()
                if response_data.get("IsErroredOnProcessing", True):
                    _log.error(f"Error in OCR processing for page {page_num + 1}: {response_data.get('ErrorMessage')}")
                else:
                    # Page processed successfully
                    success = True

                    # Extract text
                    parsed_results = response_data.get("ParsedResults", [])
                    for parsed_result in parsed_results:
                        text = parsed_result.get("ParsedText", "")
                        result["text_content"].append({
                            "page": page_num + 1,  # Display as 1-indexed
                            "text": text
                        })

                        # Extract checkbox data
                        checkbox_results = parsed_result.get("CheckboxResult", {}).get("Marks", [])
                        for mark in checkbox_results:
                            result["checkboxes"].append({
                                "page": page_num + 1,  # Display as 1-indexed
                                "state": "checked" if mark.get("IsChecked", False) else "unchecked",
                                "coordinates": mark.get("BoundingBox", {}),
                                "nearby_text": mark.get("NearbyText", "")
                            })

        except Exception as e:
            _log.error(f"Error processing page {page_num + 1}: {str(e)}")

        finally:
            # Clean up temporary image
            if Path(temp_image_path).exists():
                Path(temp_image_path).unlink()

        # Close PDF document
        pdf_document.close()

        # Save results to output_dir if provided
        if output_dir and result["text_content"]:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            file_name = pdf_path.stem

            # Save text content
            text_file = output_path / f"{file_name}_output_first_page.txt"
            with text_file.open("w", encoding="utf-8") as fp:
                for page_data in result["text_content"]:
                    fp.write(f"Page {page_data['page']}:\n{page_data['text']}\n\n")
            _log.info(f"Text content saved to: {text_file}")

        return result, success

    except Exception as e:
        _log.error(f"Error processing file {file_path}: {str(e)}")
        return {}, False


def post_process_text(text_content: list) -> Dict[str, any]:
    """
    Post-processes extracted text to extract specific fields and checkbox states.

    Args:
        text_content (list): List of page text data extracted from OCR.

    Returns:
        Dict[str, any]: Structured data extracted from the text.
    """
    import re
    processed_data = {}

    # Combine all page text for processing (now just first page)
    full_text = "\n".join(page["text"] for page in text_content)

    # Extract application number
    app_number_match = re.search(r"Application No\.\s*(\d+/\d+)", full_text, re.IGNORECASE)
    if app_number_match:
        processed_data["application_number"] = app_number_match.group(1)

    # Extract checkbox states (pattern for K or 0)
    checkbox_pattern = r"(\d+[a-z]?)\)[K0]\s*(.+)"
    checkbox_matches = re.findall(checkbox_pattern, full_text, re.MULTILINE)
    checkboxes = {}
    for match in checkbox_matches:
        checkbox_id = match[0]
        state = "checked" if match[1] == "K" else "unchecked"
        description = match[2].strip()
        checkboxes[checkbox_id] = {"state": state, "description": description}
    processed_data["checkboxes"] = checkboxes

    return processed_data


if __name__ == "__main__":
    # Replace with your OCR.space API key
    API_KEY = "DPD89Y3Y8C9BX"
    PDF_FILE = "uploads/95346092_1200_4479_8c6d_71e72faa4a3e.PDF"
    OUTPUT_DIR = "output"

    # Process the scanned PDF (first page only)
    _log.info("🚀 Starting OCR.space processing (first page only)...")
    result, success = read_scanned_pdf_ocr_space(
        PDF_FILE, API_KEY, OUTPUT_DIR, dpi=450
    )

    if success:
        _log.info("✅ OCR.space processing completed successfully!")
        print("Extracted text content:")
        for page_data in result["text_content"]:
            print(f"Page {page_data['page']}:\n{page_data['text']}\n")

        print("Detected checkboxes:")
        print(json.dumps(result["checkboxes"], indent=2))

        # Post-process the text
        processed_data = post_process_text(result["text_content"])
        print("Processed data:")
        print(json.dumps(processed_data, indent=2))
    else:
        _log.error("❌ OCR.space processing failed!")
        print("Could not extract content from the PDF.")