import pypdf
from docx import Document as DocxDocument
from langchain.tools import BaseTool, tool

@tool
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    Args:
        pdf_path: The path to the PDF file.
    Returns:
        A string of text that contains the text extracted from the PDF file.
    """
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

@tool
def extract_text_from_docx(docx_path: str) -> str:
    """
    Extract text from a DOCX file.
    Args:
        docx_path: The path to the DOCX file.
    Returns:
        A string of text that contains the text extracted from the DOCX file.
    """
    with open(docx_path, 'rb') as file:
        doc = DocxDocument(file)
        return "\n".join(p.text for p in doc.paragraphs)

@tool
def extract_text_from_txt(txt_path: str) -> str:
    """
    Extract text from a TXT file.
    Args:
        txt_path: The path to the TXT file.
    Returns:
        A string of text that contains the text extracted from the TXT file.
    """
    with open(txt_path, 'r') as file:
        return file.read()

TOOLS: list[BaseTool] = [extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt]
TOOLS_BY_NAME: dict[str, BaseTool] = {tool.name: tool for tool in TOOLS}