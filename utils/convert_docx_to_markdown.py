import mammoth
from pathlib import Path
import re


def convert_html_table_to_markdown(html_content):
    """Convert HTML tables to Markdown format with proper alignment and line breaks."""
    # Find all tables
    table_pattern = r'<table[^>]*>(.*?)</table>'
    tables = re.finditer(table_pattern, html_content, re.DOTALL | re.IGNORECASE)

    result = html_content

    for table_match in tables:
        table_html = table_match.group(0)
        table_content = table_match.group(1)

        # Extract rows
        row_pattern = r'<tr[^>]*>(.*?)</tr>'
        rows = re.findall(row_pattern, table_content, re.DOTALL | re.IGNORECASE)

        if not rows:
            continue

        # Process all rows to get cell contents and calculate column widths
        all_cells = []
        max_widths = []

        for row in rows:
            # Extract cells (th or td)
            cell_pattern = r'<t[hd][^>]*>(.*?)</t[hd]>'
            cells = re.findall(cell_pattern, row, re.DOTALL | re.IGNORECASE)

            # Clean cell content
            cleaned_cells = []
            for cell in cells:
                # Preserve line breaks within cells
                cell = cell.replace('<br>', '<br/>')
                cell = cell.replace('<br/>', '\n')
                cell = cell.replace('<br />', '\n')

                # cell = re.sub(r'\n', ' ', cell, flags=re.IGNORECASE)

                # ✅ Convert <s> tags to Markdown strikethrough
                cell = re.sub(r'<s>', '~~', cell, flags=re.IGNORECASE)
                cell = re.sub(r' </s>', '~~', cell, flags=re.IGNORECASE)  # bug fix
                cell = re.sub(r'</s>', '~~', cell, flags=re.IGNORECASE)

                # Remove other HTML tags
                cell = re.sub(r'<[^>]+>', ' ', cell)

                # Decode HTML entities
                # cell = cell.replace('&nbsp;', ' ')
                # cell = cell.replace('&lt;', '<')
                # cell = cell.replace('&gt;', '>')
                # cell = cell.replace('&amp;', '&')

                # Clean up whitespace but preserve line breaks
                lines = cell.split('\n')
                lines = [' '.join(line.split()) for line in lines]
                cell = '<br>'.join(line for line in lines if line)

                cell = re.sub(r'<br>', ' ', cell, flags=re.IGNORECASE) # bug fix

                cleaned_cells.append(cell)

            all_cells.append(cleaned_cells)

            # Update max widths
            for j, cell in enumerate(cleaned_cells):
                # Calculate display width (accounting for <br> tags)
                display_text = cell.replace('<br>', ' ')
                cell_width = len(display_text)

                if j >= len(max_widths):
                    max_widths.append(cell_width)
                else:
                    max_widths[j] = max(max_widths[j], cell_width)

        # Generate markdown table with proper spacing
        markdown_rows = []

        for i, cells in enumerate(all_cells):
            # Pad cells to max width
            padded_cells = []
            for j, cell in enumerate(cells):
                display_text = cell.replace('<br>', ' ')
                padding = max_widths[j] - len(display_text)
                padded_cell = cell + ' ' * padding
                padded_cells.append(padded_cell)

            # Create markdown row
            markdown_row = '| ' + ' | '.join(padded_cells) + ' |'
            markdown_rows.append(markdown_row)

            # Add separator after header row
            if i == 0:
                separators = ['-' * max_widths[j] for j in range(len(cells))]
                separator = '| ' + ' | '.join(separators) + ' |'
                markdown_rows.append(separator)

        # Replace HTML table with markdown table
        markdown_table = '\n'.join(markdown_rows)
        result = result.replace(table_html, '\n\n' + markdown_table + '\n\n')

    return result


def process_paragraphs(html_content):
    """Add proper line breaks between paragraphs."""
    # Convert paragraph tags to  newlines
    html_content = re.sub(r'</p>\s*<p>', '\n', html_content, flags=re.IGNORECASE)
    html_content = re.sub(r'<p[^>]*>', '', html_content, flags=re.IGNORECASE)
    html_content = re.sub(r'</p>', '\n', html_content, flags=re.IGNORECASE)

    # Convert line breaks to newlines
    html_content = re.sub(r'<br\s*/?>', '\n', html_content, flags=re.IGNORECASE)

    # Convert div tags to line breaks
    html_content = re.sub(r'</div>\s*<div>', '\n\n', html_content, flags=re.IGNORECASE)
    html_content = re.sub(r'<div[^>]*>', '', html_content, flags=re.IGNORECASE)
    html_content = re.sub(r'</div>', '\n\n', html_content, flags=re.IGNORECASE)

    return html_content


def clean_markdown(text):
    """Clean up the markdown text."""
    # Remove remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Decode common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&amp;', '&')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")

    # Clean up excessive whitespace on each line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)

    # Fix multiple blank lines (more than 2)
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    return text.strip()


def docx_to_markdown(docx_path, output_path):
    """
    Convert a DOCX file to Markdown format.

    Args:
        docx_path: Path to the input DOCX file
        output_path: Path for the output MD file

    Returns:
        The markdown content as a string
    """
    docx_path = Path(docx_path)
    output_path = Path(output_path)

    if not docx_path.exists():
        raise FileNotFoundError(f"File not found: {docx_path}")

    # Convert DOCX to HTML using mammoth
    with open(docx_path, 'rb') as docx_file:
        result = mammoth.convert_to_html(docx_file)
        html_content = result.value

    # Process paragraphs to add line breaks
    markdown_content = process_paragraphs(html_content)

    # Convert HTML tables to Markdown
    markdown_content = convert_html_table_to_markdown(markdown_content)

    # Clean up the markdown
    markdown_content = clean_markdown(markdown_content)

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as md_file:
        md_file.write(markdown_content)

    print(f"✅ Converted: {docx_path.name} → {output_path.name}")
    print(f"✅ Output saved to: {output_path}")

    return markdown_content

if __name__ == '__main__':
    # Specify your input and output file paths here
    input_file = r""  # example: C:\Users\xxxx\WordDocument.docx
    output_file = r""  # example: C:\Users\xxxx\WordDocument_converted.md

    try:
        docx_to_markdown(input_file, output_file)
    except Exception as e:
        print(f"❌ convert_docx_markdown.py Error: {e}")