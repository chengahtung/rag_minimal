import os
import platform
import subprocess
import logging

def open_file(file_path: str):
    """Open any file with the default application."""
    if not os.path.exists(file_path):
        logging.info(f"❌ File not found: {file_path}")
        return

    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(file_path)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", file_path])
        else:  # Linux and others
            subprocess.run(["xdg-open", file_path])
        logging.info(f"✅ Opened file: {file_path}")
    except Exception as e:
        logging.info(f"⚠️ Could not open file: {e}")

def open_folder(folder_path: str):
    """Open a folder in the file explorer."""
    if not os.path.isdir(folder_path):
        logging.info(f"❌ Folder not found: {folder_path}")
        return

    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(folder_path)
        elif system == "Darwin":
            subprocess.run(["open", folder_path])
        else:
            subprocess.run(["xdg-open", folder_path])
        logging.info(f"✅ Opened folder: {folder_path}")
    except Exception as e:
        logging.info(f"⚠️ Could not open folder: {e}")


# ────────────────────────────────
# ✏️ User Input Section
# ────────────────────────────────

# file_to_open = r""        # C:\Users\xxxx\document.txt
# folder_to_open = r""      # C:\Users\xxxx\

# Uncomment the one you want to use:
# open_file(file_to_open)
# open_folder(folder_to_open)
