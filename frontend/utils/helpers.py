# utils/helpers.py

def format_currency(value):
    """Formats a number as currency (NPR)."""
    try:
        return f"Rs {float(value):,.2f}"
    except (ValueError, TypeError):
        return "-"

def format_percentage(value):
    """Formats a number as a percentage."""
    try:
        return f"{float(value):+.2f}%"
    except (ValueError, TypeError):
        return "-"

def format_volume(value):
    """Formats volume numbers with commas."""
    try:
        return f"{int(float(value)):,}"
    except (ValueError, TypeError):
        return "-"
