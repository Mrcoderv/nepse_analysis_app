# utils/validators.py

def is_valid_symbol(symbol: str, company_list: list) -> bool:
    """
    Validates if a given symbol exists in the NEPSE company list.
    """
    if not symbol or not company_list:
        return False
    
    # We assume company_list is a list of dictionaries, where each has a 'symbol' key
    valid_symbols = [comp.get('symbol', '').strip().upper() for comp in company_list if isinstance(comp, dict)]
    return symbol.strip().upper() in valid_symbols
