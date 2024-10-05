
import re

def convert_currency_to_number(currency_str):
    # Remove ₹ symbol and commas
    currency_str = currency_str.replace('₹', '').replace(',', '').strip()
    
    # Check for 'Lakh' and 'Crore' and convert accordingly
    if 'Lakh' in currency_str:
        return float(re.sub(r'[^\d.]', '', currency_str)) * 10**5  # 1 Lakh = 100,000
    elif 'Crore' in currency_str:
        return float(re.sub(r'[^\d.]', '', currency_str)) * 10**7  # 1 Crore = 10,000,000
    else:
        return float(re.sub(r'[^\d.]', '', currency_str))  # Direct conversion for numbers without Lakh/Crore
    
def convert_km_to_number(km):

    formatted_km = int(km.replace(',',''))

    return formatted_km

def convert_register_year_string_to_number(year):
    try:
        formated_year= year.split()[-1]

        return formated_year
    except:
        return 0

def convert_number_to_currency(num):
    # Convert to Crore if number is greater than or equal to 1 Crore (10^7)
    if num >= 10**7:
        return f"₹ {num / 10**7:.2f} Crore"
    
    # Convert to Lakh if number is greater than or equal to 1 Lakh (10^5)
    elif num >= 10**5:
        return f"₹ {num / 10**5:.2f} Lakh"
    
    # Direct return for numbers less than 1 Lakh
    else:
        return f"₹ {num:,.0f}"  # Adding commas for better readability