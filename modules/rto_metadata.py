STATE_CODES = {
    'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh', 'AS': 'Assam', 'BR': 'Bihar',
    'CG': 'Chhattisgarh', 'GA': 'Goa', 'GJ': 'Gujarat', 'HR': 'Haryana',
    'HP': 'Himachal Pradesh', 'JH': 'Jharkhand', 'KA': 'Karnataka', 'KL': 'Kerala',
    'MP': 'Madhya Pradesh', 'MH': 'Maharashtra', 'MN': 'Manipur', 'ML': 'Meghalaya',
    'MZ': 'Mizoram', 'NL': 'Nagaland', 'OD': 'Odisha', 'PB': 'Punjab',
    'RJ': 'Rajasthan', 'SK': 'Sikkim', 'TN': 'Tamil Nadu', 'TG': 'Telangana',
    'TR': 'Tripura', 'UP': 'Uttar Pradesh', 'UK': 'Uttarakhand', 'WB': 'West Bengal',
    'AN': 'Andaman and Nicobar Islands', 'CH': 'Chandigarh', 'DN': 'Dadra and Nagar Haveli',
    'DD': 'Daman and Diu', 'DL': 'Delhi', 'JK': 'Jammu and Kashmir', 'LA': 'Ladakh',
    'LD': 'Lakshadweep', 'PY': 'Puducherry'
}

def parse_rto_metadata(plate_text):
    """
    Given an Indian License Plate string like `MH 12 AB 1234`,
    parse and return the state name and the RTO district code.
    """
    if not plate_text or len(plate_text.replace(" ", "")) < 4:
        return {"state": "Unknown", "rto_code": "Unknown"}
        
    plate_clean = plate_text.replace(" ", "").upper()
    state_abbr = plate_clean[0:2]
    rto_code = plate_clean[2:4]
    
    state_name = STATE_CODES.get(state_abbr, "Unknown RTO State")
    
    return {
        "state_abbr": state_abbr,
        "state": state_name,
        "district_code": rto_code
    }
