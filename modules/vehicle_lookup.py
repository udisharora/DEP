import re
import requests
import xml.etree.ElementTree as ET

def fetch_vehicle_data(plate: str, username: str = "udishxarora") -> dict:
    """
    Cleans a license plate string, validates it, and fetches vehicle data
    from the RegCheck SOAP API.
    """
    # Preprocessing: Remove spaces and convert to uppercase
    clean_plate = plate.replace(" ", "").upper()
    
    # Validate using regex
    pattern = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$")
    if not pattern.match(clean_plate):
        return {
            "plate": clean_plate,
            "valid": False,
            "data": None,
            "error": "Invalid Indian License Plate Format."
        }

    # SOAP API Call setup
    url = "http://www.regcheck.org.uk/api/reg.asmx"
    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "SOAPAction": '"http://regcheck.org.uk/CheckIndia"'
    }
    
    # SOAP Body formatting
    soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xmlns:xsd="http://www.w3.org/2001/XMLSchema"
               xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <CheckIndia xmlns="http://regcheck.org.uk">
      <RegistrationNumber>{clean_plate}</RegistrationNumber>
      <username>{username}</username>
    </CheckIndia>
  </soap:Body>
</soap:Envelope>"""

    try:
        response = requests.post(url, data=soap_body, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse Response
        root = ET.fromstring(response.content)
        
        import json
        
        def get_text(tag_name):
            # ElementTree does not support the local-name() XPath function.
            # We iterate through elements and match the tag name ignoring namespaces.
            for elem in root.iter():
                if elem.tag == tag_name or elem.tag.endswith(f"}}{tag_name}"):
                    return elem.text
            return None

        # The CheckIndia API actually returns a 'vehicleJson' string element
        vehicle_json_str = get_text("vehicleJson")
        
        if vehicle_json_str:
            try:
                v_data = json.loads(vehicle_json_str)
                
                def extract_val(field):
                    val = v_data.get(field)
                    if isinstance(val, dict):
                        return val.get("CurrentTextValue", "")
                    return val if val else ""
                
                make = extract_val("CarMake")
                model = extract_val("CarModel")
                fuel = extract_val("FuelType")
                engine = extract_val("EngineSize")
                year = extract_val("RegistrationYear")
                owner = extract_val("Owner")
                registration_date = extract_val("RegistrationDate")
                insurance = extract_val("Insurance")
                location = extract_val("Location")
                
                return {
                    "plate": clean_plate,
                    "valid": True,
                    "data": {
                        "make": make,
                        "model": model,
                        "fuel": fuel,
                        "engine": engine,
                        "year": year,
                        "owner": owner,
                        "registration_date": registration_date,
                        "insurance": insurance,
                        "location": location
                    }
                }
            except json.JSONDecodeError as e:
                return {
                    "plate": clean_plate,
                    "valid": False,
                    "error": f"Failed to parse inner JSON: {str(e)}",
                    "data": None
                }
                
        # Fallback error mapping if vehicleJson wasn't found
        error_msg = get_text("Description") or get_text("faultstring")
        return {
            "plate": clean_plate,
            "valid": False,
            "error": error_msg or "No vehicle data returned.",
            "data": None
        }

    except requests.exceptions.RequestException as e:
        return {
            "plate": clean_plate,
            "valid": False,
            "error": f"API request failed: {str(e)}",
            "data": None
        }
    except ET.ParseError as e:
        return {
            "plate": clean_plate,
            "valid": False,
            "error": f"XML parsing failed: {str(e)}",
            "data": None
        }
    except Exception as e:
        return {
            "plate": clean_plate,
            "valid": False,
            "error": f"Unexpected error: {str(e)}",
            "data": None
        }
