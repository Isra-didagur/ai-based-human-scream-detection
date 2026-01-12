import os
import json

# ============================================
# MAIN CONFIGURATION VARIABLES
# These are what main.py imports directly
# ============================================

# Your Twilio Credentials
TWILIO_ACCOUNT_SID = "ACa85966f9f4709b65db970489a36b3e73"  
TWILIO_AUTH_TOKEN = "5d78d4c6c78a06a8b94cf9bf56a65567"    
TWILIO_PHONE_NUMBER = "+16209128157"                        

# Your Verified Emergency Contacts (with +91 India country code)
EMERGENCY_CONTACTS = ["+916363421738", "+917022293408"]  

# Optional Settings
INCLUDE_LOCATION = True
CUSTOM_EMERGENCY_MESSAGE = None

# ============================================
# ADVANCED CONFIGURATION CLASS
# For future enhancements and settings management
# ============================================

class Config:
    """Configuration management for the scream detection app"""
    
    def __init__(self):
        self.config_file = "app_config.json"
        self.load_config()
    
    def load_config(self):
        """Load configuration from file or create default"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            else:
                self.create_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            self.create_default_config()
    
    def create_default_config(self):
        """Create default configuration with YOUR credentials"""
        self.config = {
            "twilio": {
                "account_sid": "ACa85966f9f4709b65db970489a36b3e73",
                "auth_token": "5d78d4c6c78a06a8b94cf9bf56a65567",
                "from_number": "+16209128157"
            },
            "emergency_contacts": [
                {
                    "name": "Emergency Contact 1",
                    "number": "+916363421738",
                    "relationship": "Primary",
                    "verified": True
                },
                {
                    "name": "Emergency Contact 2", 
                    "number": "+917022293408",
                    "relationship": "Secondary",
                    "verified": True
                },
                # Add more contacts after verifying them in Twilio
                # {
                #     "name": "Emergency Contact 3", 
                #     "number": "+918310694903",
                #     "relationship": "Backup",
                #     "verified": False  # Set to True after verifying in Twilio
                # }
            ],
            "app_settings": {
                "recording_duration": 10,
                "sample_rate": 44100,
                "detection_sensitivity": "medium",
                "auto_send_sms": True,
                "include_location": True,
                "show_debug_logs": False
            }
        }
        self.save_config()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def get_twilio_config(self):
        """Get Twilio configuration"""
        return self.config.get("twilio", {})
    
    def set_twilio_config(self, account_sid, auth_token, from_number):
        """Set Twilio configuration"""
        self.config["twilio"] = {
            "account_sid": account_sid,
            "auth_token": auth_token,
            "from_number": from_number
        }
        self.save_config()
        
        # Update global variables too
        global TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER
        TWILIO_ACCOUNT_SID = account_sid
        TWILIO_AUTH_TOKEN = auth_token
        TWILIO_PHONE_NUMBER = from_number
    
    def get_emergency_contacts(self):
        """Get list of emergency contact phone numbers (only verified ones)"""
        contacts = self.config.get("emergency_contacts", [])
        # Return only verified contacts in trial mode
        return [contact["number"] for contact in contacts 
                if contact.get("number") and contact.get("verified", False)]
    
    def add_emergency_contact(self, name, number, relationship="", verified=False):
        """Add emergency contact"""
        contact = {
            "name": name,
            "number": number,
            "relationship": relationship,
            "verified": verified
        }
        if "emergency_contacts" not in self.config:
            self.config["emergency_contacts"] = []
        self.config["emergency_contacts"].append(contact)
        self.save_config()
        
        # Update global EMERGENCY_CONTACTS list if verified
        if verified:
            global EMERGENCY_CONTACTS
            if number not in EMERGENCY_CONTACTS:
                EMERGENCY_CONTACTS.append(number)
    
    def get_app_settings(self):
        """Get app settings"""
        return self.config.get("app_settings", {})
    
    def update_app_setting(self, setting_name, value):
        """Update a specific app setting"""
        if "app_settings" not in self.config:
            self.config["app_settings"] = {}
        self.config["app_settings"][setting_name] = value
        self.save_config()

# ============================================
# ENVIRONMENT VARIABLE FALLBACK
# For deployment or Docker environments
# ============================================

def get_config_from_env():
    """Get configuration from environment variables (fallback)"""
    return {
        "account_sid": os.getenv("TWILIO_ACCOUNT_SID", TWILIO_ACCOUNT_SID),
        "auth_token": os.getenv("TWILIO_AUTH_TOKEN", TWILIO_AUTH_TOKEN),
        "from_number": os.getenv("TWILIO_FROM_NUMBER", TWILIO_PHONE_NUMBER),
        "emergency_contacts": os.getenv("EMERGENCY_CONTACTS", 
                                       ",".join(EMERGENCY_CONTACTS)).split(",")
    }

# ============================================
# USAGE INSTRUCTIONS
# ============================================
"""
BASIC USAGE (in main.py):
    from config import (
        TWILIO_ACCOUNT_SID,
        TWILIO_AUTH_TOKEN,
        TWILIO_PHONE_NUMBER,
        EMERGENCY_CONTACTS,
        INCLUDE_LOCATION,
        CUSTOM_EMERGENCY_MESSAGE
    )

ADVANCED USAGE (for settings UI):
    from config import Config
    
    config = Config()
    config.set_twilio_config("new_sid", "new_token", "new_number")
    config.add_emergency_contact("John Doe", "+911234567890", "Friend", verified=True)
"""