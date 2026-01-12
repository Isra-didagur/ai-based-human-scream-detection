from twilio.rest import Client
import requests
from datetime import datetime

class LocationService:
    def get_location(self):


      return {
        "city": "Bengaluru",
        "region": "Karnataka",
        "country": "India",
        "location_text": "Bengaluru, Karnataka, India",
        "map_link": "https://maps.app.goo.gl/nfg2P2vM7qi97Euj8"
    }

      try:
            r = requests.get("http://ipinfo.io/json", timeout=5)
            data = r.json()

            loc = data.get("loc", "")
            lat, lng = loc.split(",") if loc else (None, None)

            return {
                "city": data.get("city", "Unknown"),
                "region": data.get("region", "Unknown"),
                "country": data.get("country", "Unknown"),
                "lat": lat,
                "lng": lng,
            }
      except Exception as e:
            print("Location error:", e)
            return None

class SMSService:
    def __init__(self, sid, token, from_number):
        self.client = Client(sid, token)
        self.from_number = from_number
        self.loc_service = LocationService()

    def send_alert(self, to_number, base_message):
        loc = self.loc_service.get_location()

        if loc:
            map_link = f"https://maps.google.com/maps?q={loc['lat']},{loc['lng']}"
            location_text = (
                f"📍 Approx Location: {loc['city']}, {loc['region']}, {loc['country']}\n"
                f"🗺️ Map: {map_link}\n"
                f"(IP-based, may be inaccurate)"
            )
        else:
            location_text = "📍 Location unavailable"

        full_message = f"""{base_message}

{location_text}

⏰ Time: {datetime.now().strftime("%d %b %Y %I:%M:%S %p")}
"""

        msg = self.client.messages.create(
            body=full_message,
            from_=self.from_number,
            to=to_number
        )

        print(f"📨 Twilio accepted message SID: {msg.sid}")
