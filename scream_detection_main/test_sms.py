from twilio.rest import Client

# Your credentials
account_sid = 'ACa85966f9f4709b65db970489a36b3e73'
auth_token = 'your_auth_token'  # Get from console
twilio_number = '+16209128157'

client = Client(account_sid, auth_token)

# Send test SMS to verified number
message = client.messages.create(
    body='✅ Verification successful! Scream detection is now active.',
    from_=twilio_number,
    to='+916363421738'  # This number must be verified
)

print(f"✅ SMS sent! SID: {message.sid}")