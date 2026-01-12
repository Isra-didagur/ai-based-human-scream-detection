from twilio.rest import Client

# Your credentials
account_sid = "ACa85966f9f4709b65db970489a36b3e73"  
auth_token = "5d78d4c6c78a06a8b94cf9bf56a65567"    
from_number= "+16209128157"                        

# All your emergency numbers
emergency_numbers = ["+916363421738"]

try:
    client = Client(account_sid, auth_token)
    
    print("🔍 Testing each number individually:")
    print("=" * 50)
    
    success_count = 0
    total_numbers = len(emergency_numbers)
    
    for i, number in enumerate(emergency_numbers, 1):
        print(f"\n📱 Testing number {i}: {number}")
        try:
            message = client.messages.create(
                body=f"🧪 DEBUG TEST {i}/3: Testing number {number}",
                from_=from_number,
                to=number
            )
            print(f"✅ SUCCESS! Message sent with SID: {message.sid}")
            success_count += 1
        except Exception as e:
            print(f"❌ FAILED! Error: {e}")
            # Try to get more details about the error
            print(f"   Error type: {type(e).__name__}")
            if hasattr(e, 'code'):
                print(f"   Error code: {e.code}")
            if hasattr(e, 'msg'):
                print(f"   Error message: {e.msg}")
    
    print("\n" + "=" * 50)
    print(f"📊 FINAL RESULT: {success_count}/{total_numbers} messages sent successfully")
    
    if success_count < total_numbers:
        print(f"⚠️  {total_numbers - success_count} numbers failed - check the errors above")
    else:
        print("🎉 All numbers working perfectly!")
    
except Exception as e:
    print(f"❌ GENERAL ERROR: {e}")