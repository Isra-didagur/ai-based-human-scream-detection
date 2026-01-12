import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KIVY_METAL_DISABLED"] = "1"

def get_safe_path():
    """Get a safe default path that exists and is accessible"""
    paths_to_try = [
        os.getcwd(),
        os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop"),
        os.path.join(os.path.expanduser("~"), "Desktop"),
        os.path.join(os.path.expanduser("~"), "OneDrive", "Documents"),
        os.path.join(os.path.expanduser("~"), "Documents"),
        os.path.join(os.path.expanduser("~"), "Music"),
        os.path.expanduser("~"),
    ]
    
    for path in paths_to_try:
        try:
            if os.path.exists(path) and os.path.isdir(path):
                os.listdir(path)
                print(f"Using safe path: {path}")
                return path
        except (PermissionError, OSError):
            continue
    
    fallback = os.getcwd()
    print(f"Fallback to current directory: {fallback}")
    return fallback

import logging
logging.getLogger('kivy').setLevel(logging.CRITICAL)

# Suppress annoying file access errors
import warnings
warnings.filterwarnings('ignore')

# CRITICAL FIX: Try sounddevice, catch DLL blocking error
AUDIO_BACKEND = None
try:
    import sounddevice as sd
    AUDIO_BACKEND = 'sounddevice'
    print("✅ Using sounddevice for audio")
except Exception as e:
    print(f"⚠️ sounddevice not available: {e}")
    try:
        import pyaudio
        import wave
        AUDIO_BACKEND = 'pyaudio'
        print("✅ Using pyaudio for audio (fallback)")
    except Exception as e2:
        print(f"❌ No audio backend available: {e2}")
        AUDIO_BACKEND = None

from scipy.io.wavfile import write
import numpy as np
from kivymd.toast import toast
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ListProperty, ObjectProperty
from kivy.uix.popup import Popup
from kivy.core.text import LabelBase
from kivymd.uix.button import MDFlatButton
from kivy.uix.image import Image
from kivymd.uix.dialog import MDDialog
import threading
from kivymd.app import MDApp
from kivy.uix.behaviors import ButtonBehavior
from kivy.clock import Clock  # ADDED: For thread-safe UI updates
from twilio.rest import Client 
import requests
import geocoder
import platform
import socket
from datetime import datetime

try:
    from config import (
        TWILIO_ACCOUNT_SID, 
        TWILIO_AUTH_TOKEN, 
        TWILIO_PHONE_NUMBER,
        EMERGENCY_CONTACTS,
        INCLUDE_LOCATION,
        CUSTOM_EMERGENCY_MESSAGE
    )
except ImportError:
    TWILIO_ACCOUNT_SID = None
    TWILIO_AUTH_TOKEN = None
    TWILIO_PHONE_NUMBER = None
    EMERGENCY_CONTACTS = []
    INCLUDE_LOCATION = True
    CUSTOM_EMERGENCY_MESSAGE = None


class ImageButton(ButtonBehavior, Image):
    pass

class MainWindow(MDBoxLayout):
    pass

class HelpWindow(MDBoxLayout):
    pass

class LocationService:
    def __init__(self):
        self.location_data = None
    
    def get_location(self):
        """Get current location using multiple methods"""
        try:
            g = geocoder.ip('me')
            if g.ok:
                self.location_data = {
                    'latitude': g.latlng[0],
                    'longitude': g.latlng[1],
                    'address': g.address,
                    'city': g.city,
                    'country': g.country
                }
                return self.location_data
        except Exception as e:
            print(f"IP geolocation failed: {e}")
        
        try:
            response = requests.get('http://ip-api.com/json/', timeout=5)
            data = response.json()
            if data['status'] == 'success':
                self.location_data = {
                    'latitude': data['lat'],
                    'longitude': data['lon'],
                    'address': f"{data['city']}, {data['regionName']}, {data['country']}",
                    'city': data['city'],
                    'country': data['country']
                }
                return self.location_data
        except Exception as e:
            print(f"IP-API geolocation failed: {e}")
        
        try:
            hostname = socket.gethostname()
            self.location_data = {
                'latitude': None,
                'longitude': None,
                'address': f"Device: {hostname}, OS: {platform.system()}",
                'city': 'Unknown',
                'country': 'Unknown'
            }
            return self.location_data
        except Exception as e:
            print(f"Fallback location failed: {e}")
            return None
    
    def get_google_maps_link(self):
        """Generate Google Maps link if coordinates are available"""
        if self.location_data and self.location_data['latitude'] and self.location_data['longitude']:
            lat = self.location_data['latitude']
            lng = self.location_data['longitude']
            return f"https://maps.google.com/maps?q={lat},{lng}"
        return None

class PopupWarning(MDBoxLayout):
    label_of_emergency = ObjectProperty(None)

class AudioRecWindow(MDBoxLayout):
    micbutton = ObjectProperty(None)

class ContentNavigationDrawer(MDBoxLayout):
    pass

class InternalStorageWindow(MDBoxLayout):
    pass

class TeamWindow(MDBoxLayout):
    pass

class FileLoader(MDBoxLayout):
    radius = ListProperty([10, 10, 10, 10])
    filechooser = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set the path to your main project folder after the widget is created
        Clock.schedule_once(self.set_default_path, 0.1)
    
    def set_default_path(self, dt):
        if self.filechooser:
            project_path = r"C:\Users\Admin\Desktop\MAIN PPROJECT\MAIN PPROJECT\Human_Scream_Detection_using_ml_and_deep_learning-main\scream_detection_main"
            if os.path.exists(project_path):
                self.filechooser.path = project_path
                print(f"📁 File chooser set to: {project_path}")

class UiApp(MDApp):
    dialog = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recscreen = None
        self.mainscreen = None
        self.screen_manager = ScreenManager()
        self.filename = None
        self.myrecording = None
        self.externally_stopped = False
        self.processing = False  # ADDED: Flag to prevent multiple processing
        self.dialog_shown = False  # ADDED: Flag to prevent dialog from showing multiple times
        self.current_processing_file = None  # ADDED: Track which file is being processed
        
        # Initialize SMS and Location services
        self.location_service = LocationService()
        self.sms_enabled = bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER)
        if self.sms_enabled:
            self.twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
    def build(self):
        self.theme_cls.primary_palette = "Pink"
        self.theme_cls.theme_style = "Dark"
        
        # Set explicit window size to ensure it appears
        from kivy.core.window import Window
        Window.size = (400, 700)
        print("🪟 Window size set to 400x700")

        self.mainscreen = MainWindow()
        screen = Screen(name='mainscreen')
        screen.add_widget(self.mainscreen)
        self.screen_manager.add_widget(screen)

        self.recscreen = AudioRecWindow()
        screen = Screen(name='recscreen')
        screen.add_widget(self.recscreen)
        self.screen_manager.add_widget(screen)

        self.internalstoragescreen = InternalStorageWindow()
        screen = Screen(name='internalstoragescreen')
        screen.add_widget(self.internalstoragescreen)
        self.screen_manager.add_widget(screen)

        self.helpscreen = HelpWindow()
        screen = Screen(name='helpscreen')
        screen.add_widget(self.helpscreen)
        self.screen_manager.add_widget(screen)

        self.fileloaderscreen = FileLoader()
        screen = Screen(name='fileloaderscreen')
        screen.add_widget(self.fileloaderscreen)
        self.screen_manager.add_widget(screen)

        self.teamscreen = TeamWindow()
        screen = Screen(name='teamscreen')
        screen.add_widget(self.teamscreen)
        self.screen_manager.add_widget(screen)

        self.popupwarningscreen = PopupWarning()
        screen = Screen(name='popupwarningscreen')
        screen.add_widget(self.popupwarningscreen)
        self.screen_manager.add_widget(screen)

        print("✅ All screens built successfully")
        return self.screen_manager
    
    def thread_for_rec(self):
        if self.recscreen.micbutton.source == "resources/icons/micon.png":
            fs = 44100
            seconds = 10
            print('rec started')
            
            if AUDIO_BACKEND == 'sounddevice':
                # Use sounddevice
                self.myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype=np.int16)
                sd.wait()
                if self.externally_stopped == True:
                    pass
                else:
                    write('recorded.wav', fs, self.myrecording)
                    Clock.schedule_once(lambda dt: toast("Finished"))
                    self.filename = "recorded.wav"
                    
            elif AUDIO_BACKEND == 'pyaudio':
                # Use pyaudio
                import pyaudio
                import wave
                
                CHUNK = 1024
                FORMAT = pyaudio.paInt16
                CHANNELS = 1
                RATE = 44100
                
                p = pyaudio.PyAudio()
                stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                               input=True, frames_per_buffer=CHUNK)
                
                frames = []
                for i in range(0, int(RATE / CHUNK * seconds)):
                    if self.externally_stopped:
                        break
                    data = stream.read(CHUNK)
                    frames.append(data)
                
                stream.stop_stream()
                stream.close()
                p.terminate()
                
                if not self.externally_stopped and frames:
                    wf = wave.open('recorded.wav', 'wb')
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                    wf.close()
                    Clock.schedule_once(lambda dt: toast("Finished"))
                    self.filename = "recorded.wav"
            else:
                Clock.schedule_once(lambda dt: toast("No audio backend available. Cannot record."))
                print("❌ No audio recording backend available")

    def show_popup(self, text):
        show = PopupWarning()
        show.label_of_emergency.text = text
        self.popupWindow = Popup(title="Popup Window", content=show, size_hint=(None, None), size=(400, 400))
        self.popupWindow.open()
    
    def close_popup(self):
        self.popupWindow.dismiss()

    def test_sms_location(self):
        """Test method to check SMS and location functionality"""
        if not self.sms_enabled:
            toast("SMS not configured")
            return
        
        toast("Testing location service...")
        location_data = self.location_service.get_location()
        
        if location_data:
            print("Location data:", location_data)
            toast(f"Location found: {location_data.get('city', 'Unknown')}")
        else:
            toast("Location not available")
        
        test_result = self.send_sms_alert(
            custom_message="🧪 TEST MESSAGE",
            risk_level="✅ TEST - System Check"
        )
        
        if test_result:
            toast("Test SMS sent successfully!")
        else:
            toast("Test SMS failed")

    # FIXED: Process sound in background thread
    def process_the_sound(self):
        """Start processing audio in background thread"""
        if self.processing:
            print("⏭️ Already processing - ignoring duplicate request")
            return
            
        if not hasattr(self, 'filename') or self.filename is None:
            toast("Please record or load an audio file first")
            return
        
        # Check if this file was already processed
        if self.current_processing_file == self.filename and self.dialog_shown:
            print(f"⏭️ File '{self.filename}' already processed - skipping")
            toast("This audio was already analyzed")
            return

        print(f"🎵 Starting to process: {self.filename}")
        self.processing = True
        self.dialog_shown = False  # Reset dialog flag for new processing
        self.current_processing_file = self.filename  # Remember which file we're processing
        toast("Processing audio... Please wait")
        
        # Run processing in background thread
        thread = threading.Thread(target=self._process_sound_thread, daemon=True)
        thread.start()

    def _process_sound_thread(self):
        """Background thread for audio processing - DOES NOT BLOCK UI"""
        try:
            # Double-check if already processed to prevent any repeats
            if self.dialog_shown:
                print("⏭️ Dialog already shown for this file - aborting duplicate processing")
                self.processing = False
                return
            
            print(f"🔍 Processing file: {self.current_processing_file}")
            
            from modelloader import process_file
            
            output1 = False
            output2 = False
            
            # Try SVM model
            try:
                from svm_based_model.model_loader_and_predict import svm_process
                output1 = svm_process(self.filename)
                print(f"SVM Model Result: {output1}")
            except ImportError:
                print("SVM model not available")
            except Exception as e:
                print(f"SVM model error: {e}")
                output1 = False
            
            # Try Deep Learning model
            try:
                dl_result = process_file(self.filename)
                print(f"Deep Learning Model Raw Result: {dl_result}")
                
                if isinstance(dl_result, (list, np.ndarray)):
                    if len(dl_result) > 0:
                        if isinstance(dl_result[0], (list, np.ndarray)) and len(dl_result[0]) > 0:
                            prediction_value = float(dl_result[0][0])
                        else:
                            prediction_value = float(dl_result[0])
                    else:
                        prediction_value = 0.0
                elif isinstance(dl_result, (int, float, np.number)):
                    prediction_value = float(dl_result)
                elif isinstance(dl_result, bool):
                    prediction_value = 1.0 if dl_result else 0.0
                else:
                    print(f"Unexpected DL result type: {type(dl_result)}")
                    prediction_value = 0.0
                
                output2 = prediction_value > 0.5
                print(f"Deep Learning Model Processed: {output2} (raw value: {prediction_value:.4f})")
                
            except Exception as e:
                print(f"Deep Learning model error: {e}")
                import traceback
                traceback.print_exc()
                output2 = False

            print(f"\n=== FINAL DECISION ===")
            print(f"SVM says: {output1}")
            print(f"Deep Learning says: {output2}")
            
            # CRITICAL: Mark as processed IMMEDIATELY to prevent any duplicate runs
            self.dialog_shown = True
            self.processing = False
            
            # Schedule UI updates on main thread using Clock
            if output1 and output2:
                text = "[size=30]Risk is [color=#FF0000]HIGH[/color]\nSending emergency alert...[/size]"
                Clock.schedule_once(lambda dt: self.show_popup(text))
                print("🚨 HIGH RISK - Both models detected scream!")
                Clock.schedule_once(lambda dt: self.send_sms_alert(
                    custom_message="🚨 CRITICAL EMERGENCY",
                    risk_level="🔴 HIGH - Both AI models detected human scream"
                ))
                
            elif output1 or output2:
                # MEDIUM RISK - Ask user if they're safe (ONLY ONCE)
                print("⚠️ MEDIUM RISK - One model detected scream! Showing dialog ONCE")
                Clock.schedule_once(lambda dt: self.show_safety_confirmation(), 0.1)
                
            else:
                print("✅ SAFE - No threat detected")
                Clock.schedule_once(lambda dt: toast("Analysis complete - You are safe"))
                
        except Exception as e:
            error_msg = str(e)
            print(f"Error processing sound: {error_msg}")
            import traceback
            traceback.print_exc()
            Clock.schedule_once(lambda dt: toast(f"Error: {error_msg}"))
            self.processing = False
            self.dialog_shown = True  # Mark as shown even on error
    
    def show_safety_confirmation(self):
        """Show dialog asking if user is safe (MEDIUM RISK)"""
        # Prevent showing multiple dialogs
        if self.dialog and self.dialog.get_state() == 'open':
            print("⚠️ Dialog already open - not showing duplicate")
            return
            
        # Close any existing dialog first
        if self.dialog:
            try:
                self.dialog.dismiss()
            except:
                pass
            self.dialog = None
        
        print("📋 Showing safety confirmation dialog")
        
        # Create new dialog
        self.dialog = MDDialog(
            title="⚠️ MEDIUM RISK DETECTED",
            text="[size=18]Potential scream detected.\n\n[b]Are you safe?[/b][/size]",
            buttons=[
                MDFlatButton(
                    text="I'M SAFE",
                    theme_text_color="Custom",
                    text_color=(0, 0.8, 0, 1),
                    on_release=lambda x: self.user_is_safe()
                ),
                MDFlatButton(
                    text="SEND HELP!",
                    theme_text_color="Custom",
                    text_color=(1, 0, 0, 1),
                    on_release=lambda x: self.user_needs_help()
                ),
            ],
            auto_dismiss=False  # Prevent accidental dismissal
        )
        self.dialog.open()
    
    def user_is_safe(self):
        """User confirmed they are safe"""
        if self.dialog:
            self.dialog.dismiss()
            self.dialog = None
        
        # Show thankful message
        toast("Thank God! Stay safe 🙏")
        print("✅ User confirmed safety - No alert sent")
        
        # Optional: Show a nice popup
        safe_text = "[size=25][color=#00FF00]✓[/color] Glad you're safe!\n\n[size=18]Thank you for confirming.\nStay vigilant and stay safe! 🙏[/size][/size]"
        self.show_popup(safe_text)
    
    def user_needs_help(self):
        """User needs help - send emergency SMS immediately"""
        if self.dialog:
            self.dialog.dismiss()
            self.dialog = None
        
        print("🚨 USER REQUESTED HELP - Sending emergency SMS!")
        toast("Sending emergency alert...")
        
        # Send SMS immediately with MEDIUM risk level
        self.send_sms_alert(
            custom_message="🚨 EMERGENCY - User confirmed they need help!",
            risk_level="🟠 MEDIUM - One AI model detected scream, user confirmed emergency"
        )
        
        # Show confirmation
        alert_text = "[size=25][color=#FF0000]🚨 ALERT SENT[/color]\n\n[size=18]Emergency contacts have been notified.\nHelp is on the way![/size][/size]"
        self.show_popup(alert_text)

    def mic_clicked(self):
        if AUDIO_BACKEND is None:
            toast("Audio recording not available")
            return
            
        if self.recscreen.micbutton.source == "resources/icons/micoff.png":
            self.recscreen.micbutton.source = "resources/icons/micon.png"
            self.externally_stopped = False
            toast("started")
            th = threading.Thread(target=self.thread_for_rec)
            th.start()
        else:
            try:
                if AUDIO_BACKEND == 'sounddevice':
                    sd.stop()
                    fs = 44100
                    write('recorded.wav', fs, self.myrecording)
                    self.filename = "recorded.wav"
                self.externally_stopped = True
                toast("stopped")
            except:
                print("Error stopping recording")
            self.recscreen.micbutton.source = "resources/icons/micoff.png"
    
    def loadfile(self, path, selection):
        if selection:
            self.filename = str(selection[0])
            toast(f"Loaded: {os.path.basename(self.filename)}")
            self.fileloaderscreen_to_internalstoragescreen()
    
    def internalstoragescreen_to_mainscreen(self):
        self.screen_manager.transition.direction = 'right'
        self.screen_manager.current = 'mainscreen'
    
    def mainscreen_to_internalstoragescreen(self):
        self.screen_manager.transition.direction = 'left'
        self.screen_manager.current = 'internalstoragescreen'
    
    def mainscreen_to_recscreen(self):
        self.screen_manager.transition.direction = 'left'
        self.screen_manager.current = 'recscreen'
    
    def recscreen_to_mainscreen(self):
        self.screen_manager.transition.direction = 'right'
        self.screen_manager.current = 'mainscreen'
    
    def internalstoragescreen_to_fileloader(self):
        self.screen_manager.transition.direction = 'up'
        self.screen_manager.current = 'fileloaderscreen'
    
    def fileloaderscreen_to_internalstoragescreen(self):
        self.screen_manager.transition.direction = 'down'
        self.screen_manager.current = 'internalstoragescreen'
    
    def mainscreen_to_helpscreen(self):
        self.screen_manager.transition.direction = 'down'
        self.screen_manager.current = 'helpscreen'
    
    def mainscreen_to_teamscreen(self):
        self.screen_manager.transition.direction = 'down'
        self.screen_manager.current = 'teamscreen'
    
    def backforcommonscreens(self):
        self.screen_manager.transition.direction = 'up'
        self.screen_manager.current = 'mainscreen'

    def send_sms_alert(self, custom_message=None, include_location=INCLUDE_LOCATION, risk_level="UNKNOWN"):
        """Send SMS alert with date, time, location, and risk level"""
        if not self.sms_enabled:
            toast("SMS not configured. Please check config.py")
            return False
        
        if not EMERGENCY_CONTACTS:
            toast("No emergency contacts configured")
            return False
        
        try:
            # Get current date and time
            from datetime import datetime
            now = datetime.now()
            date_str = now.strftime("%B %d, %Y")  # e.g., "January 12, 2026"
            time_str = now.strftime("%I:%M:%S %p")  # e.g., "03:45:30 PM"
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")  # ISO format for logs
            
            # Get location information
            location_info = ""
            maps_link = ""
            
            if include_location:
                print("Getting location...")
                location_data = self.location_service.get_location()
                if location_data:
                    if location_data['latitude'] and location_data['longitude']:
                        location_info = f"\n📍 Location: {location_data['address']}"
                        maps_link = f"\n🗺️ Map: {self.location_service.get_google_maps_link()}"
                    else:
                        location_info = f"\n📍 Device: {location_data['address']}"
            
            # Build the message with all required information
            if custom_message:
                message_body = custom_message
            elif CUSTOM_EMERGENCY_MESSAGE:
                message_body = CUSTOM_EMERGENCY_MESSAGE
            else:
                message_body = "🚨 EMERGENCY ALERT"
            
            # Format the complete SMS
            full_message = f"""{message_body}

⚠️ RISK LEVEL: {risk_level}

📅 Date: {date_str}
⏰ Time: {time_str}{location_info}{maps_link}

This is an automated alert from the Scream Detection System."""
            
            print(f"\n📱 SMS Alert Content:\n{full_message}\n")
            
            success_count = 0
            total_contacts = len(EMERGENCY_CONTACTS)
            
            for contact in EMERGENCY_CONTACTS:
                try:
                    message = self.twilio_client.messages.create(
                        body=full_message,
                        from_=TWILIO_PHONE_NUMBER,
                        to=contact
                    )
                    print(f"✅ SMS sent to {contact} with SID: {message.sid}")
                    success_count += 1
                except Exception as e:
                    print(f"❌ Failed to send SMS to {contact}: {str(e)}")
            
            if success_count > 0:
                toast(f"Alert SMS sent to {success_count}/{total_contacts} contacts")
                return True
            else:
                toast("Failed to send SMS to any contacts")
                return False
                
        except Exception as e:
            print(f"SMS service error: {str(e)}")
            toast(f"SMS failed: {str(e)}")
            return False

# COMPATIBILITY FIX: Register MDToolbar as alias for MDTopAppBar
from kivy.factory import Factory
try:
    from kivymd.uix.toolbar import MDTopAppBar
    Factory.register('MDToolbar', cls=MDTopAppBar)
    print("✅ MDToolbar compatibility enabled")
except Exception as e:
    print(f"⚠️ Could not register MDToolbar: {e}")

if __name__ == '__main__':
    try:
        LabelBase.register(name='second', fn_regular='FFF_Tusj.ttf')
        LabelBase.register(name='first', fn_regular='Pacifico.ttf')
    except:
        print("⚠️ Custom fonts not found, using default fonts")
    
    print("🚀 Starting UiApp...")
    UiApp().run()