import sys
import os
import time
import tempfile
import threading
import queue
import urllib.parse
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# GUI
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QTextEdit, QComboBox, QWidget, 
                            QScrollArea, QFrame, QLineEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette

# Audio
import speech_recognition as sr
import pygame

# LLM for conversation
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class SpeechRecognitionThread(QThread):
    """Thread for speech recognition"""
    transcribed = pyqtSignal(str)
    status = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.recognizer = sr.Recognizer()
        self.running = False
        
    def run(self):
        self.running = True
        self.status.emit("Listening...")
        
        while self.running:
            try:
                # Use the default microphone as the audio source
                with sr.Microphone() as source:
                    # Adjust for ambient noise
                    self.recognizer.adjust_for_ambient_noise(source)
                    # Listen for the first phrase and extract it into audio data
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                try:
                    # Use Google's speech recognition
                    text = self.recognizer.recognize_google(audio)
                    if text:
                        self.transcribed.emit(text)
                except sr.UnknownValueError:
                    # Speech was unintelligible
                    self.status.emit("Could not understand audio")
                except sr.RequestError as e:
                    self.status.emit(f"Recognition error: {e}")     
                    
            except Exception as e:
                self.status.emit(f"Error listening: {str(e)}")
                time.sleep(2)  # Wait before trying again
    
    def stop(self):
        self.running = False
        self.status.emit("Stopped listening")


class ConversationalBot:
    """Core logic for the conversational assistant"""
    
    def __init__(self, groq_api_key=None, model_name="llama3-70b-8192"):
        # Initialize Groq LLM
        self.llm = ChatGroq(
            groq_api_key=groq_api_key or os.getenv("GROQ_API_KEY") or "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",
            model_name=model_name
        )
        
        # Initialize conversation history with system message for concise responses
        self.system_message = SystemMessage(content=(
            "You are a helpful assistant that provides direct, concise answers without unnecessary text. "
            "Keep responses brief, straightforward, and to the point. "
            "Answer questions directly without long introductions or excessive explanations. "
            "Use simple language and be efficient with words."
        ))
        
        self.conversation_history = [self.system_message]
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
    
    def process_input(self, text_input):
        """Process text input and generate a response"""
        if not text_input:
            return "I didn't catch that. Could you please repeat?"
        
        # Add user message to conversation history
        self.conversation_history.append(HumanMessage(content=text_input))
        
        # Get response from LLM
        try:
            # Create the messages list from conversation history (limit to last 10 exchanges but always include system message)
            filtered_history = [self.system_message] + self.conversation_history[-10:]
            # Remove duplicates if the system message appears twice
            messages = []
            for msg in filtered_history:
                if not (isinstance(msg, SystemMessage) and len(messages) > 0 and isinstance(messages[0], SystemMessage)):
                    messages.append(msg)
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Extract the content from the response
            response_text = response.content
            
            # Add AI response to conversation history
            self.conversation_history.append(AIMessage(content=response_text))
            
            return response_text
        
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I'm having trouble connecting to my language model. Please try again later."
    
    # Modify the generate_speech method in the ConversationalBot class:

    def generate_speech(self, text):
        """Convert text to speech and play it"""
        if not text or len(text) == 0:
            return
        
        try:
            # Check if the response contains code (based on common indicators)
            contains_code = False
            code_indicators = ["```", "def ", "class ", "function", "public static void", "import ", "#include", "package ", "from "]
            
            for indicator in code_indicators:
                if indicator in text:
                    contains_code = True
                    break
            
            if contains_code:
                # Extract only the first sentence or introduction before the code
                import re
                first_part = re.split(r'(```|def |class |function|public static void|import |#include|package |from )', text)[0].strip()
                
                # If no clear introduction, create a generic one
                if not first_part or len(first_part) < 10:
                    first_part = "Here's the code you requested."
                    
                # Only speak the introduction
                text_to_speak = first_part
            else:
                # For non-code responses, speak the full text
                text_to_speak = text
                
            # Break long text into chunks (Google TTS has a limit)
            max_length = 200
            chunks = [text_to_speak[i:i+max_length] for i in range(0, len(text_to_speak), max_length)]
            
            # Create a temporary file to store the audio
            temp_files = []
            
            for i, chunk in enumerate(chunks):
                # URL encode the text
                encoded_text = urllib.parse.quote(chunk)
                
                # Use Google Translate TTS with speed parameter
                url = f"https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&q={encoded_text}&tl=en&total=1&idx=0&textlen={len(chunk)}&speed=2.0"
                
                # Make the request
                response = requests.get(url)
                
                if response.status_code == 200:
                    # Save to a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                        temp_audio.write(response.content)
                        temp_files.append(temp_audio.name)
                else:
                    print(f"TTS service returned status code {response.status_code}")
            
            # Play audio files sequentially
            for audio_file in temp_files:
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                # Clean up
                os.unlink(audio_file)
                
        except Exception as e:
            print(f"Error in speech generation: {str(e)}")


class VoiceAssistantApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize the bot
        self.bot = ConversationalBot()
        
        # Initialize the speech recognition thread
        self.speech_thread = SpeechRecognitionThread()
        self.speech_thread.transcribed.connect(self.on_transcription)
        self.speech_thread.status.connect(self.update_status)
        
        # Setup UI
        self.init_ui()
        
        # State
        self.is_listening = False
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Voice Assistant")
        self.setGeometry(100, 100, 800, 600)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("LLM Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        main_layout.addLayout(model_layout)
        
        # Status display
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # Conversation display (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.conversation_widget = QWidget()
        self.conversation_layout = QVBoxLayout(self.conversation_widget)
        self.conversation_layout.setAlignment(Qt.AlignTop)
        self.conversation_layout.setSpacing(10)
        
        scroll.setWidget(self.conversation_widget)
        main_layout.addWidget(scroll)
        
        # Input area - Text input and voice buttons
        input_layout = QHBoxLayout()
        
        # Text input
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type your message here...")
        self.text_input.returnPressed.connect(self.on_send_clicked)
        
        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.on_send_clicked)
        
        # Voice control buttons
        self.listen_button = QPushButton("Start Listening")
        self.listen_button.clicked.connect(self.toggle_listening)
        
        input_layout.addWidget(self.text_input, 6)
        input_layout.addWidget(self.send_button, 1)
        input_layout.addWidget(self.listen_button, 3)
        
        main_layout.addLayout(input_layout)
        
        # Clear conversation button
        self.clear_button = QPushButton("Clear Conversation")
        self.clear_button.clicked.connect(self.clear_conversation)
        main_layout.addWidget(self.clear_button)
        
        # Set the main widget
        self.setCentralWidget(main_widget)
        
        # Welcome message
        self.add_assistant_message("Hello! I'm your voice assistant. Click 'Start Listening' and speak to me, or type a message.")
    
    def toggle_listening(self):
        """Toggle the listening state"""
        if not self.is_listening:
            # Start listening
            self.is_listening = True
            self.listen_button.setText("Stop Listening")
            self.speech_thread.start()
        else:
            # Stop listening
            self.is_listening = False
            self.listen_button.setText("Start Listening")
            self.speech_thread.stop()
            self.speech_thread.wait()  # Wait for the thread to finish
    
    def on_send_clicked(self):
        """Handle send button click"""
        user_input = self.text_input.text().strip()
        if user_input:
            # Clear the input field
            self.text_input.clear()
            
            # Add user message to UI
            self.add_user_message(user_input)
            
            # Process with bot
            self.process_user_input(user_input)
    
    def on_transcription(self, text):
        """Handle transcribed text"""
        # Add user message to UI
        self.add_user_message(text)
        
        # Process with bot
        self.update_status("Thinking...")
        
        # Use a timer to process in the background to keep UI responsive
        def process_message():
            response = self.bot.process_input(text)
            self.add_assistant_message(response)
            self.bot.generate_speech(response)
            self.update_status("Listening...")
        
        QTimer.singleShot(100, process_message)
    
    def process_user_input(self, text):
        """Process user input and generate a response"""
        self.update_status("Thinking...")
        
        # Use a timer to process in the background to keep UI responsive
        def process_message():
            response = self.bot.process_input(text)
            self.add_assistant_message(response)
            self.bot.generate_speech(response)
            if self.is_listening:
                self.update_status("Listening...")
            else:
                self.update_status("Ready")
        
        QTimer.singleShot(100, process_message)
    
    def add_user_message(self, text):
        """Add a user message to the conversation display"""
        message_frame = QFrame()
        message_frame.setFrameShape(QFrame.StyledPanel)
        message_frame.setStyleSheet("background-color: #e1f5fe; border-radius: 10px;")
        
        layout = QVBoxLayout(message_frame)
        
        label = QLabel("You:")
        label.setStyleSheet("font-weight: bold; color: #01579b; background-color: transparent;")
        
        content = QLabel(text)
        content.setWordWrap(True)
        content.setStyleSheet("background-color: transparent;")
        
        layout.addWidget(label)
        layout.addWidget(content)
        
        self.conversation_layout.addWidget(message_frame)
        
        # Scroll to bottom
        QTimer.singleShot(100, self.scroll_to_bottom)
    
    def add_assistant_message(self, text):
        """Add an assistant message to the conversation display"""
        message_frame = QFrame()
        message_frame.setFrameShape(QFrame.StyledPanel)
        message_frame.setStyleSheet("background-color: #f1f8e9; border-radius: 10px;")
        
        layout = QVBoxLayout(message_frame)
        
        label = QLabel("Assistant:")
        label.setStyleSheet("font-weight: bold; color: #33691e; background-color: transparent;")
        
        content = QLabel(text)
        content.setWordWrap(True)
        content.setStyleSheet("background-color: transparent;")
        
        layout.addWidget(label)
        layout.addWidget(content)
        
        self.conversation_layout.addWidget(message_frame)
        
        # Scroll to bottom
        QTimer.singleShot(100, self.scroll_to_bottom)
    
    def scroll_to_bottom(self):
        """Scroll the conversation view to the bottom"""
        # Find the QScrollArea's vertical scrollbar
        scrollbar = None
        parent = self.conversation_widget.parent()
        if hasattr(parent, 'verticalScrollBar'):
            scrollbar = parent.verticalScrollBar()
        
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())
    
    def update_status(self, text):
        """Update the status display"""
        self.status_label.setText(text)
    
    def clear_conversation(self):
        """Clear the conversation history"""
        # Clear the UI
        while self.conversation_layout.count():
            item = self.conversation_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Clear the bot's conversation history
        self.bot.conversation_history = [self.bot.system_message]
        
        # Add welcome message again
        self.add_assistant_message("Conversation cleared. How can I help you?")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Make sure the speech thread is stopped
        if self.speech_thread.isRunning():
            self.speech_thread.stop()
            self.speech_thread.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show the main window
    window = VoiceAssistantApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()