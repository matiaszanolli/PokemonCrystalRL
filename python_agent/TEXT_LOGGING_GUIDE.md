# 📝 Pokemon Crystal Text Transcription Logging

This system automatically captures and logs all detected text from Pokemon Crystal gameplay for analysis, debugging, and creating transcripts of training sessions.

## 🎯 Features

- **Automatic Text Detection**: Captures all text from dialogue, menus, UI elements, and world text
- **SQLite Database**: Structured storage for searching and analysis
- **Session Management**: Each training session gets a unique ID and timestamp
- **Real-time Logging**: Text is logged as it's detected during gameplay
- **Transcript Export**: Human-readable transcripts for each session
- **Deduplication**: Prevents spam from repeated identical text
- **Search Functionality**: Query specific text content or locations

## 🚀 How It Works

The text logger is automatically integrated into the monitored training system:

1. **Initialization**: Logger starts when training begins
2. **Text Detection**: Vision processor detects text from screenshots
3. **Logging**: Detected text is stored with metadata (location, confidence, frame)
4. **Transcript Generation**: At session end, a readable transcript is created

## 📁 File Structure

```
gameplay_transcripts/
├── text_transcriptions.db     # SQLite database with all text data
└── session_YYYYMMDD_HHMMSS_transcript.txt  # Human-readable transcripts
```

## 📊 Database Schema

### Text Entries Table
- `timestamp`: When text was detected
- `frame_number`: Game frame number
- `session_id`: Unique session identifier
- `screen_type`: Type of screen (dialogue, menu, overworld, etc.)
- `text_content`: The actual detected text
- `text_location`: Where text appeared (dialogue, menu, ui, world)
- `confidence`: Detection confidence (0.0-1.0)
- `bbox_*`: Bounding box coordinates
- `text_hash`: Unique hash for deduplication

### Sessions Table
- `session_id`: Unique identifier
- `start_time` / `end_time`: Session timestamps
- `total_frames`: Number of frames processed
- `total_text_detections`: Total text instances found
- `unique_text_count`: Number of unique text strings
- `session_summary`: Brief summary of session

## 📖 Usage Examples

### Automatic Usage (Recommended)
Text logging is automatically enabled when running monitored training:

```bash
python monitored_training.py --rom ../roms/pokemon_crystal.gbc --episodes 5
```

### Manual Usage
```python
from text_logger import PokemonTextLogger
from vision_processor import PokemonVisionProcessor

# Initialize
logger = PokemonTextLogger("my_session_logs")
vision = PokemonVisionProcessor()

# Process screenshots
screenshot = get_game_screenshot()  # Your screenshot source
visual_context = vision.process_screenshot(screenshot)
logger.log_visual_context(visual_context)

# Export transcript
transcript_file = logger.close_session()
```

### Search Functionality
```python
# Search for specific text
results = logger.search_text("POKEMON", location="dialogue")

# Get session statistics
stats = logger.get_session_stats()

# Get recent dialogue
dialogue = logger.get_dialogue_history(count=10)
```

## 📋 What Gets Logged

### Text Locations
- **📢 Dialogue**: Character speech, story text, dialogue boxes
- **📋 Menu**: Menu options, navigation text, buttons
- **🔧 UI**: Health bars, stats, level indicators, money
- **🌍 World**: Signs, location names, in-game text

### Screen Types
- **Dialogue**: Conversation screens
- **Menu**: Game menus and navigation
- **Overworld**: Exploration and movement
- **Battle**: Combat screens
- **Intro**: Title and startup screens

## 📈 Session Statistics

Each session tracks:
- Total frames processed
- Text detections per frame
- Unique text strings found
- Text distribution by location
- Text distribution by screen type
- Dialogue-specific statistics

## 🔍 Example Transcript

```
Pokemon Crystal Gameplay Transcript
Session: session_20250814_162345
Generated: 2025-08-14 16:25:30
Total Frames: 150
Total Text Detections: 45
============================================================

Frame 1 (overworld):
  🌍 WORLD: NEW BARK TOWN
  🔧 UI: HP

Frame 5 (dialogue):
  💬 DIALOGUE: Hello! Welcome to the world of POKEMON!
  💬 DIALOGUE: My name is OAK.

Frame 12 (menu):
  📋 MENU: NEW GAME
  📋 MENU: CONTINUE
```

## ⚙️ Configuration

### Database Location
Change the database directory:
```python
logger = PokemonTextLogger("custom_directory")
```

### Text Detection Sensitivity
Adjust vision processor settings for more/less text detection:
```python
# In vision_processor.py, modify detection thresholds
```

## 🔧 Troubleshooting

### No Text Detected
- Check if vision processor is enabled
- Verify screenshot quality and resolution
- Review text detection thresholds

### Database Errors
- Ensure write permissions for log directory
- Check available disk space
- Verify SQLite installation

### Missing Transcripts
- Check if session was properly closed
- Verify file permissions in output directory
- Look for error messages during export

## 📚 Integration with Training

The text logger seamlessly integrates with:

- **Monitored Training**: Automatic logging during training sessions
- **Web Dashboard**: Real-time text statistics
- **Vision Processing**: Direct integration with text detection
- **Performance Monitoring**: Text detection performance metrics

## 🎯 Use Cases

1. **Training Analysis**: Understanding what text the agent encounters
2. **Dialogue Tracking**: Following story progression and conversations
3. **Debug Information**: Identifying OCR accuracy and issues
4. **Gameplay Documentation**: Creating records of training sessions
5. **Research Data**: Collecting text data for analysis
6. **Progress Monitoring**: Tracking game progression through text cues

## 📝 Output Locations

After a training session, find your logs at:
- **Database**: `gameplay_transcripts/text_transcriptions.db`
- **Transcript**: `gameplay_transcripts/session_[timestamp]_transcript.txt`
- **Statistics**: Printed to console at session end

The text logging system provides comprehensive documentation of all text encountered during Pokemon Crystal gameplay, making it invaluable for analysis, debugging, and understanding agent behavior.
