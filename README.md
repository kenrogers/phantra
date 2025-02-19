# Terrapin 🐢

Turn your YouTube videos into engaging LinkedIn content, automagically.

## A Note on the Actual Content

Terrapin is in the very early prototype stages. The content it generates is not yet very good. I'll be working on improvements and refinements to the process but at the moment I would not recommend using this LinkedIn content as-is.

## Features

- 🎥 YouTube transcript extraction
- 🤖 Intelligent content processing using GPT-4
- 📊 Adaptive RAG for longer videos
- 💼 Professional LinkedIn post generation
- ✨ Automatic post optimization
- 📥 Easy post downloading

## Setup

1. Clone the repository
2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# OR
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app.py
```

## Usage

1. Launch the app
2. Enter your OpenAI API key in the sidebar
3. Paste a YouTube video URL
4. Click "Generate LinkedIn Posts"
5. Download or copy your generated posts

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages listed in `requirements.txt`

## How It Works

Terrapin processes videos through several stages:

1. **Transcript Extraction**: Pulls the video transcript using YouTube's API
2. **Content Analysis**: Uses either direct processing or RAG based on transcript length
3. **Insight Generation**: Extracts key insights using GPT-4
4. **Post Creation**: Converts insights into engaging LinkedIn posts
5. **Optimization**: Refines posts for maximum engagement

## License

MIT

## Contributing

Pull requests welcome! Please ensure you test your changes before submitting.
