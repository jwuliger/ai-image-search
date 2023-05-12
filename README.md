# JW's AI Image Search

This is an image search application that uses OpenAI's CLIP model to perform semantic searches in a local directory of images. It allows you to find images that match a particular keyword or phrase. The application is built with Streamlit, a framework for building machine learning and data science web apps.

## Features

- Extracts and stores image features for efficient searching.
- Performs semantic searches in images using natural language keywords.
- Displays search results in a web interface, with images sorted by their relevance to the keyword.
- Allows navigation through pages of results.

## Requirements

- Python 3.7 or later
- [Streamlit](https://streamlit.io)
- [PyTorch](https://pytorch.org)
- [OpenAI's CLIP](https://github.com/openai/CLIP)
- [Pillow](https://pillow.readthedocs.io)

You can install the Python dependencies with pip:


## Usage

1. Clone the repository and navigate to its directory.
2. Replace `'C:/Users/jared/Pictures/MidJourney'` in the code with the path to your own image directory, and `'image_features.pkl'` with your preferred path and filename for the file that will store the image features.
3. Run the Streamlit app with the following command: `streamlit run app.py`
4. Open your web browser and go to `http://localhost:8501` to see the app.
5. Enter a keyword to search for in the text input box and press enter.

## Note

The code is written in a way that it uses GPU if available for faster computation. If a GPU is not available, it will use the CPU.
