# Dataset Module - BestMan_Chemistry

This module is responsible for processing images to generate a structured JSON file containing information about chemical containers and substances. It reads images from the `Images` directory and outputs a JSON file named `objects_json`.

## Features

- **Input**: Processes image files from the `Images` directory.
- **Output**: Generates a JSON file (`objects_json`) containing structured information about chemical containers and substances.
> ***Note**: Users can also edit objects_json.json manually.*
- **Data Structure**: Each object in the JSON file includes detailed attributes such as category, name, size, pose, physical and chemical attributes, and an image URL.

## Usage Instructions

1. **Setup**

    Place all relevant images in the `Images` directory.

2. **Run Script**

    ```bash
    cd Dataset
    pip install openai
    python3 CEDataset.py
    ```