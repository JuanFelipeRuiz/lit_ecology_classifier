#!/bin/bash

download_and_extract() {
    local dataset="$1"
    local url=""

    # Define available links
    case "$dataset" in
        "ZooLake1")
            url="https://opendata.eawag.ch/dataset/52b6ba86-5ecb-448c-8c01-eec7cb209dc7/resource/1cc785fa-36c2-447d-bb11-92ce1d1f3f2d/download/data.zip"
            ;;
        "ZooLake2")
            url="https://opendata.eawag.ch/dataset/1aee0a2a-e1a7-4910-8151-4fc67c15dc63/resource/e241f3df-24f5-492a-9d5c-c17cacab28f2/download/2562ce1c-5015-4599-9a4e-3a1a1026af50-zoolake2.zip"
            ;;
        "ZooLake3")
            echo "Coming soon" 
            return 1
            ;;
        *)
            echo "Invalid choice. Please choose a valid option: ZooLake1, ZooLake2, or ZooLake3."
            return 1
            ;;
    esac

    # Create the folder named after the dataset if it doesn't exist
    mkdir -p "data/$dataset"

    # Extract the filename from the URL
    local filename="$dataset.zip"

    # Download the file to the folder named after the link choice
    curl -L "$url" -o "data/$dataset/$filename"

    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "Download successful: $dataset/$filename, starting extraction..."

        # Check if the file is a zip
        if [[ "$filename" == *.zip ]]; then
            unzip -o -q "$dataset/$filename" -d "$dataset"
            if [ $? -eq 0 ]; then
                rm "data/$dataset/$filename"
                echo "Extraction successful: Files extracted to ./data/$dataset/"
            else
                echo "Extraction failed. Please check the zip file."
                return 1
            fi
        else
            echo "Downloaded file is not a zip file. No extraction performed."
        fi
    else
        echo "Download failed. Please check the URL or your internet connection."
        return 1
    fi
}

# open up config/dataset_versions.json and write the version of the dataset that was downloaded and the path to the dataset




download_and_extract "$1" 

