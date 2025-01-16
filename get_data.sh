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
            echo "Coming soon" 
            return 1
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
    mkdir -p "$dataset"

    # Extract the filename from the URL
    local filename=$(basename "$url")

    # Download the file to the folder named after the link choice
    curl -L "$url" -o "$dataset/$filename"

    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "Download successful: $dataset/$filename, starting extraction..."

        # Check if the file is a zip
        if [[ "$filename" == *.zip ]]; then
            unzip -o -q "$dataset/$filename" -d "$dataset"
            if [ $? -eq 0 ]; then
                rm "$dataset/$filename"
                echo "Extraction successful: Files extracted to ./$dataset/"
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

if [[ "${BASH_SOURCE[0]}" != "${0}" && -n "$1" ]]; then
    download_and_extract "$1"
fi
