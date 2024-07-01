#!/bin/bash

# Ensure the script takes the input CSV file as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 path/to/csvfile.csv"
    exit 1
fi

# Input CSV file
CSV_FILE=$1

# Read the CSV file line by line
while IFS=',' read -r ImageID Subset OriginalURL OriginalLandingURL License AuthorProfileURL Author Title OriginalSize OriginalMD5 Thumbnail300KURL Rotation; do
    # Skip the header line
    if [ "$ImageID" == "ImageID" ]; then
        continue
    fi

    # Write the ImageID into a text file in the validation directory
    echo "$Subset/$ImageID"
done < "$CSV_FILE"

