param (
    [string]$dataset
)

# Check if a dataset was provided, else raise an error
if (-not $dataset) {
   write-host "Please provide a dataset name. Usage: get_data.ps1 <dataset_name>"
    exit
}


# Set the download URL based on the given dataset
switch ($dataset) {
    "ZooLake1" { $url = "https://opendata.eawag.ch/dataset/52b6ba86-5ecb-448c-8c01-eec7cb209dc7/resource/1cc785fa-36c2-447d-bb11-92ce1d1f3f2d/download/data.zip" }
    "ZooLake2" { $url = "https://example.com/dataset2.zip" }
    "ZooLake3" { $url = "https://example.com/dataset3.zip" }
    default {
        Write-Host "Invalid link choice. Please choose ZooLake1, ZooLake2, or ZooLake3."
        exit
    }
}

# Set the download folder
$download_folder = Join-Path (Get-Location) $dataset
if (-not (Test-Path $download_folder)) {
    New-Item -ItemType Directory -Path $download_folder | Out-Null
}

# Create temporary file path
$temp_file = Join-Path $download_folder "temp__filename.zip"

# Download the file
Write-Host "Downloading $url..."
Invoke-WebRequest -Uri $url -OutFile $temp_file -UseBasicParsing
if (-not $?) {
    Write-Host "Download failed. Please check the URL or your internet connection."
    exit
}

Write-Host "Download successful: $temp_file"

# Extract the file
Write-Host "Extracting files..."
Expand-Archive -Path $temp_file -DestinationPath $download_folder -Force
if (-not $?) {
    Write-Host "Extraction failed. Please check the zip file."
    exit
}

# Remove the zip file
Remove-Item $temp_file
Write-Host "Extraction successful: Files extracted to $download_folder/"
