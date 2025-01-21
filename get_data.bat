@echo off
setlocal

:: Define the URL based on the argument
set dataset=%1
if "%dataset%"=="" (
    echo Usage: download_and_extract.bat ZooLake1
    exit /b
)

if "%dataset%"=="ZooLake1" (
    set url=https://opendata.eawag.ch/dataset/52b6ba86-5ecb-448c-8c01-eec7cb209dc7/resource/1cc785fa-36c2-447d-bb11-92ce1d1f3f2d/download/data.zip
) else if "%dataset%"=="ZooLake2" (
    set url=https://opendata.eawag.ch/dataset/1aee0a2a-e1a7-4910-8151-4fc67c15dc63/resource/e241f3df-24f5-492a-9d5c-c17cacab28f2/download/2562ce1c-5015-4599-9a4e-3a1a1026af50-zoolake2.zip
) else if "%dataset%"=="ZooLake3" (
    set url=https://example.com/dataset3.zip
) else (
    echo Invalid link choice. Please choose ZooLake1, ZooLake2, or ZooLake3.
    exit /b
)

:: Set the download folder
set download_folder=%cd%\%dataset%

:: Create the folder
if not exist "%download_folder%" (
    mkdir "%download_folder%"
)

:: Download the file using curl
set temp_file=%download_folder%\temp__filename.zip
curl -L "%url%" -o "%temp_file%"
if errorlevel 1 (
    echo Download failed. Please check the URL or your internet connection.
    exit /b
)

echo Download successful: %temp_file%

:: Extract the file using tar (Windows 10+)
tar -xf "%temp_file%" -C "%download_folder%" >nul 2>&1
if errorlevel 1 (
    echo Extraction failed. Please check the zip file.
    exit /b
)

:: Delete the temp zip file
del "%temp_file%"
echo Extraction successful: Files extracted to %download_folder%.
