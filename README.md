## Guide Bot Simulation

this is simulation using the MVSIM , ros and python in this we use the openai API to control the bot in the simulation 

## Setup instructions
```bash
uv sync
```
## Download the pre build enviroment 
```bash
git clone https://github.com/MRPT/mvsim-models.git
```

## Change Directory
```bash
cd mvsim-models
```
# Extract All ZIP Files
```bash
# extract all model  zip powershell script below 
New-Item -ItemType Directory -Name "zip" -Force; Get-ChildItem -File -Filter "*.zip" | ForEach-Object { $name = $_.BaseName; Expand-Archive -LiteralPath $_.FullName -DestinationPath (Join-Path $PWD $name) -Force; Move-Item -LiteralPath $_.FullName -Destination ".\zip\" }
```
This guide explains how to extract all ZIP files in a directory into separate folders and move the original ZIP files into a folder named `zip`.

## Directory Structure

### Before

```text
mvsim-models/
├── turtlebot3.zip
├── Apartment.zip
├── Bed.zip
├── README.md
└── other files...
```

### After

```text
mvsim-models/
├── turtlebot3/
├── Apartment/
├── Bed/
├── zip/
│   ├── turtlebot3.zip
│   ├── Apartment.zip
│   └── Bed.zip
├── README.md
└── other files...
```

## Requirements

- Windows
- PowerShell
- ZIP files in the directory where you run the command

##
