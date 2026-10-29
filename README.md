# Guide Bot Simulation

This project simulates a college guide robot using **MVSIM, ROS, and Python**. The OpenAI API is used to process user instructions and help control the robot in the simulation.

## Prerequisites

Install the following tools:

- Python
- `uv` Python package manager
- Git
- MVSIM
- ROS 2 (compatible with your MVSIM setup)
- An OpenAI API key

> **Note:** ROS and MVSIM installation steps vary by operating system. Install and configure them according to their official documentation before running the simulation.

## 1. Clone the Project

Repository: https://github.com/Sriman-ml97/SImulation_bot

```bash
git clone https://github.com/Sriman-ml97/SImulation_bot.git
cd SImulation_bot
```

## 2. Install Project Dependencies

Run this command from the project root:

```bash
uv sync
```

This installs the dependencies declared in `pyproject.toml` and synchronizes the project's virtual environment.

## 3. Download the MVSIM Models

Clone the models repository from MRPT:

```bash
git clone https://github.com/MRPT/mvsim-models.git
cd mvsim-models
```

## 4. Extract All ZIP Files

The following commands extract each ZIP archive into a separate directory and move the original archive into a folder named `zip`.

### Windows (PowerShell)

Run inside the `mvsim-models` directory:

```powershell
New-Item -ItemType Directory -Name "zip" -Force

Get-ChildItem -File -Filter "*.zip" | ForEach-Object {
    $archive = $_
    $destination = Join-Path $PWD $archive.BaseName

    Expand-Archive -LiteralPath $archive.FullName -DestinationPath $destination -Force

    if ($?) {
        Move-Item -LiteralPath $archive.FullName -Destination ".\zip\"
    }
}
```

### Linux (Ubuntu/Debian)

Install `unzip` if necessary:

```bash
sudo apt update
sudo apt install unzip
```

Run inside the `mvsim-models` directory:

```bash
mkdir -p zip

for file in *.zip; do
    [ -f "$file" ] || continue

    name="${file%.zip}"

    if unzip -o "$file" -d "$name"; then
        mv -- "$file" zip/
    else
        echo "Failed to extract: $file"
    fi
done
```

Both scripts process ZIP files in the current directory, not recursively.

## 5. Directory Structure

Example structure after extraction:

```text
mvsim-models/
├── turtlebot3/
├── Apartment/
├── Bed/
├── zip/
│   ├── turtlebot3.zip
│   ├── Apartment.zip
│   └── Bed.zip
└── README.md
```

The filenames above are examples. The actual contents depend on the downloaded repository.

## 6. Configure the OpenAI API Key

Create a `.env` file in the **project root directory**:

```env
OPENAI_API_KEY=your_openai_api_key
```

Replace the placeholder with your API key. Ensure your Python application loads the environment variables using an appropriate library, such as `python-dotenv`.

Add the following to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
```

**Never commit or share your API key.**

## 7. Run the Simulation

Return to the project root directory:

```bash
cd ..
```

If needed, activate the virtual environment.

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux (Bash):**

```bash
source .venv/bin/activate
```

Then launch the MVSIM environment and Python controller using the commands specified by your project.

For example, if your entry-point script is `main.py`:

```bash
uv run python main.py
```

Replace `main.py` with the actual entry-point filename in your repository.

## Troubleshooting

- **`uv` not found:** Install `uv` and ensure it is available on your PATH.
- **`unzip` not found:** Install it using your Linux package manager.
- **ROS or MVSIM not found:** Check the installation and environment configuration.
- **OpenAI API errors:** Verify your API key, API access, and account usage limits.
- **ZIP extraction errors:** Check the archive and available disk space.
- **Simulation cannot find a model:** Verify that the world configuration references the correct extracted model paths.

## Notes

- Run the extraction commands from inside `mvsim-models`.
- Inspect the repository before extracting; some assets may already be unpacked.
- Existing extracted files may be overwritten by the supplied commands.
- Retain the `zip` directory if you want to keep the original archives.
- Ensure the simulation's world configuration points to the appropriate models and assets.
