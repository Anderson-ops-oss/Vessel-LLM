[Setup]
AppName=LLM-RAG Suite
AppVersion=1.0
DefaultDirName={pf}\LLM-RAG Suite
DefaultGroupName=LLM-RAG Suite
UninstallDisplayIcon={app}\web_folder\img\logo.ico
OutputDir=dist
OutputBaseFilename=LLM-RAG-Setup
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\server\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "web_folder\*"; DestDir: "{app}\web_folder"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "rag_models\*"; DestDir: "{app}\rag_models"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "cache\*"; DestDir: "{app}\cache"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "start.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "web_folder\img\logo.ico"; DestDir: "{app}\web_folder\img"; Flags: ignoreversion
Source: "dist\download_models_to_cache\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\LLM-RAG Suite Launcher"; Filename: "{app}\start.bat"; IconFilename: "{app}\web_folder\img\logo.ico"
Name: "{userdesktop}\LLM-RAG Suite Launcher"; Filename: "{app}\start.bat"; IconFilename: "{app}\web_folder\img\logo.ico"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional tasks:"

[Run]
Filename: "{app}\download_models_to_cache.exe"; WorkingDir: "{app}"; Description: "Download LLM/RAG models"; Flags: nowait postinstall skipifsilent
Filename: "{app}\start.bat"; Description: "Launch LLM-RAG Suite"; Flags: nowait postinstall skipifsilent