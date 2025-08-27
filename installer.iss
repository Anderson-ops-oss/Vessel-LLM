[Setup]
AppName=OOCL Vessel LLM
AppVersion=1.0
DefaultDirName={autopf}\OOCL Vessel LLM
DefaultGroupName=OOCL Vessel LLM
UninstallDisplayIcon={app}\web_folder\img\logo.ico
OutputDir=dist
OutputBaseFilename=OOCL Vessel LLM-Setup
Compression=lzma
SolidCompression=yes
; Allow users to choose installation directory
DisableDirPage=no
; Show welcome page
DisableWelcomePage=no
; Allow users to choose program group
DisableProgramGroupPage=no
; Set installation wizard appearance
WizardStyle=modern
; Set permissions - allow normal users to install
PrivilegesRequired=lowest
; Allow installation on 64-bit systems
ArchitecturesAllowed=x64compatible
; Install in 64-bit mode only on 64-bit systems
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "dist\server\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "web_folder\*"; DestDir: "{app}\web_folder"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "rag_models\*"; DestDir: "{app}\rag_models"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "start.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "web_folder\img\logo.ico"; DestDir: "{app}\web_folder\img"; Flags: ignoreversion
Source: "dist\download_models_to_cache\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "LM-Studio-0.3.23-3-x64.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\OOCL Vessel LLM Launcher"; Filename: "{app}\start.bat"; IconFilename: "{app}\web_folder\img\logo.ico"
Name: "{group}\Uninstall OOCL Vessel LLM"; Filename: "{uninstallexe}"; IconFilename: "{app}\web_folder\img\logo.ico"
Name: "{userdesktop}\OOCL Vessel LLM Launcher"; Filename: "{app}\start.bat"; IconFilename: "{app}\web_folder\img\logo.ico"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\OOCL Vessel LLM"; Filename: "{app}\start.bat"; IconFilename: "{app}\web_folder\img\logo.ico"; Tasks: quicklaunchicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional tasks:"
Name: "quicklaunchicon"; Description: "Create a Quick Launch shortcut"; GroupDescription: "Additional tasks:"; Flags: unchecked
Name: "installlmstudio"; Description: "Install LM Studio (Local LLM Server - Recommended)"; GroupDescription: "Additional components:"

[Run]
Filename: "{app}\LM-Studio-0.3.23-3-x64.exe"; WorkingDir: "{app}"; Description: "Install LM Studio (Local LLM Server)"; Flags: postinstall skipifsilent; Tasks: installlmstudio
Filename: "{app}\download_models_to_cache.exe"; WorkingDir: "{app}"; Description: "Download required LLM/RAG models (REQUIRED before first use)"; Flags: postinstall skipifsilent
