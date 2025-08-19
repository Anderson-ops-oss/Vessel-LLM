[Setup]
AppName=LLM-RAG Suite
AppVersion=1.0
DefaultDirName={autopf}\LLM-RAG Suite
DefaultGroupName=LLM-RAG Suite
UninstallDisplayIcon={app}\web_folder\img\logo.ico
OutputDir=dist
OutputBaseFilename=LLM-RAG-Setup
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
Source: "cache\*"; DestDir: "{app}\cache"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "start.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "web_folder\img\logo.ico"; DestDir: "{app}\web_folder\img"; Flags: ignoreversion
Source: "dist\download_models_to_cache\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\LLM-RAG Suite Launcher"; Filename: "{app}\start.bat"; IconFilename: "{app}\web_folder\img\logo.ico"
Name: "{group}\Uninstall LLM-RAG Suite"; Filename: "{uninstallexe}"; IconFilename: "{app}\web_folder\img\logo.ico"
Name: "{userdesktop}\LLM-RAG Suite Launcher"; Filename: "{app}\start.bat"; IconFilename: "{app}\web_folder\img\logo.ico"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\LLM-RAG Suite"; Filename: "{app}\start.bat"; IconFilename: "{app}\web_folder\img\logo.ico"; Tasks: quicklaunchicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional tasks:"
Name: "quicklaunchicon"; Description: "Create a Quick Launch shortcut"; GroupDescription: "Additional tasks:"; Flags: unchecked

[Run]
Filename: "{app}\download_models_to_cache.exe"; WorkingDir: "{app}"; Description: "Download required LLM/RAG models (REQUIRED before first use)"; Flags: postinstall skipifsilent
