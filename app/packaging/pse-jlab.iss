#define AppName "PSE J-Lab"
[Setup]
AppId={{b03cff76-d258-42bf-b241-b94782759828}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Andrej Ristic
AppPublisherURL=https://github.com/andrejristic/pse-j-lab
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile={#SourceDir}\LICENSE.txt
OutputDir={#OutDir}
OutputBaseFilename=PSEJLab-Setup-{#AppVersion}
SetupIconFile={#SourceDir}\pse-jlab.ico
UninstallDisplayIcon={app}\pse-jlab.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\pse-jlab.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\pse-jlab.exe"; Tasks: desktopicon
[Run]
Filename: "{app}\pse-jlab.exe"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
