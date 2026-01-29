; Inno Setup Script for HumanPose3D
; Requires Inno Setup 6.0 or later

#define MyAppName "HumanPose3D"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "HumanPose3D"
#define MyAppURL "https://github.com/JoeyKardolus/humanpose3d_backend"
#define MyAppExeName "humanpose3d-setup-run.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
AppId={{8F2E4A5B-7C3D-4E6F-9A1B-2C3D4E5F6A7B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
; Output settings
OutputDir=..\build\installer
OutputBaseFilename=HumanPose3D-Setup-{#MyAppVersion}
; Compression
Compression=lzma2
SolidCompression=yes
; Windows version requirements
MinVersion=10.0
; Privileges - per-user install by default
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; Wizard style
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The bootstrapper executable
Source: "..\build\pyinstaller\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

; Include the entire source code (needed for uv sync to work)
Source: "..\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "build\*,.venv\*,.git\*,__pycache__\*,*.pyc,*.pyo,.pytest_cache\*,.mypy_cache\*"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Optionally run the app after installation
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up the virtual environment on uninstall
Type: filesandordirs; Name: "{app}\.venv"

[Code]
// Set HUMANPOSE3D_HOME environment variable during install
procedure CurStepChanged(CurStep: TSetupStep);
var
  EnvPath: string;
begin
  if CurStep = ssPostInstall then
  begin
    EnvPath := ExpandConstant('{localappdata}\HumanPose3D');
    RegWriteStringValue(HKEY_CURRENT_USER, 'Environment', 'HUMANPOSE3D_HOME', EnvPath);
    // Notify Windows of the environment change
    // SendMessage is not available, but the change will take effect on next login
  end;
end;

// Remove environment variable on uninstall
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    RegDeleteValue(HKEY_CURRENT_USER, 'Environment', 'HUMANPOSE3D_HOME');
  end;
end;
