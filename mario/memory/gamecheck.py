"""Is Sean playing? hoi4.exe, factorio.exe, or a Java process running Minecraft (ATLauncher's own idle launcher window does not count;
its javaw.exe has ATLauncher.exe on the classpath). Prints GAME or NONE."""
import subprocess, sys
def gaming():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(hoi4|factorio|javaw|java)\.exe$' } | ForEach-Object { $_.Name + ' ' + $_.CommandLine }"],
                             capture_output=True, text=True, timeout=60).stdout.lower()
    except Exception:
        return False
    for line in out.splitlines():
        if line.startswith(("hoi4", "factorio")):
            return True
        if ("minecraft" in line or "--gamedir" in line or "net.fabricmc" in line or "cpw.mods" in line) and "atlauncher.exe;" not in line:
            return True
    return False
if __name__ == "__main__":
    print("GAME" if gaming() else "NONE")
