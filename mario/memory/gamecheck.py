"""Is Sean playing? Any game in P:/agent/games.json (shared with the game watcher and the PC lights, 2026-11-08),
or a Java process running Minecraft (ATLauncher's own idle launcher window does not count; its javaw.exe has
ATLauncher.exe on the classpath). Falls back to hoi4/factorio/Minecraft if the file can't be read. Prints GAME or NONE."""
import json, subprocess, sys
FALLBACK = {"games": ["hoi4.exe", "factorio.exe"], "minecraft_markers": ["minecraft", "--gamedir", "net.fabricmc", "cpw.mods"],
            "atlauncher_marker": "atlauncher.exe;"}
def game_list():
    try:
        with open("P:/agent/games.json", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return FALLBACK
def gaming():
    g = game_list()
    games = {x.lower() for x in g["games"]}
    marks, atl = g.get("minecraft_markers", []), g.get("atlauncher_marker") or "\0"
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_Process | ForEach-Object { if ($_.Name -match '^javaw?[.]exe$') { $_.Name + ' ' + $_.CommandLine } else { $_.Name } }"],
                             capture_output=True, text=True, timeout=60).stdout.lower()
    except Exception:
        return False
    for line in out.splitlines():
        line = line.strip()
        if line in games:
            return True
        if line.startswith(("javaw.exe ", "java.exe ")) and any(m in line for m in marks) and atl not in line:
            return True
    return False
if __name__ == "__main__":
    print("GAME" if gaming() else "NONE")
