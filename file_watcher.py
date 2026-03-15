import os
import sys
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


def ask_for_folder(prompt, default):
    """Ask for a folder path via Tk dialog, falling back to terminal input."""
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        result = simpledialog.askstring("Folder to monitor", prompt, initialvalue=default)
        root.destroy()
        return result or default
    except Exception:
        value = input(f"{prompt}\n[default: {default}]: ").strip()
        return value or default


def main():
    userdownloadspath = os.path.join(os.path.expanduser("~"), "Downloads")
    userdocumentspath = os.path.join(os.path.expanduser("~"), "Documents")

    SRC = ask_for_folder(
        f"Documents folder to monitor (default: {userdocumentspath})",
        userdocumentspath,
    )
    DST = ask_for_folder(
        f"Downloads folder to monitor (default: {userdownloadspath})",
        userdownloadspath,
    )

    print(f"[watcher] Monitoring:\n  SRC = {SRC}\n  DST = {DST}")
    print(f"[watcher] Python: {sys.executable}")

    class Handler(FileSystemEventHandler):
        def on_created(self, event):
            print(f"[+] New:      {event.src_path}")
        def on_deleted(self, event):
            print(f"[-] Deleted:  {event.src_path}")
        def on_modified(self, event):
            print(f"[~] Modified: {event.src_path}")

    observer_src = Observer()
    observer_src.schedule(Handler(), path=SRC, recursive=False)
    observer_dst = Observer()
    observer_dst.schedule(Handler(), path=DST, recursive=False)
    observer_src.start()
    observer_dst.start()
    print("[watcher] Started. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        observer_src.stop()
        observer_dst.stop()
        observer_src.join()
        observer_dst.join()
        print("[watcher] Stopped.")


if __name__ == "__main__":
    main()