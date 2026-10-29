"""Extract trusted SDK archives preserving executable bits and Unix symlinks."""
import argparse
from pathlib import Path
import stat
from zipfile import ZipFile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.archive) as archive:
        for entry in archive.infolist():
            path = root / entry.filename
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Archive path outside destination: {entry.filename}")
            mode = entry.external_attr >> 16
            if stat.S_ISLNK(mode):
                target = archive.read(entry).decode()
                if not (path.parent / target).resolve().is_relative_to(root):
                    raise ValueError(f"Symlink outside destination: {entry.filename}")
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() or path.is_symlink():
                    path.unlink()
                path.symlink_to(target)
            else:
                path = Path(archive.extract(entry, root))
                if mode:
                    path.chmod(mode & 0o777)


if __name__ == "__main__":
    main()
