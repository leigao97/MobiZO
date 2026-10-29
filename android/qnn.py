"""Training exporter entry point: python -m android.qnn --help."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from android.export_training import main

if __name__ == "__main__":
    main()
