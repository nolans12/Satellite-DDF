import importlib

def check_poliastro_installed():
    try:
        importlib.import_module('poliastro')
        print("Poliastro is installed.")
    except ImportError:
        print("Poliastro is not installed.")

if __name__ == "__main__":
    check_poliastro_installed()
