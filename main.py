import sys

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

from ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("PyImageEditor")
    app.setApplicationVersion("0.13.0")
    app.setOrganizationName("ImageSuite")

    # Stile moderno nativo della piattaforma
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
