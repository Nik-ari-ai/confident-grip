"""Скачивает одну запись DB2. Сами данные не включаются в репозиторий."""

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile


def main():
    folder = Path("data")
    folder.mkdir(exist_ok=True)
    target = folder / "S1_E2_A1.mat"
    if target.exists():
        print(f"Уже существует: {target}")
        return
    archive = folder / "DB2_s1.zip"
    print("Источник: https://ninapro.hevs.ch/instructions/DB2.html")
    print("Архив участника S1 занимает около 448 МиБ.")
    urlretrieve("https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s1.zip", archive)
    with ZipFile(archive) as source:
        target.write_bytes(source.read("DB2_s1/S1_E2_A1.mat"))
    print(target)


if __name__ == "__main__":
    main()
