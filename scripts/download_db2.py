"""Скачивает упражнение C выбранного участника DB2."""

import argparse
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=int, choices=range(1, 41), default=1)
    parser.add_argument('--folder', type=Path, default=Path('data'))
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    target = args.folder / f'S{args.subject}_E2_A1.mat'
    if target.exists():
        print(f'Уже существует: {target}')
        return
    name = f'DB2_s{args.subject}'
    archive = args.folder / f'{name}.zip'
    print('Описание и условия: https://ninapro.hevs.ch/instructions/DB2.html')
    print('Архив одного участника занимает около 0,5 ГБ.')
    # Сначала .part: прерванное скачивание не выдаётся за готовый архив.
    if not archive.exists():
        temporary = archive.with_suffix('.zip.part')
        urlretrieve(f'https://ninapro.hevs.ch/files/DB2_Preproc/{name}.zip', temporary)
        temporary.replace(archive)
    with ZipFile(archive) as source:
        temporary = target.with_suffix('.mat.part')
        with source.open(f'{name}/{target.name}') as src, temporary.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        temporary.replace(target)
    print(target)


if __name__ == '__main__':
    main()
