@echo off
REM Clean LaTeX auxiliary files recursively in the ./keynotes/ directory

setlocal enabledelayedexpansion

REM List of extensions to clean
set exts=aux bbl bcf blg log nav out run.xml snm synctex.gz toc

for %%e in (%exts%) do (
    for /r ".\latex" %%f in (*.%%e) do (
        del /q "%%f"
    )
)

echo LaTeX auxiliary files in ./latex/ and subdirectories deleted.