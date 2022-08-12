venv/Scripts/activate.ps1
pip install Nuitka==0.9.6
pip install orderedset
nuitka --standalone --onefile --enable-plugin=tk-inter --enable-plugin=numpy pyGUS
# mac
# nuitka3 --standalone --onefile --enable-plugin=tk-inter --enable-plugin=numpy pyGUS
