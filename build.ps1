# currently powershell should use 7.2
venv/Scripts/activate.ps1
pip install Nuitka==1.0.7
pip install imageio
pip install orderedset
nuitka --standalone --onefile --enable-plugin=tk-inter --enable-plugin=numpy pyGUS --windows-icon-from-ico=icon.png
# mac
# nuitka3 --standalone --onefile --enable-plugin=tk-inter --enable-plugin=numpy pyGUS
