# currently powershell should use 7.2
# pyinstaller pyGUS/__main__.py -c -F -n pygus1.3 -i icon.png
venv/Scripts/activate.ps1
pip install Nuitka==1.0.7
pip install imageio
pip install orderedset
pip install coloredlogs==15.0.1
pip install matplotlib==3.5.2
pip install numpy==1.22.4
pip install opencv-contrib-python==4.5.5.64
# in windows 1.7.2 works
pip install scipy==1.7.2
pip install tqdm==4.64.1
nuitka --standalone --onefile --enable-plugin=tk-inter --enable-plugin=numpy pyGUS --windows-icon-from-ico=icon.png
# mac
# nuitka3 --standalone --onefile --enable-plugin=tk-inter --enable-plugin=numpy pyGUS
