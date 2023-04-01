# currently powershell should use 7.2
# pyinstaller pyGUS/__main__.py -c -F -n pygus1.3 -i icon.png
&deactivate
# rm -r -Force build_venv
python3 -m venv build_venv
build_venv/Scripts/activate.ps1
# Some of these packages are only used for build, so they are not listed in
# requirement.txt
# higher version of niutka failed on multiprocessing
pip install Nuitka==1.0.7
# pip install imageio==2.26.0
pip install orderedset
pip install coloredlogs==15.0.1
pip install matplotlib==3.5.2
# old version of scipy require old numpy
pip install numpy==1.22.4
pip install opencv-contrib-python==4.6.0.66
# in windows 1.7.2 works
pip install scipy==1.7.2
pip install tqdm==4.64.1
nuitka --standalone --onefile --enable-plugin=tk-inter pyGUS --enable-plugin=numpy --windows-icon-from-ico=icon.ico
# mac
# nuitka3 --standalone --onefile --enable-plugin=tk-inter --enable-plugin=numpy pyGUS
