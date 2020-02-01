# PyAi-SelfDrivingCar
Repo for a Plural Sight project for Machine Learning with Pytorch

# Project Dependencies

$ sudo python3 -m ensurepip
$ pip3 --version
$ pip3 install pytest
$ pytest --version
$ pip3 install pylint
$ pip3 install pytest
$ pip3 install kivy
$ pip3 install torch

# PyCharm Interpreter Setup

To fix "unresolved references" errors in individual
python packages, you'll need to right click directories
with module imports, right click, and select
"Mark Directory As... Sources Root"

Inspect the .idea/misc.xml file and confirm that the jdk-name
is "Python 3.7", and not "Python 3.7 (Project Name)".

Inspect the .idea/Project.iml file and confirm there is an
order entry for:
<orderEntry type="jdk" jdkName="Python 3.7" jdkType="Python SDK" />
