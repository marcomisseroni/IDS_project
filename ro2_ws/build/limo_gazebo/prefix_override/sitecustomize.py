import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/marco/marco/uni/magistrale/IDS_project/ro2_ws/install/limo_gazebo'
