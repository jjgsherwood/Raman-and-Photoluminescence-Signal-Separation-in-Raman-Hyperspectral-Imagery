import os
import glob

__user_dir = os.path.expanduser('~')


"""
change variables below to change config settings.
"""

# Current directory
# DEFAULT_DIR = os.getcwd()
# User directory
# DEFAULT_DIR = __user_dir
# my documents directory
# DEFAULT_DIR = [dir+'/' for dir in glob.glob(__user_dir+'/*') if 'doc' in dir.lower()][0]
# my data directory
DEFAULT_DIR = [dir+'/' for dir in glob.glob(__user_dir+'/*') if 'doc' in dir.lower()][0]+'data/'

# this gives an extra dialog to check input variables
SHOW_INPUT = False

# if True the files are loaded without checking each line of the file for X,Y,wavenumbers
FAST_LOADING = True
