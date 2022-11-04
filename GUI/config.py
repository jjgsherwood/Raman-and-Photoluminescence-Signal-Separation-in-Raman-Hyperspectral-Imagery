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
# DEFAULT_DIR = [dir+'\\' for dir in glob.glob(__user_dir+'\\*') if 'doc' in dir.lower()][0]+'data\\'
DEFAULT_DIR = "J:\\Jonne\\Documents\\UvA\\data"

# my save data directory
DEFAULT_SAVE_DIR = DEFAULT_DIR

# this gives an extra dialog to check input variables
SHOW_INPUT = False
