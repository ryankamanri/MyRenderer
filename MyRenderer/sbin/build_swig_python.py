###############################################################
# Generate SWIG python module recursively
###############################################################
from __future__ import annotations
import typing
import os

handle_map = {
    0: "Done.",
    1: "An Error occured!",
    255: ""
}

def Execute(exec_str: str) -> int:
    print("\nExecute: " + exec_str + "\n")
    res = os.system(exec_str)
    print("\n", handle_map[res])
    return res


class Path:
    _current_dir: str = None
    def __init__(self, current_dir: str = ".\\") -> None:
        self._current_dir = os.path.abspath(os.path.join(os.getcwd(), current_dir))
    def __str__(self) -> str:
        return self._current_dir
    def MakeDir(self) -> int:
        if(os.path.exists(self._current_dir)):
            print("Path.MakeDir: The directory {} is already exist.".format(self._current_dir))
            return 0
        return Execute("mkdir " + self._current_dir)
    def ChangeDir(self, change_str: str) -> Path:
        self._current_dir = os.path.abspath(os.path.join(self._current_dir, change_str))
        return self
    def Scan(self, filter: typing.Callable[[os.DirEntry], bool]) -> list[Path]:
        res: list[Path] = []
        for item in os.scandir(self._current_dir):
            if(filter(item)):
                res.append(Path(item.path))
        return res
    def MapLocation(self, self_source: Path, target: Path, target_source: Path) -> Path:
        if(os.path.commonpath([str(target), str(target_source)]) != str(target_source)):
            print("Path.MapLocation: The target path must based on target source.")
            return self
        relative_path = ".{}".format(str(target).split(str(target_source))[1])
        return self.ChangeDir(str(self_source)).ChangeDir(relative_path)
    def LastName(self) -> str:
        return self._current_dir.split("\\").pop()
    def Execute(self, exec_str) -> int:
        return Execute("cd {} && {}".format(self._current_dir, exec_str))



class Command:
    _command_str: str = None
    # GAP TYPE
    NO_GAP: str = ""
    BLANK_GAP: str = " "
    COLON_GAP: str = ":"
    def __init__(self, executable: str) -> None:
        self._command_str = executable
    def __str__(self) -> str:
        return self._command_str
    def AddItem(self, item: str) -> Command:
        self._command_str += " \"{}\"".format(item)
        return self
    def AddItems(self, items: list[str]) -> Command:
        for item in items:
            self.AddItem(item)
        return self
    def AddOption(self, option: str, value: str, gap_type: str = BLANK_GAP) -> Command:
        self._command_str += " {}{}\"{}\"".format(option, gap_type, value)
        return self
    def AddOptions(self, option: str, values: list[str], gap_type: str = BLANK_GAP) -> Command:
        for value in values:
            self.AddOption(option, value, gap_type)
        return self


#######################################################################################

# define environment paths

PYTHON_PATH = "C:\\ProgramData\\Anaconda3"
MSVC_PATH = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.30.30705"
WIN_KITS_INCLUDE_PATH = "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.20348.0"
WIN_KITS_LIBS_PATH = "C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\10.0.20348.0"
PROJECT_PATH = "C:\\Users\\97448\\totFolder\\source\\repos\\MyRenderer\\MyRenderer"
PROJECT_LIBS_PATH = PROJECT_PATH + "\\build\\windows-default\\Debug"

INCLUDE_DIR_LIST = [
    PYTHON_PATH + "\\include", 
    MSVC_PATH + "\\include", 
    WIN_KITS_INCLUDE_PATH + "\\um", 
    WIN_KITS_INCLUDE_PATH + "\\ucrt", 
    WIN_KITS_INCLUDE_PATH + "\\shared", 
    PROJECT_PATH
]

LIB_DIR_LIST = [
    PYTHON_PATH + "\\libs", 
    MSVC_PATH + "\\lib\\x64", 
    WIN_KITS_LIBS_PATH + "\\um\\x64", 
    WIN_KITS_LIBS_PATH + "\\ucrt\\x64", 
    PROJECT_LIBS_PATH
]

LIB_LIST = [
    "kamanri.lib", 
    "gdi32.lib", 
    "user32.lib"

]


COMPILE_MULTITHREAD_LIB_OPTION = "/MD"

# select a default single/multi thread library by COMPILE_MULTITHREAD_LIB_OPTION, and annotate it.
# reference https://blog.csdn.net/jianchiweiyi1/article/details/25715135?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-25715135-blog-119639790.235%5Ev32%5Epc_relevant_increate_t0_download_v2_base&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-25715135-blog-119639790.235%5Ev32%5Epc_relevant_increate_t0_download_v2_base&utm_relevant_index=1
NO_DEFAULT_MULTITHREAD_LIB_LIST = [
    "libc.lib", # release single
    "libcmt.lib", # release multiple        /MT
    # "msvcrt.lib", # release dll multiple    /MD
    "libcd.lib", # debug single
    "libcmtd.lib", # debug multiple         /MTd
    "msvcrtd.lib" # debug dll multiple    /MDd
]

#######################################################################################

# define prefixes
PRIVATE_PREFIX = "_"
# define suffixes
NONE_SUFFIX = ""
I_SUFFIX = ".i"
CXX_SUFFIX = ".cxx"
OBJ_SUFFIX = ".obj"
PYD_SUFFIX = ".pyd"
EXP_SUFFIX = ".exp"
LIB_SUFFIX = ".lib"

def GetModuleName(swig_source_file: Path) -> str:
    with open(str(swig_source_file)) as file:
        while(True):
            line = file.readline()
            if(line == ""):
                raise NameError(f"Module name not found in {str(swig_source_file)}")
            if(line.startswith("%module")):
                return line.split(" ")[1].split("\n")[0]
        

def Build(project_path: Path, swig_source_file: Path, target_path: Path) -> None:
    
    source_name = GetModuleName(swig_source_file)
    cxx_name = PRIVATE_PREFIX + source_name + CXX_SUFFIX
    obj_name = PRIVATE_PREFIX + source_name + OBJ_SUFFIX
    pyd_name = PRIVATE_PREFIX + source_name + PYD_SUFFIX
    exp_name = PRIVATE_PREFIX + source_name + EXP_SUFFIX
    lib_name = PRIVATE_PREFIX + source_name + LIB_SUFFIX

    res = project_path.Execute(
        Command("swig") \
            .AddItem("-python") \
            .AddItem("-c++") \
            .AddItem("-includeall") \
            .AddOptions("-I", INCLUDE_DIR_LIST, Command.NO_GAP) \
            .AddOption("-o", "{}\\{}".format(str(target_path), cxx_name)) \
            .AddItem(str(swig_source_file))
    )

    if(res != 0): exit(res)

    res = target_path.Execute(
        Command("cl") \
        .AddOption("/c", cxx_name) \
        .AddOption("/std", "c++17", Command.COLON_GAP) \
        .AddOption("/wd", "4819") \
        .AddOptions("/I", INCLUDE_DIR_LIST) \
        .AddItem(COMPILE_MULTITHREAD_LIB_OPTION) \
        .AddItem("/nologo")
    )

    if(res != 0): exit(res)

    res = target_path.Execute(
        Command("link") \
            .AddItem(obj_name) \
            .AddOption("/OUT", pyd_name, Command.COLON_GAP) \
            .AddItem("/DLL") \
            .AddOptions("/LIBPATH", LIB_DIR_LIST, Command.COLON_GAP) \
            .AddOptions("/NODEFAULTLIB", NO_DEFAULT_MULTITHREAD_LIB_LIST, Command.COLON_GAP) \
            .AddItems(LIB_LIST) \
            .AddItem("/nologo")
    )

    if(res != 0): exit(res)

    # delete middle generated files.
    target_path.Execute(Command("rm").AddItem(cxx_name).AddItem(exp_name).AddItem(lib_name).AddItem(obj_name))

###################################################################################

# define static path
project_path = Path(PROJECT_PATH)
source_path = Path(PROJECT_PATH + "\\kamanri")
target_path = Path(PROJECT_PATH + "\\build\\kamanri")
target_path.MakeDir()

# define moveable path
m_source_path = Path(str(source_path))
m_target_path = Path(str(target_path))


# define path stack
path_stack: list[Path] = []

# append root directory
path_stack.append(m_source_path)

# recursive scan
while(len(path_stack) != 0):
    m_source_path = path_stack.pop()

    m_target_path.MapLocation(target_path, m_source_path, source_path)

    swig_source_files = m_source_path.Scan(lambda sub: sub.is_file() and str(sub.name).endswith(I_SUFFIX))

    if(len(swig_source_files) > 0):
        m_target_path.MakeDir()
        for swig_source_file in swig_source_files:
            print("Build for " + str(swig_source_file))
            Build(project_path, swig_source_file, m_target_path)


    for sub_dir in m_source_path.Scan(lambda sub: sub.is_dir()):
        path_stack.append(sub_dir)
