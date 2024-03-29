# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import os
import time
import win32gui
import math
from kamanri.Utils import Log
from kamanri.Renderer import ObjModel
from kamanri.renderer.World import World3D, Camera, BlinnPhongReflectionModel, DefaultResult
import kamanri.renderer.world.BlinnPhongReflectionModelE as BlinnPhongReflectionModelE
from kamanri.Maths import ElementList, Vector, SMatrix
import kamanri.utils.LogE as LogE
from kamanri.Windows import WinGDI_Window, IntToPtr
from kamanri.windows.WinGDI_WindowE import Painter, PainterFactor
from kamanri.window_procedures.WinGDI_Window import UpdateProcedure, MakeUpdateProcedure, UpdateFuncBaseWrapper, DefaultResult

WINDOW_LENGTH: int = 600
PI = math.pi
IS_USE_CUDA = False

BASE_PATH = "C:/Users/97448/totFolder/source/repos/MyRenderer/MyRenderer/models/"
DIABLO3_POSE_OBJ = BASE_PATH + "diablo3_pose/diablo3_pose.obj"
DIABLO3_POSE_TGA = BASE_PATH + "diablo3_pose/diablo3_pose_diffuse.tga"
FLOOR_OBJ = BASE_PATH + "floor/floor.obj"
FLOOR_TGA = BASE_PATH + "floor/floor_diffuse.tga"

class UpdateFuncWrapper(UpdateFuncBaseWrapper):
    def __init__(self):
        UpdateFuncBaseWrapper.__init__(self)
        pass
        
    def UpdateFunc(self, world: World3D):
        print("called Update Func!")
        # time.sleep(100)
        camera: Camera = world.GetCamera()
        print(world)
        print(camera)
        camera.Transform()
        
        
        world.Build()
        print(world)
        return 0
    

# g_world: World3D = 0


def WindowProc(h_wnd: int, u_msg: int, w_param: float, l_param: float) -> int:
    print(h_wnd, u_msg, w_param, l_param)
    if(u_msg == 0x0010 or u_msg == 0x0002): # WM_CLOSE
        win32gui.DestroyWindow(h_wnd)
        win32gui.PostQuitMessage(0)
    return win32gui.DefWindowProc(h_wnd, u_msg, w_param, l_param)

def StartRender() -> None:
    p_list = BlinnPhongReflectionModelE.PointLightList([
            BlinnPhongReflectionModelE.PointLight(
                Vector(ElementList([0, 3, 4, 1])), 
                800, 0xffffff
            )])

    bpr_model = BlinnPhongReflectionModel(
            p_list, WINDOW_LENGTH, WINDOW_LENGTH, 0.95, 1 / PI * 2, 0.2, IS_USE_CUDA
        )
    camera = Camera(
                Vector(ElementList([0, -1, 5, 1])), 
                Vector(ElementList([0, 0, -1, 0])), 
                Vector(ElementList([0, 1, 0, 0])), 
                -1, 
                -5, 
                WINDOW_LENGTH, 
                WINDOW_LENGTH
            )
    world = World3D(
        camera, bpr_model, IS_USE_CUDA
    )

    world.AddObjModel(
        ObjModel(
            DIABLO3_POSE_OBJ, 
            DIABLO3_POSE_TGA), 
        SMatrix(ElementList([
                2, 0, 0, 0,
			    0, 2, 0, 0,
			    0, 0, 2, 1,
			    0, 0, 0, 1
        ]))
    )
    world.AddObjModel(
        ObjModel(
            FLOOR_OBJ, 
            FLOOR_TGA), 
        SMatrix(ElementList([
                2, 0, 0, 0,
			    0, 2, 0, 0,
			    0, 0, 2, 0,
			    0, 0, 0, 1
        ]))
    )
    def_res = DefaultResult()
    world.Commit()
    world.Build()

    # global g_world 
    # g_world = world

    # create a main window

    w_class_name = "ZWX"
    w_class = win32gui.WNDCLASS()
    w_class.hbrBackground = win32gui.CreateSolidBrush(0x0)
    w_class.lpfnWndProc = WindowProc
    w_class.lpszClassName = w_class_name
    w_class.style = 0x0003


    win32gui.RegisterClass(w_class)
    handle = win32gui.CreateWindow(w_class_name, "Renderer", 0x00CA0000, 0, 0, 0, 0, 0x0, 0x0, 0x0, None)
    win32gui.ShowWindow(handle, 5)

    #

    wingdi_window = WinGDI_Window(IntToPtr(handle), world, WINDOW_LENGTH, WINDOW_LENGTH)

    update_func_wrapper = UpdateFuncWrapper()
    update_procedure = MakeUpdateProcedure(update_func_wrapper, WINDOW_LENGTH, WINDOW_LENGTH)

    wingdi_window.AddProcedure(update_procedure)
    wingdi_window.Show()
    wingdi_window.MessageLoop()

    while(win32gui.GetMessage(handle, 0, 0)):
        pass

    print("Finished.")
    win32gui.DestroyWindow(handle)
    
    pass

    



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    
    Log.SetLevel(LogE.DEBUG_LEVEL)
    print(f"May you have a nice day!")
    StartRender()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
