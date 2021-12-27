import os

print("please run with root account/请在root账户下执行")

model="""[Desktop Entry]
Name=thisname
Comment=thiscomment
Exec=thiscommand
Terminal=false
MultipleArgs=false
Type=Application
Categories=Application;Development;
StartupNotify=true"""

YN={"y":True,"Y":True,"N":False,"n":False}

def question(tip,choises=None):
    while True:
        print(tip)
        ipt = str(input(">>>"))
        if choises is not None and not ipt in choises:
            print("Wrong input!/输入错误！")
        else:
            return ipt

if not YN[question("are you log in with root account?/是否root账户登录？(y/n)",["Y","N","y","n"])]:
    exit(0)

language = question("choose language/选择语言(1.English/2.简体中文)",['1','2'])
if language == '1':
    commandname  = question("please input your command name")
    command = question("please input your command")
else:
    commandname = question("请输入命令名称")
    command = question("请输入命令")


service_name = commandname + ".desktop"
this_string = model.replace("thisname",commandname).replace("thiscommand",command)
# print(this_string)

service_dir = "/home/" + str(os.popen("echo $USER").read()).replace("\n","") + "/.config/autostart/"
open(service_dir + service_name,"w").write(this_string)

print("over")
