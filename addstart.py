import os

print("please run with root account/请在root账户下执行")

model="""[Unit]
Description=test
Requires=network-online.target
After=network-online.target

[Service]
Type=forking
ExecStart=/bin/bash 脚本地址

[Install]
WantedBy=multi-user.target"""

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
    net_need = YN[question("network-needed?(y/n)",["Y","N","y","n"])]
    address  = question("please input your code absolute address")
else:
    net_need = YN[question("该服务是否需要联网?(y/n)", ["Y", "N", "y", "n"])]
    address = question("请输入服务代码的绝对地址")

service_name = address.split("/")
service_name.reverse()
service_name = service_name[0].split(".")[0] + ".service"

string = model.replace("脚本地址",address)
if not net_need:
    string = string.replace("Requires=network-online.target\nAfter=network-online.target\n","")

service_dir = "/lib/systemd/system/"
open(service_dir + service_name,"w").write(string)

print("processing...")
os.system("systemctl daemon-reload")
os.system("chmod +x " + address)
os.system("systemctl enable " + service_name)

if language=='1':
    print("input 'systemctl status " + service_name + "' to check service status")
else:
    print("输入'systemctl status " + service_name + "'查看服务状态")
