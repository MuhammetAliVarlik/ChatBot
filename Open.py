import os


class Open:
    """
    open_smthg=Open()

    !! Uygulamarı listeler
    open_smthg.help()

    !! Girilen uygulamayı çalıştırır
    open_smthg.start("Spotify")
    """

    def __init__(self):
        self.app_list = os.listdir("C:/Users/Administrator/Desktop")

    def start(self, app_name):
        app_name = "{}.lnk".format(app_name)
        self.app_index = self.app_list.index(app_name)
        os.startfile("C:/Users/Administrator/Desktop/{}".format(self.app_list[self.app_index]))

    def help(self):
        a = 0
        print("Açabileceğim uygulamalar:")
        print(self.app_list)
        for i in self.app_list:
            a += 1
            print(a, ")", i.strip(".lnk"))


"""open_smthg = Open()
open_smthg.help()
open_smthg.start("Spotify")"""
