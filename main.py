from chatbot import ChatBot
from FaceDetection import FaceDetection
from HandDetection import *
from PoseDetection import *
from Open import *
from Weather import *
from multiprocessing import Process


class Vikipedi:
    def __init__(self, aranan):
        self.URL = "https://tr.wikipedia.org/wiki/{}".format(turkishCh(aranan).lower())
        self.page = requests.get(self.URL)
        self.soup = BeautifulSoup(self.page.content, "html.parser")
        # self.h_temp = self.soup.find("div", attrs={"class": "search-content-degree-big"}).text.strip()


class Responses(FaceDetection, ChatBot):

    def giveResponse(self):
        while True:
            sehir = "ankara"
            FaceDetection.findFace(self, opencam=1)
            if self.results.detections:
                tag, response, inp = ChatBot().chat()
                if tag == "selamlama2":
                    if 'günler' in inp.split():
                        response = response.replace("+", "günler")
                    elif 'akşamlar' in inp.split():
                        response = response.replace("+", "akşamlar")
                    elif 'geceler' in inp.split():
                        response = response.replace("+", "geceler")
                    elif 'günaydın' in inp.split():
                        response = response.replace("+", "günler")
                    elif 'tünaydın' in inp.split():
                        response = response.replace("+", "öğlenler")
                    else:
                        response = response.replace("+", "günler")
                if tag == "hava durumu1":
                    if 'bugün' in inp.split() or 'bugünün' in inp.split() or 'bugünkü' in inp.split():
                        response = response.replace("-", "bugün")
                        hava_durumu = Hava(sehir)
                        hava_durumu.sehir.upper()
                        response = response.replace("+", hava_durumu.weather[0][0].split()[0])
                        response = response.replace("*", hava_durumu.weather[0][0].split()[2])
                        response = response.replace("?", hava_durumu.weather[0][1])
                    elif 'yarın' in inp.split() or 'yarının' in inp.split() or 'yarınki' in inp.split():
                        print("nope")
                        response = response.replace("-", "yarın")
                        hava_durumu = Hava(sehir)
                        hava_durumu.sehir.upper()
                        response = response.replace("+", hava_durumu.weather[1][0].split()[0])
                        response = response.replace("*", hava_durumu.weather[1][0].split()[2])
                        response = response.replace("?", hava_durumu.weather[1][1])
                    else:
                        response = response.replace("-", "bugün")
                        hava_durumu = Hava(sehir)
                        hava_durumu.sehir.upper()
                        response = response.replace("+", hava_durumu.weather[0][0].split()[0])
                        response = response.replace("*", hava_durumu.weather[0][0].split()[2])
                        response = response.replace("?", hava_durumu.weather[0][1])
                print(response)


if past < now:
    time_delta = now - past

Responses().giveResponse()
