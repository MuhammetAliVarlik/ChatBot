import datetime
from bs4 import BeautifulSoup
import requests

now = datetime.datetime.now()

with  open("datas.txt", "r") as file:
    past_lines = file.readlines()
    if past_lines:
        past = datetime.datetime.strptime(past_lines[0].rstrip("\n"), '%Y-%m-%d %H:%M:%S.%f')


def turkishCh(str):
    turkish_ones = ['Ç', 'ç', 'Ğ', 'ğ', 'ı', 'İ', 'Ö', 'ö', 'Ş', 'ş', 'Ü', 'ü']
    eng_ones = ['C', 'c', 'G', 'g', 'i', 'I', 'O', 'o', 'S', 's', 'U', 'u']
    for specialChar in turkish_ones:
        rep_no = turkish_ones.index(specialChar)
        str = str.replace(specialChar, eng_ones[rep_no])
    return str


class Hava:
    """
    İnput olarak şehir girmemiz yeterli.
    Ntv hava kullanarak bugünkü en yüksek ve en düşük sıcaklıkları tespit ediyoruz
    5 günlük hava durumunu dictionary olarak veriyor

    !! Hava classı Obje tanımlama
    hava_durumu=Hava("izmir")

    !! Şehir ismi
    hava_durumu.sehir.upper()

    !! 5 günlük hava durumu derece
    hava_durumu.weather[2][0]

    !! 5 günlük hava durumu hava olayı
    hava_durumu.weather[2][1]

    !! yarın
    hava_durumu.weather[0][0]

    !! 5 gün sonra
    hava_durumu.weather[5][0]
    """

    def __init__(self, sehir):
        self.weather = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        self.sehir = sehir
        if (now - past).seconds >= 3600 or sehir.split() != past_lines[1].split():
            self.URL = "https://www.ntvhava.com/".format(turkishCh(sehir).lower())
            self.page = requests.get(self.URL)
            self.soup = BeautifulSoup(self.page.content, "html.parser")
            self.h_temp = self.soup.find("div", attrs={"class": "search-content-degree-big"}).text.strip()
            self.l_temp = self.soup.find("div", attrs={"class": "search-content-degree-small"}).text.strip()
            self.text = self.soup.find("div", attrs={"class": "search-content-degree-text"}).text.strip()
            self.five_day = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
            for key in self.five_day:
                self.five_day[key] = ["{}{}".format(self.soup.find_all("div", attrs={
                    "class": "daily-report-tab-content-pane-item-box-bottom-degree-big"})[key].text.strip(),
                                                    self.soup.find_all("div", attrs={
                                                        "class": "daily-report-tab-content-pane-item-box-bottom"
                                                                 "-degree-small"})[
                                                        key].text.strip()),
                                      self.soup.find_all("div",
                                                         attrs={"class": "daily-report-tab-content-pane-item-text"})[
                                          key].text.strip()]
            with  open("datas.txt", "w") as file:
                file.write("{}\n".format(now))
                file.write("{}\n".format(sehir))
                for i in self.five_day:
                    file.write("{}\n".format(self.five_day[i]))
                file.close()
            self.weather = self.five_day
        elif (now - past).seconds < 3600 or sehir.split() == past_lines[1].split():
            self.five_days_nc = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
            for i in self.five_days_nc:
                if i != 0 or i != 1 and i + 2 < len(self.five_days_nc):
                    self.five_days_nc[i] = past_lines[i + 2].replace('[', ' ').replace(']', ' ').replace("'",
                                                                                                         "").replace(
                        ', ', ',').strip().split(',')
            self.weather = self.five_days_nc

# hava_durumu=Hava("izmir")
# print(hava_durumu.weather[2][0])
