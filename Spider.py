import requests
import re
from bs4 import BeautifulSoup
import time  
import urllib
import xlsxwriter as xw

class CropContentSpider:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
        }
        self.cookies = {
            "ttcid": "cda563bfab9144f59a159ee97a2059cc59",
            "MONITOR_WEB_ID": "c5ede508-62ef-4a2d-ab00-fdefea7ec1de",
            "tt_webid": "6927098241788904971",
            "tt_scid": "fxILdpczLbQZfvUrSseKHCxqPgwMa2AwEKbp7bDKDOI3qzaAduVbag6CdHo4Vfl97132"
        }
        self.start_url = "http://bcch.ahnw.cn/"
        self.num = 3523
        # http://bcch.ahnw.cn/CropContent.aspx?id=

    def getHTML(self, url):
        resp = requests.get(url, headers=self.headers, cookies=self.cookies, timeout=3)
        print(url)
        return resp.content.decode("gbk",errors='ignore') 

    # def getImage(self, html, num, entity):
    #     req = re.compile(r'(.*?)', re.S)         # image list
    #     image_list = re.findall(req, html)
    #     if len(image_list) == 0:
    #         f = open("error.txt", "a")
    #         f.write(num + " " + entity + "\n")
    #         f.close()
    #     else:
    #         image_url = "http://" + image_list[0]
    #         print(image_url)
    #         urllib.request.urlretrieve(image_url, "image\\%s.png" % num)

    def getTextInfo(self):
        data_list=[]
        for i in range(1, 10):
            url = self.start_url + "CropContent.aspx?id=" + str(i)
            html = self.getHTML(url)
            soup = BeautifulSoup(html, "html.parser") 
            data_item_nameZHCN = soup.find(attrs={"id": "lblNameZHCN"})
            nameZHCN = data_item_nameZHCN.get_text()  
            # print(nameZHCN)
            data_item_nameEng = soup.find(attrs={"id": "lblNameEng"})
            nameEng = data_item_nameEng.get_text()  
            # print(nameEng)
            data_item_introduction = soup.find(attrs={"id": "lblIntroduction"})
            introduction = data_item_introduction.get_text()  
            # print(introduction)
            data_item_damageSym = soup.find(attrs={"id": "lblDamageSym"})
            damageSym = data_item_damageSym.get_text()
            damageSym = damageSym.replace("[为害症状]", "")
            # print(damageSym)
            data_item_oFactor = soup.find(attrs={"id": "lblOFactor"})
            oFactor = data_item_oFactor.get_text()  
            oFactor = oFactor.replace("[发生规律]", "") 
            # print(oFactor)
            data_item_cMethod = soup.find(attrs={"id": "lblCMethod"})
            cMethod = data_item_cMethod.get_text()  
            cMethod = cMethod.replace("[防治]", "") 
            # print(cMethod)
            row=[nameZHCN,nameEng,introduction,damageSym,oFactor,cMethod]
            data_list.append(row)
            if i%100==0:
                time.sleep(30)  
        workbook = xw.Workbook('data0.xls')
        worksheet1 = workbook.add_worksheet("sheet1")
        worksheet1.activate()
        title = ['名称','英文名','简介','为害症状','发生规律','防治']
        worksheet1.write_row('A1',title)
        i = 2
        for j in range(len(data_list)):
            insertData = [data_list[j][0],data_list[j][1],data_list[j][2],data_list[j][3],data_list[j][4],data_list[j][5]]
            row = 'A' + str(i)
            worksheet1.write_row(row, insertData)
            i += 1
        workbook.close() 

if __name__ == '__main__':
    spider = CropContentSpider()
    spider.getTextInfo()
