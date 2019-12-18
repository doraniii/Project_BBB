import urllib.request
from  bs4 import BeautifulSoup
from selenium import webdriver  # 웹 애플리케이션의 테스트를 자동화하기 위한 프레임 워크
from selenium.webdriver.common.keys import Keys
import time # 중간중간 sleep 을 걸어야 해서 time 모듈 import

########################### url 받아오기 ###########################
# 웹브라우져로 크롬을 사용할거라서 크롬 드라이버를 다운받아 위의 위치에 둔다
# 팬텀 js로 하면 백그라운드로 실행할 수 있음
binary = 'c:\chromedriver/chromedriver.exe'
browser = webdriver.Chrome(binary) # 브라우져를 인스턴스화
browser.get("https://www.bing.com/?scope=images&FORM=Z9LH1") # bing(아무것도 검색하지 않을 url)
# 이미지 검색에 해당하는 input 창의 class가 존재(검색창에 해당하는 html코드를 찾아서 elem 사용하도록 설정)
elem = browser.find_element_by_id("sb_form_q") # id 이름으로 검색할때
#elem = browser.find_element_by_xpath("//*[@class='gLFyf gsfi']") 

########################### 검색어 입력 ###########################
# elem 이 input 창과 연결되어 스스로 제주도를 검색
elem.send_keys("신세계 박성웅")
# 웹에서의 submit 은 엔터의 역할을 함
elem.submit()

########################### 반복할 횟수 ###########################
# 스크롤을 내리려면 브라우져 이미지 검색결과 부분(바디부분)에 마우스 클릭 한번 하고 End키를 눌러야함
for i in range(6, 50):
    browser.find_element_by_xpath("//body").send_keys(Keys.END)
    time.sleep(10) # END 키 누르고 내려가는데 시간이 걸려서 sleep 해줌

time.sleep(10) # 네트워크 느릴까봐 안정성 위해 sleep 해줌
html = browser.page_source # 크롬브라우져에서 현재 불러온 소스 가져옴
soup = BeautifulSoup(html, "lxml") # html 코드를 검색할 수 있도록 설정

########################### 그림파일 저장 ###########################
def fetch_list_url():
    params = []
    imgList = soup.find_all("img", class_="mimg") 
    # 네이버 이미지 url 이 있는 img 태그의 _img 클래스에 가서
    for im in imgList:
        try :
            params.append(im["src"]) # params 리스트에 image url 을 담음
        except KeyError:
            params.append(im["data-src"])
    return params

def fetch_detail_url():
    params = fetch_list_url()

    for idx,p in enumerate(params,1):
        # 다운받을 폴더경로 입력
        urllib.request.urlretrieve(p, "K:/deepdata/park" + str(idx) + ".jpg")

if __name__ == '__main__':
    # 메인 실행 함수
    fetch_detail_url()

    # 끝나면 브라우져 닫기
    browser.quit()
