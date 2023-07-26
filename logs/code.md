
import pandas as pd
df = pd.read_csv('/kaggle/input/cid-nepal-2076-final/CID_CSV_final/Distribution of number of trafficking cases by modus operandi of trafficking, Nepal.csv')
df.head()

#/kaggle/input/xinjiang-victims

victims_df = pd.read_csv('/kaggle/input/child-victims-by-age/Child Victims by Age.csv')
victims_df.head(), victims_df.shape

#response = requests.get(page)


from bs4 import BeautifulSoup
from urllib.request import urlopen
import urllib

import requests, re, string
import pandas as pd
import csv, time, regex

if 0:      
    # https://stackoverflow.com/questions/71237892/scraping-the-next-page-next-pages-url-staying-on-the-same-page

    dictionary = []

    def get_words_in_page( url ):
        print(url)
        res = urllib.request.urlopen(url)
        print('res',res)
        soup = BeautifulSoup(res, "html")
        lst = ""
        for w in soup.findAll("a",{"href":regex}):
            dictionary.append(w.string)
            lst=w.string
        print(lst)

    #base_url = "https://www.cnrtl.fr/portailindex/LEXI/TLFI/"

    for l in string.ascii_lowercase:    
        base_url = base_url + l.upper()    
        get_words_in_page( base_url )        
        next_index = 0    
        while True:    
            next_index += 80
            url = base_url+"/"+str(next_index)        
            try:
                res = urllib.request.urlopen(url)
            except ValueError:
                break    
            get_words_in_page( url )
        #print()
        
        
if 1:
    page = urlopen( base_url )
    html = page.read().decode("utf-8",'ignore') 
    soup = BeautifulSoup(html, "html.parser") #U+60DC  U+751F U+547D
s=soup.find_all('h1', {'class': 'listing-page-header mb-6 md:mb-8' })
s

# %% [code] {"execution":{"iopub.status.busy":"2023-07-24T21:01:28.583829Z","iopub.execute_input":"2023-07-24T21:01:28.584687Z","iopub.status.idle":"2023-07-24T21:01:29.379424Z","shell.execute_reply.started":"2023-07-24T21:01:28.584606Z","shell.execute_reply":"2023-07-24T21:01:29.377300Z"}}
import requests
from bs4 import BeautifulSoup

base_url = f'https://www.hk01.com/tag/6273'

if 1:
    base_url=f'https://www.hk01.com/search?q=%E5%A2%AE%E6%A8%93'
    page = urlopen( base_url )
    html = page.read().decode("utf-8",'ignore') 
    soup = BeautifulSoup(html, "html.parser") #U+60DC U+751F U+547D

# soup 
#"page":"/v2/search","query":{"q":"墮樓"},"buildId":"LUBhWFdNf-IXL8NT2Xhp4",
#"runtimeConfig":{"ROOT_DOMAIN":"hk01.com",
#"API_BASE_URL":"https://web-data.api.hk01.com/v2/",
#"REMOTE_CONFIG_API_BASE_URL":"https://rcfg-api.hk01.com/v1/",
#"COMMENT_API_BASE_URL_V2":"https://prod-comment-api.dwnews.com/v2/api/",
#"GOOGLE_MAP_API_KEY":"AIzaSyCkLrMHfd8uGQDsq8K_a65oy4WNYtbk79o",
#"PIWIK_TRACKING_URL":"https://track.hk01.com/v1web/piwik.php","PIWIK_SITE_ID":6,"BUGSNAG_API_KEY":"7b604435a5dd27a99f65731feff3d327","HK01_CDN_HOST":"https://cdn.hk01.com","VIDEO_API_BASE_URL":"https://video-api.hk01.com","mediaServerUrl":"https://vdo.cdn.hk01.com","ENABLE_GRIEF_MODE":false,"CONFIG_ENV":"production","WWW_DOMAIN":"www.hk01.com","HK01_SITE_GATEWAY_BASE_URL":"https://hk01-site-gateway.hk01.com/v2","HK01_WEB_STRAPI_BASE_URL":"https://01web-strapi.hk01.com/api","GA_TRACKING_ID":["UA-70981149-1","G-P13VP8RY2F","G-F5LMN5VKW1"],"lruCache":{"enabled":true,"max":50,"maxAge":1800000},"ads":{"enable":true,"tagPrefix":""},

data = {"query":"%E5%A2%AE%E6%A8%93","perPage":10,"page":2} 
r = requests.post(base_url, data = data)
#print(r.json())
r

if 1:
    c="card-main"
    s=soup.find_all('div', {'class':c} ) 
    print( len(s) )
    s
    
c=""
s=soup.find_all('time'  ) 
print( len(s) )
s

#a class=
c="card-title break-words"
s=soup.find_all('a', {'class':c} ) 
print( len(s) )
s

soup.find_all('div','珍') # 珍惜生命'

#re.sub(ur'[\u064B-\u0652\u06D4\u0670\u0674\u06D5-\u06ED]+', ' ', soup)


