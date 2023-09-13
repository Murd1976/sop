import requests
from bs4 import BeautifulSoup
import urllib.parse
import re

visited = set()

def get_domain(url):
    parsed_url = urllib.parse.urlparse(url)
    domain = "{uri.scheme}://{uri.netloc}/".format(uri=parsed_url)
    return domain

def get_links(url, original_domain):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    for link in soup.findAll('a', href=True):
        link = urllib.parse.urljoin(url, link.get('href'))
        
        if link not in visited and get_domain(link) == original_domain:
            visited.add(link)
            yield link

def scrape_site(start_url, output_file):
    original_domain = get_domain(start_url)
    urls_to_scrape = [start_url]
    info_list = []
    count = 0
    with open(output_file, 'w', encoding='utf-8') as file:
        while (urls_to_scrape and count < 200):
            url = urls_to_scrape.pop(0)
            #print(f'\n{url}\n')
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'lxml')
            text = soup.get_text()
            
            # Разделяем текст по символам перевода строки
            lines = text.split('\n')
            
            # Удаляем пробельные символы в начале и конце каждой строки
            # и пропускаем пустые строки
            text = '\n'.join(line.strip() for line in lines if line.strip() != "")
            count += 1
            if(text != ""):
                print(f'\n{text}\n')
                info_list.append(text)
                file.write(text)
                file.write('\n')
            urls_to_scrape.extend(get_links(url, original_domain))
            
    print(f"\n {visited} \n")
    for ur in urls_to_scrape:
        print(ur)
    '''
    f = open(output_file,'r').read()
    total = f.split('\n')
    f = f.split('\n')
    f = set(f)
    open('short_' + output_file, 'w').write('\n'.join(f))
    print(f)
    print(len(total) - len(f),'всего удалённых строк')
    '''
    return info_list
#start_url = 'http://your.website.com'
#output_file = 'output.txt'
#scrape_site(start_url, output_file)
