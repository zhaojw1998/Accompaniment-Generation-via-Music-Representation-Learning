import time
from selenium import webdriver
from selenium.common.exceptions import UnexpectedAlertPresentException
from selenium.webdriver.common.by import By
import os
import shutil
import numpy as np
from tqdm import tqdm
import sys

def get_link():
    xml_list = []
    midi_list = []
    sole_xml_list = []
    cannotLoad = 0
    for i in tqdm(range(263, 1169+1, 1)):
        url = "https://www.musicalion.com/en/scores/notes/composition/detailed-search-composition?metaarrangementid=67&ppage=" + str(i)
        driver = webdriver.Chrome()
        driver.get(url)
        #driver.implicitly_wait(2)
        #time.sleep(2)
        for link in driver.find_elements(By.CLASS_NAME, 'cm')[:5]:
            new_url = link.get_attribute('href')
            new_driver = webdriver.Chrome()
            new_driver.get(new_url)
            #new_driver.implicitly_wait(2)
            #time.sleep(2)
            xml_link = None
            midi_link = None
            try:
                for new_link in new_driver.find_elements(By.CLASS_NAME, 'cpDownloadFile'):
                    if '.mid' in str(new_link.get_attribute('href')):
                        midi_link = new_link.get_attribute('href')
                    if '.xml' in str(new_link.get_attribute('href')):
                        xml_link = new_link.get_attribute('href')
                if (xml_link != None and midi_link != None):
                    print('save midi and xml')
                    xml_list.append(xml_link+'\n')
                    midi_list.append(midi_link+'\n')
                elif (xml_link != None and midi_link == None):
                    print('save xml')
                    sole_xml_list.append(xml_link)
            except : 
                print("can't load")
                cannotLoad += 1
                pass
            new_driver.quit()
        driver.quit()
        with open('D:\\Computer Music Research\\score scrape and analysis\\scrape_musicalion\\scrape_list\\xml_test.txt', 'w') as f:
            f.writelines(xml_list)
        with open('D:\\Computer Music Research\\score scrape and analysis\\scrape_musicalion\\scrape_list\\midi_test.txt', 'w') as f:
            f.writelines(midi_list)
        with open('D:\\Computer Music Research\\score scrape and analysis\\scrape_musicalion\\scrape_list\\sole_xml_test.txt', 'w') as f:
            f.writelines(sole_xml_list)
    print('cannot load:', cannotLoad)

def get_file():
    url = "https://www.musicalion.com/en/"
    download_root = 'D:\\Computer Music Research\\score scrape and analysis\\scrape_musicalion\\download'
    xml_root = 'D:\\Computer Music Research\\score scrape and analysis\\scrape_musicalion\\musicalion_xml'
    midi_root = 'D:\\Computer Music Research\\score scrape and analysis\\scrape_musicalion\\musicalion_midi'
    options = webdriver.ChromeOptions()
    prefs = {'download.default_directory': download_root, 'download.prompt_for_download': False, 'download.extensions_to_open': 'xml', 'safebrowsing.enabled': True}
    options.add_experimental_option('prefs', prefs)
    options.add_argument("start-maximized")
    options.add_argument("--disable-extensions")
    options.add_argument("--safebrowsing-disable-download-protection")
    options.add_argument("safebrowsing-disable-extension-blacklist")
    driver = webdriver.Chrome(chrome_options=options)
    driver.get(url)
    driver.implicitly_wait(10)
    time.sleep(5)
    login_name = driver.find_element_by_name("login[userIdentifier]")
    #print(login_name)
    #login_name = driver.find_element_by_id("f__globalLoginForm__userIdentifier")
    login_name.send_keys("musicx")
    time.sleep(3)
    password = driver.find_element_by_name("login[password]")
    #password = driver.find_element_by_id("f__globalLoginForm__password")
    password.send_keys("Musicx710")
    time.sleep(3)
    login_button = driver.find_element_by_id("f__globalLoginForm__submitbtn")
    login_button.click()
    time.sleep(5)

    with open('D:\\Computer Music Research\\score scrape and analysis\\scrape_musicalion\\scrape_list\\xml_test.txt') as f:
        musicalion_xml_list = f.readlines()
    with open('D:\\Computer Music Research\\score scrape and analysis\\scrape_musicalion\\scrape_list\\midi_test.txt') as f:
        musicalion_midi_list = f.readlines()
    
    print('begin scrape xml from musicalion')
    for i in tqdm(range(len(os.listdir(xml_root)), len(musicalion_xml_list), 1)):
        newwindow = 'window.open("' + musicalion_xml_list[i].strip('\n') +'")'
        driver.execute_script(newwindow)
        time.sleep(30+ 30 * np.random.rand())
        while True:
            try:
                file_name = os.path.join(download_root, os.listdir(download_root)[0])
                new_name = os.path.join(download_root, str(i).zfill(5) + '.xml')
                os.rename( file_name, new_name)
                break
            except PermissionError:
                time.sleep(.5)
            except IndexError:
                print('get caught and restricted. Today we end at index', i)
                sys.exit()
        shutil.move(new_name, xml_root)
    
    print('begin scrape midi from musicalion')
    for i in tqdm(range(len(os.listdir(xml_root)), len(musicalion_midi_list), 1)):
        newwindow = 'window.open("' + musicalion_midi_list[i].strip('\n') +'")'
        driver.execute_script(newwindow)
        time.sleep(30 + 30 * np.random.rand())
        while True:
            try:
                file_name = os.path.join(download_root, os.listdir(download_root)[0])
                new_name = os.path.join(download_root, str(i).zfill(5) + '.mid')
                os.rename( file_name, new_name)
                break
            except PermissionError:
                time.sleep(.5)
            except IndexError:
                print('get caught and restricted. Today we end at index', i)
                sys.exit()
        shutil.move(new_name, midi_root)

    driver.close()

if __name__ == '__main__':
    get_file()