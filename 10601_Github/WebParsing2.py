from bs4 import BeautifulSoup as bs
import numpy as np
from urllib.request import urlopen as uReq
import requests
import re

# list "cars" englobes model year of the car, model name, asking price and mileage

cars = [[],
        [],
        [],
        []]


def webscrapping(num):
    global cars
    myurl = "https://www.cargurus.com/Cars/inventorylisting/viewDetailsFilterViewInventoryListing.action?source" \
             "Context=carGurusHomePageModel&entitySelectingHelper.selectedEntity=m4&zip=19019#resultsPage="+str(num+1)
    print(myurl)
    page_soup = bs(uReq(myurl), features="html.parser")
    # to find model information
    containers = page_soup.findAll("h4", {"cg-dealFinder-result-model"})

    for container in containers:
        # grab year, model, package
        line_ = container.findAll("span", {"itemprop": "name"})
        car_name = str(line_).split(">")[1].strip("</span")
        car_year = car_name.split(" ")[0]
        cars[0].append(int(car_year))
        car_model = car_name.split("Acura ")[1]
        cars[1].append(str(car_model))
        if re.search(string=car_name, pattern="with") is None:
            car_package = "None"
        else:
            car_package = re.split(string=car_name, pattern="with")[1]

    # to find price information
    containers_two = page_soup.findAll("div", {"class": "cg-dealFinder-result-stats"})
    for container_two in containers_two:

        # grab price

        car_priceinfo = container_two.p.span
        car_price = re.split(pattern=">", string=str(car_priceinfo))[1]
        car_price = re.split(pattern="<", string=str(car_price))[0]
        cars[2].append(str(car_price))

        # continue to use containers_two for mileage
        # soup.find('div', {'id': 'first'}).div.a

        # grab mileage
        mileage = str(str(container_two.select("p:nth-of-type(2)")).split("<span>")[1]).split("</span>")[0]
        cars[3].append(str(mileage))


for i in range(20):
    webscrapping(i)
for j in range(len(cars)):
    print(np.unique(cars[j]))