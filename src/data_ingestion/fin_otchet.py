import requests
import os
from time import sleep

def download_reports():
    companies = {
    "MSNG": "+МосЭнерго",
    "DATA": "iАренадата",
    "WUSH": "iВУШХолднг",
    "DIAS": "iДиасофт",
    "IVAT": "iИВА",
    "DELI": "iКаршеринг",
    "POSI": "iПозитив",
    "SOFL": "iСофтлайн",
    "MDMG": "MDMG-ао",
    "AKRN": "Акрон",
    "ALRS": "АЛРОСА ао",
    "APTK": "Аптеки36и6",
    "ASTR": "Астра ао",
    "AFLT": "Аэрофлот",
    "BANE": "Башнефт ап",
    "BSPB": "БСП ао",
    "VSEH": "ВИ.ру",
    "VSMO": "ВСМПО-АВСМ",
    "VTBR": "ВТБ ао",
    "GAZP": "ГАЗПРОМ ао",
    "GMKN": "ГМКНорНик",
    "FESH": "ДВМП ао",
    "DVEC": "ДЭК ао",
    "LEAS": "Европлан",
    "EUTR": "ЕвроТранс",
    "ZAYM": "Займер ао",
    "AQUA": "ИНАРКТИКА",
    "IRAO": "ИнтерРАОао",
    "LENT": "Лента ао",
    "LSRG": "ЛСР ао",
    "LKOH": "ЛУКОЙЛ",
    "MVID": "М.видео",
    "MGNT": "Магнит ао",
    "MGTS": "МГТС-4ап",
    "MTLR": "Мечел ао",
    "MTLRP": "Мечел ап",
    "CBOM": "МКБ ао",
    "VKCO": "МКПАО ВК",
    "GEMC": "МКПАО ЮМГ",
    "MAGN": "ММК",
    "MOEX": "МосБиржа",
    "MBNK": "МТС Банк",
    "MTSS": "МТС-ао",
    "NKNC": "НКНХ ап",
    "NKHP": "НКХП ао",
    "NLMK": "НЛМК ао",
    "NMTP": "НМТП ао",
    "BELU": "НоваБев ао",
    "NVTK": "Новатэк ао",
    "OGKB": "ОГК-2 ао",
    "OZPH": "ОзонФарм",
    "KZOS": "ОргСинт ао",
    "KZOSP": "ОргСинт ап",
    "PIKK": "ПИК ао",
    "PLZL": "Полюс",
    "PRMD": "ПРОМОМЕД",
    "RASP": "Распадская",
    "RENI": "Ренессанс",
    "ROSN": "Роснефть",
    "FEES": "Россети",
    "MRKU": "Россети Ур",
    "MRKC": "РоссЦентр",
    "RTKM": "Ростел-ао",
    "RTKMP": "Ростел-ап",
    "MRKV": "РсетВол ао",
    "LSNGP": "РСетиЛЭ-п",
    "MSRS": "РСетиМР ао",
    "MRKP": "РСетиЦП ао",
    "RUAL": "РУСАЛ ао",
    "HYDR": "РусГидро",
    "RNFT": "РуссНфт ао",
    "SMLT": "Самолет ао",
    "SBER": "Сбербанк",
    "SBERP": "Сбербанк-п",
    "CHMF": "СевСт-ао",
    "SGZH": "Сегежа",
    "SELG": "Селигдар",
    "AFKS": "Система ао",
    "SVCB": "Совкомбанк",
    "FLOT": "Совкомфлот",
    "SVAV": "СОЛЛЕРС",
    "SPBE": "СПБ Биржа",
    "SNGS": "Сургнфгз",
    "SNGSP": "Сургнфгз-п",
    "T": "Т-Техно ао",
    "TATN": "Татнфт 3ао",
    "TATNP": "Татнфт 3ап",
    "TGKA": "ТГК-1",
    "TGKB": "ТГК-2",
    "TRMK": "ТМК ао",
    "TRNFP": "Транснф ап",
    "PHOR": "ФосАгро ао",
    "HEAD": "Хэдхантер",
    "HNFG": "ХЭНДЕРСОН",
    "ELFV": "ЭЛ5Энер ао",
    "SFIN": "ЭН+ГРУП ао",
    "ESFA": "ЭсЭфАй ао",
    "UGLD": "ЮГК",
    "UPRO": "Юнипро ао",
    "YDEX": "ЯНДЕКС"
    }
    
    # Создаем папки для отчетов
    if not os.path.exists('Data_storage/reports'):
        os.makedirs('Data Storage/reports')
    if not os.path.exists('Data Storage/reports/quarterly'):
        os.makedirs('Data Storage/reports/quarterly')
    if not os.path.exists('Data Storage/reports/yearly'):
        os.makedirs('Data Storage/reports/yearly')

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    # Скачиваем квартальные отчеты
    print("Скачиваем квартальные отчеты...")
    for ticker, name in companies.items():
        url = f"https://smart-lab.ru/q/{ticker}/f/q/MSFO/download/"
        filename = f"Data Storage/reports/quarterly/{ticker}_quarterly.csv"
        
        try:
            print(f"Скачиваем квартальный отчет {name} ({ticker})...")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Успешно сохранено в {filename}")
        except Exception as e:
            print(f"Ошибка при скачивании {ticker}: {str(e)}")
        
        sleep(2)  # Пауза между запросами

    # Скачиваем годовые отчеты
    print("\nСкачиваем годовые отчеты...")
    for ticker, name in companies.items():
        url = f"https://smart-lab.ru/q/{ticker}/f/y/MSFO/download/"
        filename = f"data/reports/yearly/{ticker}_yearly.csv"
        
        try:
            print(f"Скачиваем годовой отчет {name} ({ticker})...")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Успешно сохранено в {filename}")
        except Exception as e:
            print(f"Ошибка при скачивании {ticker}: {str(e)}")
        
        sleep(2)  # Пауза между запросами

    print("\nЗагрузка завершена!")

if __name__ == "__main__":
    download_reports()